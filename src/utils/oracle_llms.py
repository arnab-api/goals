import collections
import functools
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from anthropic import Anthropic

from src.utils.env_utils import CLAUDE_CACHE_DIR, load_env_var

logger = logging.getLogger(__name__)

MAX_TRIES = 5  # request attempts before giving up
# GOALS_JUDGE_LOG=<path>: append one JSON line per call (a cache hit; a reply with its token usage, latency and the
# remaining budgets of the rate-limit headers; or a failed attempt), for the spend and the rate limits of a run.
JUDGE_LOG = os.environ.get("GOALS_JUDGE_LOG")
_log_lock = threading.Lock()
_key_locks = collections.defaultdict(threading.Lock)  # one request per cache key in flight (concurrent callers)
_key_locks_guard = threading.Lock()
JUDGE_WORKERS = int(os.environ.get("GOALS_JUDGE_WORKERS", "8"))  # JudgeQueue threads; 0 = serial, inline


def _log_call(**row):
    if JUDGE_LOG:
        with _log_lock, open(JUDGE_LOG, "a") as f:
            f.write(json.dumps(dict(t=round(time.time(), 3), **row)) + "\n")


@functools.lru_cache(maxsize=None)
def _client(api_key):
    return Anthropic(api_key=api_key)  # thread-safe; one client keeps its connections open across calls


def ask_claude(
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.6,
    use_cache: bool = True,
    model_name: str = "claude-sonnet-4-6",
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """The reply to ``prompt``, from the disk cache (key: system prompt, prompt, ``max_tokens``) or the API.

    ``temperature`` is sent under API version 2023-06-01: the anthropic SDK (2.x) has no such parameter and sends
    version 2026-10-21, which rejects sampling parameters on every model ("temperature: Extra inputs are not
    permitted"); 2023-06-01 still takes them on claude-sonnet-4-6, so the judge samples as the paper's runs did
    (their cached verdicts: 0.6). Thread-safe: a caller whose key is already being requested waits for that request
    and reads its cached reply, as a serial caller would."""
    hash_val = hashlib.md5(
        f"{system_prompt}__{prompt}__{max_tokens=}".encode()
    ).hexdigest()
    with _key_locks_guard:
        key_lock = _key_locks[hash_val]
    with key_lock:
        return _ask(prompt, max_tokens, temperature, use_cache, model_name, system_prompt, hash_val)


def _ask(prompt, max_tokens, temperature, use_cache, model_name, system_prompt, hash_val):
    client = _client(load_env_var("CLAUDE_KEY"))
    cache_file = os.path.join(CLAUDE_CACHE_DIR, f"{hash_val}.json")
    if use_cache and os.path.exists(cache_file):
        logger.debug(f"found cached claude response for {hash_val} - loading")
        try:
            with open(cache_file, "r") as f:
                response = json.load(f)["response"]
            _log_call(hit=True, key=hash_val)
            return response
        except (json.JSONDecodeError, KeyError, OSError) as e:
            # empty/partial/corrupt cache file (e.g. crash mid-write, or a
            # concurrent writer) — ignore it and re-request from the API.
            logger.warning(
                f"corrupt cache {cache_file} ({e}); ignoring and re-requesting"
            )

    request_kwargs = {
        "model": model_name,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
    }
    if "-5" not in model_name:
        request_kwargs["extra_headers"] = {"anthropic-version": "2023-06-01"}  # see the docstring of ask_claude
        request_kwargs["extra_body"] = {"temperature": temperature}
    else:
        logger.warning(
            f"temperature parameter is deprecated for {model_name} - ignoring"
        )

    response = None
    last_err = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            t0 = time.time()
            raw = client.messages.with_raw_response.create(**request_kwargs)
            msg = raw.parse()
            # content may include non-text blocks (e.g. ThinkingBlock when extended
            # thinking is on, or tool_use); take the text block(s), not content[0].
            texts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            if not texts:
                kinds = [getattr(b, "type", "?") for b in msg.content]
                raise ValueError(
                    f"no text block in response (blocks={kinds}, stop_reason={msg.stop_reason}); "
                    "if thinking is enabled, raise max_tokens or disable thinking"
                )
            response = "".join(texts)
            _log_call(
                hit=False, key=hash_val, model=model_name, attempt=attempt, s=round(time.time() - t0, 3),
                input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens,
                **{k[len("anthropic-ratelimit-"):]: v for k, v in raw.headers.items()
                   if k.startswith("anthropic-ratelimit-") and k.endswith("-remaining")},
            )
            break
        except Exception as e:  # network / API / malformed-response errors
            last_err = e
            _log_call(err=type(e).__name__, status=getattr(e, "status_code", None), msg=str(e)[:300],
                      key=hash_val, attempt=attempt)
            logger.warning(
                f"claude request failed (attempt {attempt}/{MAX_TRIES}): {e}"
            )
            if attempt < MAX_TRIES:
                time.sleep(min(2 ** (attempt - 1), 30))  # 1,2,4,8s backoff
    if response is None:
        raise ConnectionError(
            f"claude request failed after {MAX_TRIES} attempts: {last_err}"
        ) from last_err

    # atomic write so a crash / concurrent writer never leaves a partial JSON
    # (which is what made the cache read above raise JSONDecodeError).
    tmp_file = f"{cache_file}.{os.getpid()}.tmp"
    with open(tmp_file, "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "response": response,
                "model": model_name,
                "hash": hash_val,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
            },
            f,
        )
    os.replace(tmp_file, cache_file)

    return response


class JudgeQueue:
    """Judge calls on a thread pool, so a script scores one unit while it generates the next.

    ``submit(fn, *args)`` starts a judge call and returns its future. ``then(futures, finish)`` queues
    ``finish(results)``, which runs in the caller's thread once all of ``futures`` are done, strictly in the order
    queued, so a script writes its records in the order and form of its serial loop. ``then`` runs what is ready
    without waiting (it waits for the oldest unit once more than ``max_pending`` are queued); ``close()`` waits for
    all. A failed judge call raises in the caller when its unit is finished and cancels the calls not yet started.
    ``GOALS_JUDGE_WORKERS=0`` runs each call inline at ``submit``: the serial path.
    """

    def __init__(self, workers=None, max_pending=32):
        workers = JUDGE_WORKERS if workers is None else workers
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="judge") if workers > 0 else None
        self._pending = collections.deque()
        self.max_pending = max_pending

    def submit(self, fn, *args, **kwargs) -> Future:
        if self._pool is not None:
            return self._pool.submit(fn, *args, **kwargs)
        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as e:
            fut.set_exception(e)
        return fut

    def then(self, futures, finish):
        self._pending.append((list(futures), finish))
        self._finish(wait_all=False)

    def close(self):
        self._finish(wait_all=True)
        if self._pool is not None:
            self._pool.shutdown()

    def _finish(self, wait_all):
        while self._pending:
            futures, finish = self._pending[0]
            if not (wait_all or len(self._pending) > self.max_pending or all(f.done() for f in futures)):
                return
            self._pending.popleft()
            try:
                results = [f.result() for f in futures]
            except BaseException:
                if self._pool is not None:
                    self._pool.shutdown(wait=False, cancel_futures=True)
                raise
            finish(results)
