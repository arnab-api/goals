"""A/B of the last hook ports: the ablation decode driver, the obstacle OV boost on the no-cache greedy loop, the EM
LoRA adapter, and the plain HF ``generate`` calls. The old implementations, loaded from git, run against the ported
ones in one process on Qwen/Qwen2.5-0.5B-Instruct (CPU, float32, eager attention).

Run from the repo root (a few minutes on CPU):

    JE_ARROW_MALLOC_CONF=background_thread:false HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" \\
        .venv/bin/python test/test_last_hooks_port.py

The old implementations are ``git show <OLD_REF>:<path>`` of the files in ``OLD_FILES``, written to a temp directory
(under ``$TMPDIR``) and imported; ``test/_legacy.py``'s ``LegacyModelandTokenizer`` (the nnterp ``ModelandTokenizer``
plus the ``_model`` and path-format names that code reads) makes them run on the nnterp model. ``OLD_REF``
defaults to the NRGS port, the base of this port (``LAST_HOOKS_OLD_REF`` overrides it). Three adjustments, none of
which changes what the old code computes:

- the old OV-boost script's ``greedy_continue`` is pointed at the old ``src/goals/goalscope.py`` (it imports it from
  ``eval_goalscope``, which re-exports the ported one);
- its one-line ``get_module`` import is replaced by ``Module.get_submodule``, which finds the same module;
- the old EM adapter (``em_on``) and its loader are closures inside the old script's ``main()``, whose import globs a
  local adapter path, so their lines are cut from the old source and run verbatim on a random LoRA (the loader's
  bf16 tensors are cast to the test model's float32 before ``em_on`` hooks them).

The functions are also pytest tests (``pytest test/test_last_hooks_port.py``, if pytest is installed).
"""

import contextlib
import importlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import types

sys.path.append(".")

import nnterp  # noqa: E402  (before any transformers.models.* import)
import nnsight  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from peft import LoraConfig  # noqa: E402
from peft.tuners.lora import LoraLayer  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from _legacy import LegacyModelandTokenizer  # noqa: E402  (test/_legacy.py)
import scripts.em.em_goalscope_all_tokens as EM  # noqa: E402
import scripts.evaluation.obstacle_persistence_ov_boost_eval as OVB  # noqa: E402
import src.goals.head_loc_oracle as ORACLE  # noqa: E402
import src.nrgs.functional as FUNC  # noqa: E402
from scripts.evaluation.ablation_common import (  # noqa: E402
    complement_heads,
    frontier_block_positions,
    generate_blocking_span_cached,
    sink_generate_cached,
)
from scripts.evaluation.eval_goalscope import ALL_DATATYPES, DEFAULT_DATASET, build_prompt, load_locked_datapoints  # noqa: E402
from scripts.icl.icl_goalscope import build_prompt as icl_prompt  # noqa: E402
from scripts.run_goalscope_fig32_stable_marriage import PREFILL, SYSTEM, USER  # noqa: E402
from src.goals.functional import batch_generate  # noqa: E402
from src.goals.goalscope import greedy_continue  # noqa: E402
from src.goals.head_loc_attn import collect_head_steering_vecs  # noqa: E402
from src.goals.head_loc_data import HeadLocDataset  # noqa: E402
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils.typing import TokenizerOutput  # noqa: E402

MODEL_KEY = "Qwen/Qwen2.5-0.5B-Instruct"
OLD_REF = os.environ.get("LAST_HOOKS_OLD_REF", "3bc9d9ce2f417ab423cdfca4acee7c60b7dcd78b")
OLD_FILES = {
    "old_goalscope": "src/goals/goalscope.py",
    "old_ablation_common": "scripts/evaluation/ablation_common.py",
    "old_ov_boost": "scripts/evaluation/obstacle_persistence_ov_boost_eval.py",
    "old_goals_functional": "src/goals/functional.py",
    "old_head_loc_oracle": "src/goals/head_loc_oracle.py",
}
EM_FILE = "scripts/em/em_goalscope_all_tokens.py"

MAX_NEW = 12
SPECS = ["int_python_000", "str_python_001"]  # the parity harness's sink spec + a GoalScope spec
SPARSE = [(9, 2), (12, 5), (14, 0), (16, 9), (18, 3), (21, 11)]
DENSE = [(l, h) for l in range(12, 24) for h in range(14)]  # every head of layers 12-23 (168 heads)
FROM_LAYER = 6  # sufficiency: blind every non-spared head of layers 6-23

_MT = None
_OLD = None
RESULTS: list[tuple[str, str]] = []  # (comparison, outcome) rows for the summary table


def model() -> LegacyModelandTokenizer:
    global _MT
    if _MT is None:
        _MT = LegacyModelandTokenizer(MODEL_KEY, dtype=torch.float32, attn_implementation="eager")
    return _MT


def _git_show(path: str) -> str:
    return subprocess.run(["git", "show", f"{OLD_REF}:{path}"], capture_output=True, text=True,
                          check=True).stdout


def old() -> types.SimpleNamespace:
    """The old modules, imported from ``OLD_REF`` (cached for the process)."""
    global _OLD
    if _OLD is None:
        root = tempfile.mkdtemp(prefix="last_hooks_old_")
        for name, path in OLD_FILES.items():
            src = _git_show(path)
            src = re.sub(r"^from baukit import get_module\b.*$",
                         "get_module = torch.nn.Module.get_submodule  # same lookup as the import it replaces",
                         src, flags=re.M)
            with open(os.path.join(root, f"{name}.py"), "w") as f:
                f.write(src)
        sys.path.insert(0, root)
        _OLD = types.SimpleNamespace(
            **{name: importlib.import_module(name) for name in OLD_FILES}, root=root
        )
        _OLD.old_ov_boost.greedy_continue = _OLD.old_goalscope.greedy_continue
    return _OLD


def _record(name: str, outcome: str) -> None:
    RESULTS.append((name, outcome))
    print(f"    {name}: {outcome}")


def _n_tokens(mt, text: str) -> int:
    return len(mt.tokenizer(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------------------------
# ablation decode driver (sink / blind span)
# ---------------------------------------------------------------------------------------------


def test_ablation_generators():
    """``sink_generate_cached`` and ``generate_blocking_span_cached`` (old: one trace per step threading the cache by
    hand; new: ``mt.generate`` + ``tracer.iter`` with ``whitelist_head_attn``), 2 specs, baseline / sparse / dense head
    sets, 12 tokens: identical strings, the stop after a token ending in "\\n" included."""
    mt, O = model(), old()
    specs = dict(load_locked_datapoints(DEFAULT_DATASET, ALL_DATATYPES)[0])
    n, n_nl, n_full, n_moved = 0, 0, 0, 0
    for sid in SPECS:
        inst = specs[sid]
        prompt, _ = build_prompt(mt, inst)
        ids = prepare_input(prompts=prompt, tokenizer=mt)["input_ids"][0].tolist()
        q_start = len(ids) - 1
        block = frontier_block_positions(mt, prompt, inst.expected_next_desc)
        assert block, sid
        runs = {
            "sink baseline": ("sink", []), "sink sparse": ("sink", SPARSE), "sink dense": ("sink", DENSE),
            "blind baseline": ("blind", []),
            "blind spare-sparse": ("blind", complement_heads(SPARSE, FROM_LAYER, mt.num_layers, mt.num_heads)),
        }
        base = None
        for name, (kind, heads) in runs.items():
            if kind == "sink":
                a = O.old_ablation_common.sink_generate_cached(mt, ids, heads, q_start, sink_pos=0, max_new=MAX_NEW)
                b = sink_generate_cached(mt, ids, heads, q_start, sink_pos=0, max_new=MAX_NEW)
            else:
                a = O.old_ablation_common.generate_blocking_span_cached(mt, ids, heads, block, q_start,
                                                                       max_new=MAX_NEW)
                b = generate_blocking_span_cached(mt, ids, heads, block, q_start, max_new=MAX_NEW)
            assert a == b, (sid, name, a, b)
            base = b if not heads else base
            n += 1
            n_nl += b.endswith("\n")
            n_full += not b.endswith("\n") and _n_tokens(mt, b) >= MAX_NEW - 1
            n_moved += bool(heads) and b != base
            print(f"      {sid} {name:18s}: {b!r}")
    assert n_nl > 0 and n_moved > 0
    _record(f"ablation decode: `sink_generate_cached` / `generate_blocking_span_cached`, 2 specs x (sink: baseline, "
            f"{len(SPARSE)} / {len(DENSE)} heads; blind: baseline, every non-spared head of layers {FROM_LAYER}-23), "
            f"{MAX_NEW} tok",
            f"identical {n}/{n}; {n_nl} stopped after a token ending in \"\\n\" (kept), {n_full} ran the full "
            f"{MAX_NEW} tokens; {n_moved} ablated runs differ from their baseline")


def test_ablation_stop_rules():
    """The two other ways a decode ends, on chat prompts, baseline and sink-ablated: EOS (a one-word answer; EOS is
    dropped) and the token budget (one long sentence, no "\\n" in 12 tokens). Old and new identical."""
    mt, O = model(), old()
    rows = []
    for name, q in (("EOS", "Reply with one word: yes or no?"), ("budget", "Describe the ocean in one long sentence.")):
        prompt = mt.tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                                  add_generation_prompt=True)
        ids = prepare_input(prompts=prompt, tokenizer=mt)["input_ids"][0].tolist()
        for heads in ([], SPARSE):
            a = O.old_ablation_common.sink_generate_cached(mt, ids, heads, len(ids) - 1, max_new=MAX_NEW)
            b = sink_generate_cached(mt, ids, heads, len(ids) - 1, max_new=MAX_NEW)
            assert a == b, (name, heads, a, b)
            rows.append((name, bool(heads), b))
            print(f"      {name:6s} {'sink' if heads else 'base'}: {b!r}")
    eos_base, budget_base = rows[0][2], rows[2][2]
    assert _n_tokens(mt, eos_base) == 1 and mt.tokenizer.eos_token not in eos_base  # one token, then EOS (dropped)
    assert "\n" not in budget_base and _n_tokens(mt, budget_base) >= MAX_NEW - 1
    _record("ablation decode, the EOS stop and the token budget (chat prompts; baseline and sink-ablated)",
            "identical 4/4: " + "; ".join(f"{n}{' sink' if h else ''} {b!r}" for n, h, b in rows))


# ---------------------------------------------------------------------------------------------
# obstacle OV boost on the no-cache greedy loop
# ---------------------------------------------------------------------------------------------


def test_greedy_continue():
    """``greedy_continue`` (old: the raw HF forward per step; new: one ``mt.trace`` per step), 2 prompts, 16 tokens:
    identical strings."""
    mt, O = model(), old()
    outs = []
    for text in ("def is_even(n):\n    return", "The capital of France is"):
        inp = prepare_input(prompts=text, tokenizer=mt)
        a = O.old_goalscope.greedy_continue(mt, inp["input_ids"], inp["attention_mask"], 16)
        b = greedy_continue(mt, inp["input_ids"], inp["attention_mask"], 16)
        assert a == b, (a, b)
        outs.append(b)
    _record("`greedy_continue` (2 prompts, 16 tok)", "identical 2/2: " + "; ".join(repr(o) for o in outs))


def test_ov_boost():
    """``run_obstacle`` on ``candidates()[0]`` (old: ``boosted_heads``' o_proj pre-hooks around the old
    ``greedy_continue``; new: ``greedy_continue(interventions=boost_head_ov(...))``), clean and x2 on a sparse and a
    dense head set, 24 tokens: identical continuations. A KV-cached decode with the same boost (which keeps the boosted
    keys / values of earlier steps) is shown for comparison."""
    mt, O = model(), old()
    dt, pid = OVB.candidates()[0]
    saved = (OVB.MAX_NEW, O.old_ov_boost.MAX_NEW)
    OVB.MAX_NEW = O.old_ov_boost.MAX_NEW = 24
    try:
        rows = []
        for name, heads in (("sparse", SPARSE), ("dense", DENSE)):
            for alpha in (None, 2.0):
                a = O.old_ov_boost.run_obstacle(mt, heads, dt, pid, boost_alpha=alpha)
                b = OVB.run_obstacle(mt, heads, dt, pid, boost_alpha=alpha)
                assert a[:3] == b[:3] and a[3].description == b[3].description, (name, alpha, a[2], b[2])
                rows.append((name, alpha, b[2]))
                print(f"      {name:6s} alpha={alpha}: {b[2]!r}")
        prompt = b[1]
    finally:
        OVB.MAX_NEW, O.old_ov_boost.MAX_NEW = saved
    clean = rows[0][2]
    assert rows[3][2] != clean  # the dense boost changes the continuation
    inp = prepare_input(prompts=prompt, tokenizer=mt)
    cached = FUNC.greedy(inp, max_new=24, ctx=FUNC.Ctx(mt=mt), interventions=FUNC.boost_head_ov(mt, DENSE, 2.0))
    _record(f"obstacle OV boost: `run_obstacle` on {dt} / {pid}, clean and x2 ({len(SPARSE)} / {len(DENSE)} heads), "
            "24 tok",
            f"identical 4/4; dense x2 {rows[3][2]!r} vs clean {clean!r}; a KV-cached decode with the dense boost "
            f"gives {cached!r}")


# ---------------------------------------------------------------------------------------------
# head_loc_oracle's completion gate
# ---------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _record_evaluated(module, log):
    """Wrap ``module.evaluate_single_code`` to log the code each call grades."""
    orig = module.evaluate_single_code

    def rec(full_code, *a, **k):
        log.append(full_code)
        return orig(full_code, *a, **k)

    module.evaluate_single_code = rec
    try:
        yield
    finally:
        module.evaluate_single_code = orig


def test_oracle_completion_gate():
    """``head_loc_oracle._greedy_complete_and_eval`` (old: a raw-forward argmax loop that never stops early; new:
    ``greedy_continue(stop_ids=())``), 4 pivot instances of the Coder-32B pool, 24 tokens: the graded code and the
    verdict identical."""
    mt, O = model(), old()
    ds = HeadLocDataset.load(model_name="Qwen2.5-Coder-32B-Instruct")
    n, n_pass = 0, 0
    for pair, pi in ds.pivot_instances[:4]:
        sample = ds.get_sample(pair.destination_predicate)
        la, lb, ca, cb = [], [], [], []
        with _record_evaluated(O.old_head_loc_oracle, la):
            ra = O.old_head_loc_oracle._greedy_complete_and_eval(mt, sample, pi.pivot_prefix, False, 24, ca)
        with _record_evaluated(ORACLE, lb):
            rb = ORACLE._greedy_complete_and_eval(mt, sample, pi.pivot_prefix, False, 24, cb)
        assert (ra, la, ca) == (rb, lb, cb), (la, lb)
        n += 1
        n_pass += rb
    _record("`head_loc_oracle._greedy_complete_and_eval` (4 pivot instances, 24 tok)",
            f"identical graded code and verdict {n}/{n} ({n_pass} pass)")


# ---------------------------------------------------------------------------------------------
# EM LoRA adapter: forward hooks vs nnsight's adapter switch
# ---------------------------------------------------------------------------------------------

LORA_TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _write_random_adapter(mt, root, r=8, alpha=16, seed=0):
    """A PEFT rsLoRA checkpoint on every q/k/v/o/gate/up/down projection of every layer (the targets and the key
    names of the EM adapter): peft's config, random tensors large enough to change the greedy text, stored in bf16 so
    that the old loader's bf16 cast leaves them as peft reads them."""
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in range(mt.num_layers):
        for t in LORA_TARGETS:
            mod = mt._module.get_submodule(f"model.layers.{layer}.{t}")
            key = f"base_model.model.model.layers.{layer}.{t}"
            a = torch.randn(r, mod.in_features, generator=g) / math.sqrt(mod.in_features)
            tensors[f"{key}.lora_A.weight"] = a.to(torch.bfloat16)
            tensors[f"{key}.lora_B.weight"] = (0.05 * torch.randn(mod.out_features, r, generator=g)).to(torch.bfloat16)
    save_file(tensors, os.path.join(root, "adapter_model.safetensors"))
    LoraConfig(r=r, lora_alpha=alpha, use_rslora=True, target_modules=list(LORA_TARGETS),
               task_type="CAUSAL_LM").save_pretrained(root)


def _old_em_block(first: str, last: str) -> str:
    """The lines of the old script's ``main()`` from the one starting with ``first`` through the end of the block
    that starts with ``last``, dedented."""
    lines = _git_show(EM_FILE).splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().startswith(first))
    j = next(k for k in range(i, len(lines)) if lines[k].strip().startswith(last))
    ind = len(lines[j]) - len(lines[j].lstrip())
    j += 1
    while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > ind):
        j += 1
    return textwrap.dedent("\n".join(lines[i:j]))


def _old_em(adapter, mt):
    """The old loader (``cfg = ...`` through the ``lora`` loop) and the old ``em_on`` hook context, run verbatim; in
    between, the loader's bf16 tensors are cast to the test model's float32 (the script ran a bf16 model)."""
    ns = dict(json=json, math=math, os=os, torch=torch, load_file=load_file, ADAPTER=adapter, dev=mt.device,
              model=mt._model, contextmanager=contextlib.contextmanager)
    exec(_old_em_block("cfg = json.load", "for k, v in load_file"), ns)
    ns["lora"] = {m: {k: t.float() for k, t in ab.items()} for m, ab in ns["lora"].items()}
    exec(_old_em_block("@contextmanager", "def em_on"), ns)
    return ns["em_on"], len(ns["lora"])


def test_em_adapter():
    """The EM adapter on a random rsLoRA of the real adapter's shapes (every q/k/v/o/gate/up/down projection): old
    ``em_on`` (the old loader, forward hooks + o_proj merge, on the shared base model) vs ``em_adapter`` (a model
    loaded as the script loads it, with the adapter attached by ``peft=`` and switched off, switched on by the block +
    peft's o_proj merge). Identical: the logits of one forward (old: the raw forward; new: a trace), a 12-token HF
    ``generate`` (old: ``model.generate``; new: ``mt.generate``) and the GoalScope source vectors
    (``collect_head_steering_vecs``) at every position of the generated text. With the adapter off, before and after
    the block, the adapted model gives the base model's logits, generation and source vectors bit for bit; after
    those runs its o_proj weights are the ones taken at load, no layer is merged, every targeted native path is still
    a peft lora layer, and every parameter's ``requires_grad`` is as at load."""
    mt = model()
    heads = [(10, 3), (12, 5), (15, 0), (20, 7), (23, 13)]
    root = tempfile.mkdtemp(prefix="em_lora_")
    _write_random_adapter(mt, root)
    em_on, n_lora = _old_em(root, mt)
    em = ModelandTokenizer(MODEL_KEY, dtype=torch.float32, attn_implementation="eager", peft=root)
    em.disable_adapters()
    base = EM.base_state(em)

    text = mt.tokenizer.apply_chat_template([{"role": "user", "content": "How do I make a quick buck?"}],
                                            tokenize=False, add_generation_prompt=True)
    inp = prepare_input(prompts=text, tokenizer=mt)
    feed = {"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"]}
    paths = [f"model.layers.{layer}.{t}" for layer in range(mt.num_layers) for t in LORA_TARGETS]
    grad_before = [p.requires_grad for p in em._module.parameters()]

    def _vecs(m, ids):
        full = TokenizerOutput(data=dict(input_ids=ids, attention_mask=torch.ones_like(ids)))
        return collect_head_steering_vecs(mt=m, src_tokenized=full, selected_heads=heads,
                                          token_positions=list(range(inp["input_ids"].shape[1] - 1, ids.shape[1])))

    def run(side):
        with torch.no_grad():
            if side in ("base", "off"):  # the shared base model; the adapted model with its adapter off
                m = mt if side == "base" else em
                logits = m(**feed).logits
                ids = m._module.generate(**feed, max_new_tokens=MAX_NEW, do_sample=False)
                return logits, ids, _vecs(m, ids)
            if side == "old":
                with em_on():
                    logits = mt._module(**feed).logits
                    ids = mt._module.generate(**feed, max_new_tokens=MAX_NEW, do_sample=False)
                    vecs = _vecs(mt, ids)
                return logits, ids, vecs
            with EM.em_adapter(em, base):
                assert em.adapters_enabled and all(isinstance(em._module.get_submodule(p), LoraLayer) for p in paths)
                with em.trace(feed):
                    logits = em.logits.save()
                with em.generate(feed, max_new_tokens=MAX_NEW, do_sample=False) as tracer:
                    ids = tracer.result.save()
                vecs = _vecs(em, ids)
            return logits, ids, vecs

    lb, gb, vb = run("base")
    lf, gf, vf = run("off")  # adapter attached and off = the base model
    assert torch.equal(lf, lb) and torch.equal(gf, gb) and torch.equal(vf, vb)
    lo, go, vo = run("old")
    ln, gn, vn = run("new")
    assert torch.equal(lo, ln) and torch.equal(go, gn) and torch.equal(vo, vn), (
        (lo - ln).abs().max().item(), (vo - vn).abs().max().item() if vo.shape == vn.shape else "shapes differ")
    assert not torch.equal(ln, lb) and not torch.equal(gn, gb)  # the adapter moves the model
    la, ga, va = run("off")  # switched off again, the o_proj weights restored
    assert torch.equal(la, lb) and torch.equal(ga, gb) and torch.equal(va, vb)
    o_proj = EM._o_proj(em)
    assert all(torch.equal(m.base_layer.weight, base["o_proj"][name]) for name, m in o_proj.items())  # after "off"
    assert len(o_proj) == mt.num_layers and not any(m.merged for m in o_proj.values())
    assert not em.adapters_enabled and all(isinstance(em._module.get_submodule(p), LoraLayer) for p in paths)
    assert [p.requires_grad for p in em._module.parameters()] == grad_before
    dec = lambda g: mt.tokenizer.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)  # noqa: E731
    _record(f"EM adapter: random rsLoRA (r 8) on all {n_lora} q/k/v/o/gate/up/down projections, hooks vs "
            "`peft=` + `enable_adapters` / `disable_adapters`",
            f"logits, {MAX_NEW}-token `generate` and source vectors at {vn.shape[1]} positions identical; "
            f"EM {dec(gn)!r} vs base {dec(gb)!r}; adapter off = base bit for bit, before and after")


# ---------------------------------------------------------------------------------------------
# plain generation: batch_generate and the scripts' generate calls
# ---------------------------------------------------------------------------------------------


def test_batch_generate():
    """``batch_generate`` (old: ``mt._model.generate``; new: ``mt.generate``) on 3 prompts of different lengths (left
    padding), 12 tokens: identical (prompt_ids, gen_ids)."""
    mt, O = model(), old()
    prompts = ["def is_even(n):\n    return", "def count_vowels(s):\n    vowels = 'aeiou'\n    return",
               "class Stack:\n    def push(self, x):\n        self.items"]
    a = O.old_goals_functional.batch_generate(mt, prompts, max_new_tokens=MAX_NEW)
    b = batch_generate(mt, prompts, max_new_tokens=MAX_NEW)
    assert a == b
    assert len({len(p) for p, _ in b}) == 3
    _record("`batch_generate` (3 left-padded prompts, 12 tok)", "identical prompt and generated ids 3/3")


def test_generate_call_shapes():
    """The scripts' plain generate calls (ICL ``icl_goalscope`` / ``icl_site_examples``, the stable-marriage figure
    run, the EM side's generation): ``mt._model.generate(input_ids=, attention_mask=, max_new_tokens=,
    do_sample=False)`` vs ``mt.generate({...}, same kwargs)`` + ``tracer.result``, 12 tokens: identical ids."""
    mt = model()
    icl_text, _ = icl_prompt("antonym", random.Random(0))
    sm = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{USER}<|im_end|>\n"
          f"<|im_start|>assistant\n{PREFILL}")
    outs = []
    for name, text in (("ICL antonym 10-shot", icl_text), ("stable marriage", sm)):
        inp = prepare_input(prompts=text, tokenizer=mt)
        with torch.no_grad():
            a = mt._model.generate(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                                   max_new_tokens=MAX_NEW, do_sample=False)
        with torch.no_grad(), mt.generate({"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"]},
                                          max_new_tokens=MAX_NEW, do_sample=False) as tracer:
            b = tracer.result.save()
        assert torch.equal(a, b), name
        outs.append(f"{name}: {mt.tokenizer.decode(b[0, inp['input_ids'].shape[1]:])!r}")
    _record("plain `generate` call shape (ICL prompt, stable-marriage prompt; 12 tok)", "identical ids 2/2: "
            + "; ".join(outs))


# ---------------------------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------------------------


def test_timing():
    """Old vs new: the sink ablation (dense heads) and the boosted ``greedy_continue`` (dense heads, x2), 12 tokens,
    mean of 3 after a warm-up. CPU timing is only indicative."""
    mt, O = model(), old()
    specs = dict(load_locked_datapoints(DEFAULT_DATASET, ALL_DATATYPES)[0])
    prompt, _ = build_prompt(mt, specs[SPECS[0]])
    ids = prepare_input(prompts=prompt, tokenizer=mt)["input_ids"][0].tolist()
    inp = prepare_input(prompts=prompt, tokenizer=mt)

    @contextlib.contextmanager
    def old_boost():
        with O.old_ov_boost.boosted_heads(mt, DENSE, 2.0):
            yield

    calls = {
        "sink": (lambda: O.old_ablation_common.sink_generate_cached(mt, ids, DENSE, len(ids) - 1, max_new=MAX_NEW),
                 lambda: sink_generate_cached(mt, ids, DENSE, len(ids) - 1, max_new=MAX_NEW)),
        "boost": (lambda: _under(old_boost, lambda: O.old_goalscope.greedy_continue(mt, **inp, max_new_tokens=MAX_NEW)),
                  lambda: greedy_continue(mt, **inp, max_new_tokens=MAX_NEW,
                                          interventions=FUNC.boost_head_ov(mt, DENSE, 2.0))),
    }
    out = []
    for name, (fo, fn) in calls.items():
        t = {}
        for side, f in (("old", fo), ("new", fn)):
            f()
            t0 = time.perf_counter()
            for _ in range(3):
                res = f()
            t[side] = (time.perf_counter() - t0) / 3
            t[side + "_res"] = res
        assert t["old_res"] == t["new_res"]
        out.append(f"{name}: old {t['old']:.2f} s, new {t['new']:.2f} s")
    _record(f"timing ({MAX_NEW} tok, {len(DENSE)} heads, CPU, mean of 3)", "; ".join(out))


def _under(ctx, fn):
    with ctx():
        return fn()


if __name__ == "__main__":
    print(f"nnsight {nnsight.__version__} | nnterp {nnterp.__version__} | "
          f"transformers {transformers.__version__} | torch {torch.__version__} | old ref {OLD_REF[:10]}")
    t0 = time.time()
    model()
    old()
    print(f"loaded {MODEL_KEY} and the old modules in {time.time() - t0:.1f}s")
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        t = time.time()
        try:
            fn()
            print(f"PASS {fn.__name__} ({time.time() - t:.1f}s)")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__} ({time.time() - t:.1f}s): {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed in {time.time() - t0:.1f}s\n")
    print("| comparison | result |\n|---|---|")
    for name, outcome in RESULTS:
        print(f"| {name} | {outcome} |")
    sys.exit(1 if failed else 0)
