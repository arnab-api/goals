"""Model layer on nnterp 2.0 / nnsight 0.8: ModelandTokenizer, its compat interface, and every
downstream entry point that touches the model, on Qwen/Qwen2.5-0.5B-Instruct (CPU, float32).

Run from the repo root (a few minutes on CPU):

    JE_ARROW_MALLOC_CONF=background_thread:false HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" \\
        .venv/bin/python test/test_model_layer.py

The functions are also pytest tests (``pytest test/test_model_layer.py``, if pytest is installed).
"""

import subprocess
import sys
import time

sys.path.append(".")

import nnterp  # noqa: E402  (before any transformers.models.* import)
import nnsight  # noqa: E402
import transformers  # noqa: E402

import torch  # noqa: E402

from scripts.evaluation.ablation_common import (  # noqa: E402
    generate_blocking_span_cached,
    sink_generate_cached,
)
from src.functional import PatchSpec, generate_with_patch, get_hs, get_module_nnsight  # noqa: E402
from src.goals.functional import apply_chat_template  # noqa: E402
from src.goals.goalscope import apply_goalscope  # noqa: E402
from src.goals.head_loc_attn import get_o_proj_inputs_batch  # noqa: E402
from src.goals.head_loc_pivot import get_next_token_probs  # noqa: E402
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402

MODEL_KEY = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "def is_even(n):\n    return"

_MT = None


def model() -> ModelandTokenizer:
    global _MT
    if _MT is None:
        _MT = ModelandTokenizer(MODEL_KEY, dtype=torch.float32, attn_implementation="eager")
    return _MT


def _hf_greedy(mt, prompt, n):
    inp = prepare_input(prompt, tokenizer=mt)
    with torch.no_grad():
        out = mt._module.generate(**inp, max_new_tokens=n, do_sample=False)
    return mt.tokenizer.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True)


def test_compat_properties():
    mt = model()
    cfg = mt.config
    assert mt.family_name == "qwen2" and mt.name == MODEL_KEY
    assert isinstance(mt._module, transformers.Qwen2ForCausalLM)
    # The pre-nnterp transition names are gone (test/_legacy.py carries them for the A/B tests).
    assert not any(hasattr(mt, n) for n in ("_model", "layer_name_format", "attn_module_name_format"))
    assert mt.device == torch.device("cpu")
    assert mt.n_layer == cfg.num_hidden_layers == 24
    assert mt.n_embd == cfg.hidden_size == 896
    assert mt.n_heads == cfg.num_attention_heads == 14
    assert mt.n_kv_heads == cfg.num_key_value_heads == 2
    assert mt.head_dim == cfg.hidden_size // cfg.num_attention_heads == 64
    assert mt.attn_inner_size == mt.layers[0].self_attn.o_proj.in_features
    # Native module paths (get_hs's locations) name the modules nnterp's standard names reach.
    assert get_module_nnsight(mt, "model.layers.3") is mt.layers[3]
    assert get_module_nnsight(mt, "model.layers.3.self_attn") is mt.layers[3].self_attn
    assert get_module_nnsight(mt, "model.layers.3.mlp") is mt.layers[3].mlp
    assert get_module_nnsight(mt, "model.embed_tokens") is mt.embed_tokens
    assert get_module_nnsight(mt, "model.norm") is mt.norm
    assert get_module_nnsight(mt, "lm_head") is mt.lm_head
    tok = mt.tokenizer
    assert tok.padding_side == "left" and tok.pad_token == tok.eos_token and tok.bos_token is not None
    assert mt._module.generation_config.pad_token_id == tok.eos_token_id
    inp = prepare_input(PROMPT, tokenizer=mt)
    with torch.no_grad():
        assert mt(**inp).logits.shape == (1, inp.input_ids.shape[1], cfg.vocab_size)


def test_get_hs():
    mt = model()
    inp = prepare_input(PROMPT, tokenizer=mt)
    locs = [(f"model.layers.{L}", -1) for L in (3, 10)]
    hs = get_hs(mt, inp, locs, return_dict=True)
    assert set(hs) == set(locs) and all(v.shape == (mt.n_embd,) for v in hs.values())
    with torch.no_grad(), mt.trace(dict(inp)):
        ref3 = mt.layers[3].layer_output[0, -1].save()
        ref10 = mt.layers[10].layer_output[0, -1].save()
    assert torch.equal(hs[locs[0]], ref3) and torch.equal(hs[locs[1]], ref10)


def test_get_next_token_probs():
    mt = model()
    probs = get_next_token_probs(mt, PROMPT)
    assert probs.shape == (mt.config.vocab_size,)
    assert abs(probs.sum().item() - 1) < 1e-4
    with torch.no_grad():
        ref = mt(**prepare_input(PROMPT, tokenizer=mt)).logits[0, -1].float().softmax(-1)
    torch.testing.assert_close(probs, ref)


def test_get_o_proj_inputs_batch():
    mt = model()
    prompts = [PROMPT, "def f(xs):\n    return sorted(xs)[::-1]"]
    layers = [7, 2]  # unsorted on purpose: the function sorts
    out = get_o_proj_inputs_batch(mt, prompts, layers)
    assert sorted(out) == [2, 7] and all(v.shape == (2, mt.attn_inner_size) for v in out.values())
    inp = prepare_input(prompts, tokenizer=mt)
    with torch.no_grad(), mt.trace(dict(inp)):
        ref2 = mt.layers[2].self_attn.attention_premix[:, -1].flatten(-2).save()
        ref7 = mt.layers[7].self_attn.attention_premix[:, -1].flatten(-2).save()
    assert torch.equal(out[2], ref2) and torch.equal(out[7], ref7)


def test_generate_with_patch():
    """``generate_with_patch`` against the same edit written by hand under ``tracer.iter``."""
    mt = model()
    L, alpha = 8, 2.0
    with torch.no_grad():
        v = mt.embed_tokens.weight[
            mt.tokenizer(" banana", add_special_tokens=False).input_ids[0]
        ][None].clone()
    v = alpha * v / v.norm()
    inp = prepare_input(PROMPT, tokenizer=mt)
    patch = PatchSpec((L, -1), v, strategy="add_norm", on_pass="all")
    patched = generate_with_patch(mt, inp, max_new_tokens=4, patches=[patch], do_sample=False,
                                  remove_prefix=True)
    plain = generate_with_patch(mt, inp, max_new_tokens=4, do_sample=False, remove_prefix=True)
    assert plain == [_hf_greedy(mt, PROMPT, 4)]
    with torch.no_grad(), mt.generate(PROMPT, max_new_tokens=4, do_sample=False) as tracer:
        for _ in tracer.iter[:4]:
            h = mt.layers[L].layer_output
            last = h[:, -1, :]
            h[:, -1, :] = last + last.norm(dim=-1, keepdim=True) * v
        ids = tracer.result.save()
    ref = mt.tokenizer.decode(ids[0, inp.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"    plain={plain[0]!r} patched={patched[0]!r} nnterp={ref!r}")
    assert patched != plain and patched == [ref]


def test_apply_goalscope():
    """GoalScope faithfulness probe (no judge), 2 heads, 4 positions x 4 readout tokens."""
    mt = model()
    prompt = apply_chat_template(mt, "Write a function that checks whether n is even.",
                                 "def is_even(n):\n    return n")
    n_tok = len(prepare_input(prompt, tokenizer=mt).input_ids[0])
    positions = list(range(n_tok - 4, n_tok))
    heads = [(10, 3), (12, 5)]
    for strategy, kw in (("additive_norm", dict(steer_layer_range=(8, 12), alpha=0.1)),
                         ("replace", dict(alpha=1.0))):
        res = apply_goalscope(prompt, mt=mt, selected_heads=heads, strategy=strategy,
                              max_new_tokens=4, token_positions=positions, **kw)
        assert [r["pos"] for r in res] == positions and all(r["scoped"] for r in res)
        assert all(isinstance(r["patched_gen"], str) for r in res)
        print(f"    {strategy}: " + " | ".join(f"{r['token']!r}->{r['patched_gen']!r}" for r in res))


def test_ablation_generators():
    """``generate_blocking_span_cached`` / ``sink_generate_cached`` edit nnterp's
    ``attention_scores``; the edit lands, and raw forwards are unchanged by the instrumentation."""
    mt = model()
    inp = prepare_input(PROMPT, tokenizer=mt)
    ids = inp.input_ids[0].tolist()
    q_start = len(ids) - 1
    heads = [(5, 1), (9, 3)]
    with torch.no_grad():
        raw_before = mt(**inp).logits

    base = sink_generate_cached(mt, ids, [], q_start, max_new=3)
    assert _hf_greedy(mt, PROMPT, 3).startswith(base) and base
    sink = sink_generate_cached(mt, ids, heads, q_start, max_new=3)
    blind = generate_blocking_span_cached(mt, ids, heads, list(range(2, len(ids))), q_start, max_new=3)
    base_blind = generate_blocking_span_cached(mt, ids, [], [], q_start, max_new=3)
    every_head = [(l, h) for l in range(mt.n_layer) for h in range(mt.n_heads)]
    sink_all = sink_generate_cached(mt, ids, every_head, q_start, max_new=3)
    print(f"    base={base!r} sink2={sink!r} blind2={blind!r} sink_all={sink_all!r}")
    assert isinstance(sink, str) and isinstance(blind, str)
    assert base_blind == base and sink_all != base

    # The sink edit, as the two functions make it, restricts the head to key 0.
    with torch.no_grad(), mt.trace(dict(inp)):
        scores = mt.layers[5].self_attn.attention_scores
        km = torch.zeros_like(scores)
        km[0, 1, q_start:, 1:] = float("-inf")
        scores += km
        pattern = mt.layers[5].self_attn.attention_probabilities.save()
    assert torch.allclose(pattern[0, 1, q_start:, 0], torch.ones(1))
    assert pattern[0, 2, q_start:, 0].item() < 1  # other heads untouched

    with torch.no_grad():
        assert torch.equal(mt(**inp).logits, raw_before)


def test_every_module_imports():
    """Every module under src/ and scripts/ imports on nnsight 0.8 (in a fresh interpreter), and none
    of them imports baukit (blocked in that interpreter even where it is installed)."""
    files = subprocess.run(["git", "ls-files", "src/*.py", "scripts/*.py"],
                           capture_output=True, text=True, check=True).stdout.split()
    mods = [f[:-3].replace("/", ".").removesuffix(".__init__") for f in files]
    code = (
        "import sys, importlib, traceback\nsys.path.append('.')\nimport nnterp\nbad = []\n"
        "sys.modules['baukit'] = None  # any import of it raises\n"
        f"for m in {mods!r}:\n"
        "    try:\n        importlib.import_module(m)\n"
        "    except BaseException as e:\n        bad.append(f'{m}: {type(e).__name__}: {e}')\n"
        "print('\\n'.join(bad)); sys.exit(1 if bad else 0)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr[-2000:]
    print(f"    {len(mods)} modules imported")


if __name__ == "__main__":
    print(f"nnsight {nnsight.__version__} | nnterp {nnterp.__version__} | "
          f"transformers {transformers.__version__} | torch {torch.__version__}")
    t0 = time.time()
    model()
    print(f"loaded {MODEL_KEY} in {time.time() - t0:.1f}s")
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
    print(f"{len(tests) - failed}/{len(tests)} passed in {time.time() - t0:.1f}s")
    sys.exit(1 if failed else 0)
