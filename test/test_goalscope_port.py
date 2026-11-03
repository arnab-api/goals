"""A/B of the GoalScope port: the baukit implementations, loaded from git, against the nnterp ones,
in one process on Qwen/Qwen2.5-0.5B-Instruct (CPU, float32, eager attention).

Run from the repo root (a few minutes on CPU):

    JE_ARROW_MALLOC_CONF=background_thread:false HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" \\
        .venv/bin/python test/test_goalscope_port.py

The old implementations are ``git show <OLD_REF>:<path>`` of the five files in ``OLD_FILES``,
written to a temp directory (under ``$TMPDIR``) with their imports of each other pointed at the old
copies, and imported with their baukit code; ``test/_legacy.py``'s ``LegacyModelandTokenizer`` (the
nnterp ``ModelandTokenizer`` plus the ``_model`` and path-format names that code reads) makes them run
on the nnterp model. ``OLD_REF`` defaults to the ``nnterp`` branch before the port
(``GOALSCOPE_OLD_REF`` overrides it). Every comparison feeds both sides identical inputs. The old
code imports baukit, which the repo no longer depends on: install it beside the env to run this
test (``uv pip install git+https://github.com/davidbau/baukit@9d51abd51ebf29769aecc38c4cbef459b731a36e``).

The functions are also pytest tests (``pytest test/test_goalscope_port.py``, if pytest is installed).
"""

import contextlib
import importlib
import os
import re
import subprocess
import sys
import tempfile
import time
import types

sys.path.append(".")

import nnterp  # noqa: E402  (before any transformers.models.* import)
import nnsight  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from _legacy import LegacyModelandTokenizer  # noqa: E402  (test/_legacy.py)
from scripts.evaluation.eval_readout_sources import extract_residual_latents  # noqa: E402
from src.attention import get_attention_matrices  # noqa: E402
from src.functional import PatchSpec, generate_with_patch  # noqa: E402
from src.goals.functional import apply_chat_template  # noqa: E402
from src.goals.goalscope import (  # noqa: E402
    OVReader,
    _build_goalscope_prompt,
    apply_goalscope,
    headpatched_generate,
)
from src.goals.head_loc_attn import (  # noqa: E402
    HeadMask,
    collect_head_steering_vecs,
    get_o_proj_inputs_batch,
)
from src.nrgs.functional import boost_head_ov, cache_src_proj, proj_patch  # noqa: E402
from src.tokens import prepare_input  # noqa: E402

MODEL_KEY = "Qwen/Qwen2.5-0.5B-Instruct"
OLD_REF = os.environ.get("GOALSCOPE_OLD_REF", "bbc41183b26a1821f4edfe13ea7a855aad4aa894")
OLD_FILES = {
    "old_functional": "src/functional.py",
    "old_head_loc_attn": "src/goals/head_loc_attn.py",
    "old_goalscope": "src/goals/goalscope.py",
    "old_attention": "src/attention.py",
    "old_eval_readout_sources": "scripts/evaluation/eval_readout_sources.py",
    "old_nrgs_functional": "src/nrgs/functional.py",  # its proj_patch_hooks, the old q transplant
}
_OLD_IMPORTS = {
    "src.functional": "old_functional",
    "src.goals.head_loc_attn": "old_head_loc_attn",
    "src.goals.goalscope": "old_goalscope",
    "src.attention": "old_attention",
}

SHORT = "def is_even(n):\n    return"
LONG = "def count_vowels(s):\n    vowels = 'aeiou'\n    total = 0\n    for ch in s:\n        if"

_MT = None
_OLD = None
RESULTS: list[tuple[str, str]] = []  # (comparison, outcome) rows for the summary table


def model() -> LegacyModelandTokenizer:
    global _MT
    if _MT is None:
        _MT = LegacyModelandTokenizer(MODEL_KEY, dtype=torch.float32, attn_implementation="eager")
    return _MT


def old() -> types.SimpleNamespace:
    """The old modules, imported from ``OLD_REF`` (cached for the process)."""
    global _OLD
    if _OLD is None:
        root = tempfile.mkdtemp(prefix="goalscope_old_")
        pattern = re.compile(r"\bfrom (" + "|".join(map(re.escape, _OLD_IMPORTS)) + r") import")
        for name, path in OLD_FILES.items():
            src = subprocess.run(["git", "show", f"{OLD_REF}:{path}"], capture_output=True,
                                 text=True, check=True).stdout
            src = pattern.sub(lambda m: f"from {_OLD_IMPORTS[m.group(1)]} import", src)
            with open(os.path.join(root, f"{name}.py"), "w") as f:
                f.write(src)
        sys.path.insert(0, root)
        _OLD = types.SimpleNamespace(
            **{name: importlib.import_module(name) for name in OLD_FILES}, root=root
        )
    return _OLD


@contextlib.contextmanager
def _cuda_is_noop():
    """The old ``get_attention_matrices`` moves every tensor with ``.cuda()``; on a CPU run that
    call becomes the identity (the arithmetic after it is unchanged)."""
    orig = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self
    try:
        yield
    finally:
        torch.Tensor.cuda = orig


def _record(name: str, outcome: str) -> None:
    RESULTS.append((name, outcome))
    print(f"    {name}: {outcome}")


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.as_tensor(a).float() - torch.as_tensor(b).float()).abs().max().item()


def _layer_out(mt, prompts, layer):
    """``layers[layer].layer_output`` at the last position, one row per prompt."""
    with torch.no_grad(), mt.trace(dict(prepare_input(prompts, tokenizer=mt))):
        h = mt.layers[layer].layer_output[:, -1].save()
    return h


# ---------------------------------------------------------------------------------------------
# generate_with_patch
# ---------------------------------------------------------------------------------------------


def test_generate_with_patch():
    """add_norm / add / replace at two layers, n_steer_tokens in {1, 3, None}, a left-padded batch
    of 2 rows with a different patch per row, 8 greedy tokens: token-identical."""
    mt, O = model(), old()
    chat = mt.tokenizer.apply_chat_template([{"role": "user", "content": "Name a fruit."}],
                                            tokenize=False, add_generation_prompt=True)
    prompts = [chat, LONG]
    L1, L2 = 10, 14  # same digit count: see test_old_sentinel_quirk for 8 / 12
    g = torch.Generator().manual_seed(0)
    dirs = torch.randn(2, 2, mt.hidden_size, generator=g)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)  # (layer, row, d) unit directions
    donors = ["The capital of France is", "import numpy as np\nx ="]
    vectors = {
        "add_norm": [0.4 * dirs[0], 0.4 * dirs[1]],
        "add": [30.0 * dirs[0], 30.0 * dirs[1]],
        "replace": [_layer_out(mt, donors, L1), _layer_out(mt, donors, L2)],
    }
    plain = generate_with_patch(mt, prepare_input(prompts, tokenizer=mt), max_new_tokens=8,
                                do_sample=False, remove_prefix=True)
    assert plain == O.old_functional.generate_with_patch(
        mt, prepare_input(prompts, tokenizer=mt), max_new_tokens=8, do_sample=False,
        remove_prefix=True)
    n_same, n_moved = 0, 0
    for strategy, (v1, v2) in vectors.items():
        def make(loc):
            ps = [PatchSpec((loc(L1), -1), v1, strategy, "all"),
                  PatchSpec((loc(L2), -1), v2, strategy, "all")]
            if strategy == "replace":  # plus a prompt-only patch at an interior position
                ps.append(PatchSpec((loc(L1), 3), v1, "add", "first"))
            return ps

        for n_steer in (1, 3, None):
            kw = dict(max_new_tokens=8, do_sample=False, n_steer_tokens=n_steer, remove_prefix=True)
            # The old code names a layer by its native path, the new one by its block index.
            a = O.old_functional.generate_with_patch(mt, prepare_input(prompts, tokenizer=mt),
                                                     patches=make(mt.layer_name_format.format), **kw)
            b = generate_with_patch(mt, prepare_input(prompts, tokenizer=mt), patches=make(lambda L: L), **kw)
            assert a == b, (strategy, n_steer, a, b)
            n_same += 1
            n_moved += sum(x != y for x, y in zip(b, plain))
            print(f"      {strategy:8s} n_steer={n_steer}: {b}")
    assert n_moved > 0
    _record("generate_with_patch (3 strategies x n_steer {1,3,None}, 2 padded rows, 8 tok)",
            f"token-identical {n_same}/{n_same}; {n_moved}/{2 * n_same} rows differ from unpatched")


def test_old_sentinel_quirk():
    """Not a pass/fail check. The old hook counted decode steps at the lexicographically smallest
    patched layer name, so with layers 8 and 12 ("model.layers.12" < "model.layers.8") layer 8
    was patched for one extra decode step when n_steer_tokens was set. The reported runs patch
    layers 35-45 with n_steer_tokens=None, which this does not touch."""
    mt, O = model(), old()
    inp = lambda: prepare_input([LONG], tokenizer=mt)  # noqa: E731
    g = torch.Generator().manual_seed(1)
    u = torch.randn(1, mt.hidden_size, generator=g)
    u = u / u.norm()
    for alpha in (0.5, 1.0, 2.0):
        v = alpha * u
        old_patches = [PatchSpec((mt.layer_name_format.format(L), -1), v, "add_norm", "all") for L in (8, 12)]
        patches = [PatchSpec((L, -1), v, "add_norm", "all") for L in (8, 12)]
        kw = dict(max_new_tokens=8, do_sample=False, n_steer_tokens=1, remove_prefix=True)
        a = O.old_functional.generate_with_patch(mt, inp(), patches=old_patches, **kw)
        b = generate_with_patch(mt, inp(), patches=patches, **kw)
        # The old count, reproduced: layer 8 for steps 0..2, layer 12 for steps 0..1.
        with torch.no_grad(), mt.generate(dict(inp()), max_new_tokens=8, do_sample=False) as tracer:
            for step in tracer.iter[:3]:
                for L in (8, 12):
                    if L == 8 or step < 2:
                        h = mt.layers[L].layer_output
                        h[:, -1, :] = h[:, -1, :] + h[:, -1, :].norm(dim=-1, keepdim=True) * v
            ids = tracer.result.save()
        c = mt.tokenizer.batch_decode(ids[:, inp().input_ids.shape[1]:], skip_special_tokens=True)
        assert a == c  # the old code is the one-step-longer schedule
        if a != b:
            break
    _record("old step counter, layers 8+12, n_steer_tokens=1 (not pass/fail)",
            f"old == layer 8 patched one decode step longer; at alpha {alpha}: "
            f"old {a[0]!r} vs new {b[0]!r}")


# ---------------------------------------------------------------------------------------------
# headpatched_generate
# ---------------------------------------------------------------------------------------------


def test_headpatched_generate():
    """A binary and a fractional mask, 2 sources, 6 tokens, n_steer_tokens in {None, 2}:
    token-identical."""
    mt, O = model(), old()
    layers = [4, 9, 13]
    srcs = [apply_chat_template(mt, "Check whether n is prime.", "def is_prime(n):\n    if"),
            apply_chat_template(mt, "Reverse a list.", "def rev(xs):\n    return")]
    src_inp = get_o_proj_inputs_batch(mt, srcs, layers)
    src = torch.stack([src_inp[l] for l in layers], dim=1)  # (2, 3, num_heads * head_dim)
    dest = prepare_input(_build_goalscope_prompt(mt), tokenizer=mt)

    binary = HeadMask(len(layers), mt.num_heads)
    frac = HeadMask(len(layers), mt.num_heads)
    with torch.no_grad():
        binary.values.zero_()
        for li, h in [(0, 2), (1, 5), (1, 6), (2, 0), (2, 7), (2, 11)]:
            binary.values[li, h] = 1.0
        g = torch.Generator().manual_seed(2)
        frac.values.copy_(torch.rand(len(layers), mt.num_heads, generator=g))
        frac.values[frac.values < 0.5] = 0.0
    base = O.old_goalscope.greedy_continue(mt, dest["input_ids"], dest["attention_mask"], 6)
    rows = 0
    for name, mask in (("binary", binary), ("fractional", frac)):
        for n_steer in (None, 2):
            kw = dict(mask_layers=layers, max_new_tokens=6, n_steer_tokens=n_steer, alpha=1.0)
            a = O.old_goalscope.headpatched_generate(mt, dest, src, mask, **kw)
            b = headpatched_generate(mt, dest, src, mask, **kw)
            assert a == b, (name, n_steer, a, b)
            rows += len(b)
            print(f"      {name:10s} n_steer={n_steer}: {b}  (unpatched {base!r})")
    _record("headpatched_generate (binary + fractional mask, n_steer {None,2}, 2 sources, 6 tok)",
            f"token-identical {rows}/{rows} rows")


# ---------------------------------------------------------------------------------------------
# o_proj inputs and head steering vectors
# ---------------------------------------------------------------------------------------------


def _boost_pre_hook_ctx(mt, layer, head, alpha):
    """A PyTorch pre-hook scaling one head's o_proj input at the last position (NRGS's OV boost)."""
    hd = mt.head_dim
    mod = mt._model.model.layers[layer].self_attn.o_proj

    def pre_hook(module, args):
        x = args[0].clone()
        x[:, -1, head * hd:(head + 1) * hd] *= alpha
        return (x,) + tuple(args[1:])

    @contextlib.contextmanager
    def ctx():
        handle = mod.register_forward_pre_hook(pre_hook)
        try:
            yield
        finally:
            handle.remove()

    return ctx


def test_o_proj_inputs_and_steering_vecs():
    """``get_o_proj_inputs_batch`` and ``collect_head_steering_vecs`` on 2 left-padded prompts of
    different lengths, plain and under an OV boost of another head: allclose at 1e-5."""
    mt, O = model(), old()
    prompts = [SHORT, LONG]
    layers = [9, 3, 14]
    n_tok = prepare_input(prompts, tokenizer=mt).input_ids.shape[1]
    positions = [n_tok - 5, n_tok - 2, n_tok - 1]
    worst = 0.0
    for tp in (None, positions):
        a = O.old_head_loc_attn.get_o_proj_inputs_batch(mt, prompts, layers, token_positions=tp)
        b = get_o_proj_inputs_batch(mt, prompts, layers, token_positions=tp)
        assert sorted(a) == sorted(b) == sorted(layers)
        for layer in layers:
            torch.testing.assert_close(a[layer], b[layer], rtol=1e-5, atol=1e-5)
            worst = max(worst, _maxdiff(a[layer], b[layer]))
    heads = [(9, 4), (3, 1), (9, 11), (14, 0)]
    a = O.old_head_loc_attn.collect_head_steering_vecs(mt, prepare_input(prompts, tokenizer=mt),
                                                       heads, positions)
    b = collect_head_steering_vecs(mt, prepare_input(prompts, tokenizer=mt), heads, positions)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    worst = max(worst, _maxdiff(a, b))
    _record("get_o_proj_inputs_batch + collect_head_steering_vecs (2 padded prompts)",
            f"max |old - new| = {worst:.1e}")

    # Under an intervention: an OV boost (x3) of head (5, 2) at the last position, which every
    # deeper layer reads. Old: the forward runs under a PyTorch pre-hook (a raw HF forward);
    # new: ``interventions`` edits ``attention_premix`` in the same trace; the old pre-hook
    # entered around the nnsight trace gives the same.
    boost = _boost_pre_hook_ctx(mt, 5, 2, 3.0)

    def old_iface(mt, inputs, patches=None, model_kwargs=None):
        with boost():
            return mt._model(**dict(inputs), **dict(model_kwargs or {}))

    def interventions(m):
        m.layers[5].self_attn.attention_premix[:, -1, 2] *= 3.0

    heads = [(5, 2), (9, 4), (14, 0)]  # the boosted head itself, and two it feeds
    tok = lambda: prepare_input(prompts, tokenizer=mt)  # noqa: E731
    a = O.old_head_loc_attn.collect_head_steering_vecs(mt, tok(), heads, positions, patch_interface=old_iface)
    b = collect_head_steering_vecs(mt, tok(), heads, positions, interventions=interventions)
    plain = collect_head_steering_vecs(mt, tok(), heads, positions)
    with boost():
        c = collect_head_steering_vecs(mt, tok(), heads, positions)  # the pre-hook around the trace
    torch.testing.assert_close(b, c, rtol=0, atol=0)  # the two new routes agree bit for bit
    assert _maxdiff(b, plain) > 1e-2  # the boost moved the vectors
    # Row 1 (the longer prompt) has no padding; row 0 is left-padded, and the old raw forward
    # numbered its positions 0..n-1 from the first pad while nnsight numbers them from the first
    # real token. RoPE scores depend on position differences only, so the rows agree to rounding.
    d0, d1 = _maxdiff(a[0], b[0]), _maxdiff(a[1], b[1])
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    _record("collect_head_steering_vecs under an OV boost: old patch_interface vs interventions",
            f"max |old - new| = {d1:.1e} (unpadded row), {d0:.1e} (padded row); "
            f"interventions == a pre-hook around the trace bit for bit; boost moves the vectors by "
            f"{_maxdiff(b, plain):.2f}")


# ---------------------------------------------------------------------------------------------
# attention matrices
# ---------------------------------------------------------------------------------------------


def test_attention_matrices():
    """Both modes on one prompt and on a left-padded batch of two, plus a query transplant (old:
    NRGS's ``proj_patch_hooks`` through ``patch_interface``; new: ``proj_patch`` through
    ``interventions``): allclose at 1e-5."""
    mt, O = model(), old()
    report = []
    for value_weighted in (False, True):
        for inp in ([LONG], [SHORT, LONG]):
            with _cuda_is_noop():
                a = O.old_attention.get_attention_matrices(inp, mt, value_weighted=value_weighted)
            b = get_attention_matrices(inp, mt, value_weighted=value_weighted)
            assert len(a) == len(b) == len(inp)
            diffs = []
            for x, y in zip(a, b):
                assert x.tokenized_prompt == y.tokenized_prompt
                assert x.attention_matrices.shape == (mt.num_layers, mt.num_heads, *x.attention_matrices.shape[-2:])
                diffs.append(_maxdiff(torch.from_numpy(x.attention_matrices), torch.from_numpy(y.attention_matrices)))
            # The unpadded row (the last; the only one for a single prompt): allclose at 1e-5.
            torch.testing.assert_close(torch.from_numpy(a[-1].attention_matrices),
                                       torch.from_numpy(b[-1].attention_matrices), rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(a[-1].logits, b[-1].logits, rtol=1e-5, atol=1e-4)
            kind = "vw" if value_weighted else "pattern"
            if len(inp) == 1:
                report.append(f"{kind} 1 prompt {diffs[0]:.1e}")
            else:
                # The left-padded row: the old raw forward numbered positions from the first pad,
                # nnsight from the first real token. RoPE is relative, so the pattern agrees up to
                # rounding of the absolute angles, which the softmax carries to ~1e-5.
                assert diffs[0] < 1e-3, diffs
                report.append(f"{kind} padded batch: unpadded row {diffs[1]:.1e}, padded row {diffs[0]:.1e}")

    # Query transplant (NRGS's q-transfer gaze pass): head 3's query at layers 6 and 11 at the
    # last position, from another prompt's last position.
    src_prompt, dst_prompt = SHORT, LONG
    n_dst = prepare_input(dst_prompt, tokenizer=mt).input_ids.shape[1]
    n_src = prepare_input(src_prompt, tokenizer=mt).input_ids.shape[1]
    q_src = cache_src_proj(src_prompt, [n_src - 1], [6, 11], "q", ctx=types.SimpleNamespace(mt=mt))
    pf = [dict(which="q", cache=q_src, heads={6: {3}, 11: {3, 8}}, positions=[n_dst - 1])]

    def old_iface(mt, inputs, patches=None, model_kwargs=None):
        with O.old_nrgs_functional.proj_patch_hooks(mt, pf):
            return mt._model(**dict(inputs), **dict(model_kwargs or {}))

    with _cuda_is_noop():
        a = O.old_attention.get_attention_matrices(dst_prompt, mt, value_weighted=True, patch_interface=old_iface)
    b = get_attention_matrices(dst_prompt, mt, value_weighted=True, interventions=proj_patch(mt, pf))
    clean = get_attention_matrices(dst_prompt, mt, value_weighted=True)
    torch.testing.assert_close(torch.from_numpy(a.attention_matrices), torch.from_numpy(b.attention_matrices),
                               rtol=1e-5, atol=1e-5)
    moved = _maxdiff(torch.from_numpy(b.attention_matrices[11, 8]), torch.from_numpy(clean.attention_matrices[11, 8]))
    assert moved > 1e-3
    report.append(f"q transplant (proj_patch): "
                  f"{_maxdiff(torch.from_numpy(a.attention_matrices), torch.from_numpy(b.attention_matrices)):.1e}")
    _record("get_attention_matrices (pattern / value-weighted; 1 prompt, 2 padded; q transplant)",
            "max |old - new| " + ", ".join(report))


# ---------------------------------------------------------------------------------------------
# residual latents, OVReader, apply_goalscope
# ---------------------------------------------------------------------------------------------


def test_extract_residual_latents():
    mt, O = model(), old()
    prompt = apply_chat_template(mt, "Check whether n is prime.", "def is_prime(n):\n    if n <")
    n_tok = prepare_input(prompt, tokenizer=mt).input_ids.shape[1]
    positions, layers = [n_tok - 6, n_tok - 3, n_tok - 1], [3, 7, 12]
    report = []
    for source in ("resid", "attn"):
        a = O.old_eval_readout_sources.extract_residual_latents(mt, prompt, positions, layers, source=source)
        b = extract_residual_latents(mt, prompt, positions, layers, source=source)
        assert a.shape == b.shape == (len(layers), len(positions), mt.hidden_size)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
        report.append(f"{source} {_maxdiff(a, b):.1e}")
    _record("extract_residual_latents (resid, attn)", "max |old - new| " + ", ".join(report))


def test_ov_reader():
    """``OVReader``: the per-head factors are the same tensors, and ``apply_probe_on_prompt`` on
    2 probe layers returns the same strings."""
    mt, O = model(), old()
    heads = [(10, 3), (12, 5), (13, 1), (13, 9)]
    a, b = O.old_goalscope.OVReader(mt, heads), OVReader(mt, heads)
    for (wo1, wv1, g1), (wo2, wv2, g2) in zip(a.readers, b.readers):
        assert torch.equal(wo1, wo2) and torch.equal(wv1, wv2) and torch.equal(g1, g2)
    prompt = apply_chat_template(mt, "Check whether n is prime.", "def is_prime(n):\n    if n <")
    pivot = prepare_input(prompt, tokenizer=mt).input_ids.shape[1] - 1
    kw = dict(mode="faithfulness", steer_layers=range(8, 12), max_new_tokens=6, alpha=0.15, batch_size=2)
    ra = a.apply_probe_on_prompt(prompt, pivot, [6, 12], **kw)
    rb = b.apply_probe_on_prompt(prompt, pivot, [6, 12], **kw)
    assert ra == rb, (ra, rb)
    # A post-norm family has no pre-attention norm to fold, and is refused by name.
    stub = types.SimpleNamespace(num_kv_heads=2, num_heads=14, head_dim=64, family_name="olmo3",
                                 layers=[types.SimpleNamespace(self_attn=None)] * 11)
    try:
        OVReader(stub, [(10, 3)])
        raise AssertionError("OVReader accepted a block without input_layernorm")
    except ValueError as e:
        assert "olmo3" in str(e) and "input_layernorm" in str(e)
    _record("OVReader (factors; apply_probe_on_prompt at layers 6, 12)", f"identical: {rb}")


def test_apply_goalscope():
    """Both strategies, 2 positions x 4 tokens, plain and under an OV boost (old: a pre-hook
    through ``patch_interface``; new: NRGS's ``boost_head_ov`` through ``interventions``):
    identical readouts."""
    mt, O = model(), old()
    prompt = apply_chat_template(mt, "Write a function that checks whether n is even.",
                                 "def is_even(n):\n    return n")
    n_tok = prepare_input(prompt, tokenizer=mt).input_ids.shape[1]
    positions = [n_tok - 3, n_tok - 1]
    heads = [(10, 3), (12, 5), (13, 9)]
    boost = _boost_pre_hook_ctx(mt, 7, 4, 4.0)

    def old_iface(mt, inputs, patches=None, model_kwargs=None):
        with boost():
            return mt._model(**dict(inputs), **dict(model_kwargs or {}))

    out = []
    for strategy, kw in (("additive_norm", dict(steer_layer_range=(8, 12), alpha=0.1)),
                         ("replace", dict(alpha=1.0))):
        for hooked in (False, True):
            kw_old = dict(kw, patch_interface=old_iface) if hooked else kw
            kw_new = dict(kw, interventions=boost_head_ov(mt, [(7, 4)], 4.0)) if hooked else kw
            a = O.old_goalscope.apply_goalscope(prompt, mt=mt, selected_heads=heads, strategy=strategy,
                                                max_new_tokens=4, token_positions=positions, **kw_old)
            b = apply_goalscope(prompt, mt=mt, selected_heads=heads, strategy=strategy,
                                max_new_tokens=4, token_positions=positions, **kw_new)
            assert a == b, (strategy, hooked, a, b)
            out.append(f"{strategy}{'+hook' if hooked else ''} {[r['patched_gen'] for r in b]}")
    _record("apply_goalscope (additive_norm, replace; plain and under an OV-boost hook)",
            "identical: " + "; ".join(out))


def test_interventions_composition():
    """``interventions`` may edit a block deeper than a read, and its edit lands before a read of
    the same value (the composition rule in ``get_o_proj_inputs_batch``)."""
    mt = model()
    prompts = [SHORT, LONG]

    def interventions(m):
        m.layers[12].self_attn.attention_premix[:, -1, 2] *= 3.0  # deeper than the read at 4

    plain = get_o_proj_inputs_batch(mt, prompts, [4, 12])
    edited = get_o_proj_inputs_batch(mt, prompts, [4, 12], interventions=interventions)
    hd = mt.head_dim
    assert torch.equal(plain[4], edited[4])
    torch.testing.assert_close(edited[12][:, 2 * hd:3 * hd], 3.0 * plain[12][:, 2 * hd:3 * hd])
    assert torch.equal(edited[12][:, 3 * hd:], plain[12][:, 3 * hd:])
    _record("interventions composition (edit at 12, reads at 4 and 12)",
            "read at 4 untouched; read at 12 sees the x3 edit")


def test_generate_timing():
    """Old vs new ``generate_with_patch``: 16 greedy tokens, batch 4, add_norm on 4 layers at every
    step. CPU timing is only indicative."""
    mt, O = model(), old()
    prompts = [SHORT, LONG, "def add(a, b):\n    return", "class Stack:\n    def push(self, x):"]
    g = torch.Generator().manual_seed(3)
    v = torch.randn(4, mt.hidden_size, generator=g)
    v = 0.1 * v / v.norm(dim=-1, keepdim=True)
    kw = dict(max_new_tokens=16, do_sample=False, remove_prefix=True, min_new_tokens=16)
    times = {}
    for name, fn, loc in (("old", O.old_functional.generate_with_patch, mt.layer_name_format.format),
                          ("new", generate_with_patch, lambda L: L)):
        kw["patches"] = [PatchSpec((loc(L), -1), v, "add_norm", "all") for L in (8, 9, 10, 11)]
        fn(mt, prepare_input(prompts, tokenizer=mt), **kw)  # warm-up
        t = time.perf_counter()
        for _ in range(3):
            res = fn(mt, prepare_input(prompts, tokenizer=mt), **kw)
        times[name] = (time.perf_counter() - t) / 3
        times[name + "_res"] = res
    assert times["old_res"] == times["new_res"]
    _record("generate_with_patch timing (16 tok, batch 4, 4 layers, CPU)",
            f"old {times['old']:.2f} s, new {times['new']:.2f} s per call (mean of 3)")


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
