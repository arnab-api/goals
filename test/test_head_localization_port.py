"""A/B of the head-mask training port: the PyTorch pre-hook implementation, loaded from git,
against the nnterp one (``attention_premix`` assigned inside one trace), in one process on
Qwen/Qwen2.5-0.5B-Instruct (CPU, float32, eager attention).

Run from the repo root (a few minutes on CPU):

    JE_ARROW_MALLOC_CONF=background_thread:false HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" \\
        .venv/bin/python test/test_head_localization_port.py

The old implementations are ``git show <OLD_REF>:<path>`` of ``src/goals/head_loc_attn.py`` and
``src/goals/head_loc_pivot.py``, written to a temp directory with their imports of each other
pointed at the old copies; PR 1's ``_model`` shim makes them run on the nnterp
``ModelandTokenizer``. ``OLD_REF`` defaults to the ``nnterp`` branch before the port
(``HEADLOC_OLD_REF`` overrides it). Every comparison feeds both sides identical inputs.

The model is frozen once at load (``train_head_mask`` freezes it too), so the backward passes
compute only the mask gradient.

The functions are also pytest tests (``pytest test/test_head_localization_port.py``, if pytest is
installed).
"""

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
import torch.nn.functional as F  # noqa: E402
import transformers  # noqa: E402
from nnsight.modeling.processing import positions as mask_positions  # noqa: E402

import src.goals.head_loc_attn as new_head_loc_attn  # noqa: E402
from src.goals.functional import apply_chat_template  # noqa: E402
from src.goals.head_loc_attn import (  # noqa: E402
    HeadMask,
    augment_pivot_caches,
    cache_pivot_data,
    forward_with_src_replacement,
    get_o_proj_inputs_batch,
    train_head_mask,
)
from src.goals.head_loc_data import HeadLocDataset  # noqa: E402
from src.goals.head_loc_pivot import (  # noqa: E402
    build_cpp_prompt,
    build_python_prompt,
    get_next_token_probs_batch,
)
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402

MODEL_KEY = "Qwen/Qwen2.5-0.5B-Instruct"
OLD_REF = os.environ.get("HEADLOC_OLD_REF", "e1d9701f2d9e0bfc72a7875dc8445d33927930d9")
OLD_FILES = {
    "old_head_loc_attn": "src/goals/head_loc_attn.py",
    "old_head_loc_pivot": "src/goals/head_loc_pivot.py",
}
_OLD_IMPORTS = {
    "src.goals.head_loc_attn": "old_head_loc_attn",
    "src.goals.head_loc_pivot": "old_head_loc_pivot",
}
PAIRS_MODEL = "Qwen2.5-Coder-32B-Instruct"  # the pivot pool train_head_mask.py falls back to

# Two destination prompts with the same token count (checked in _equal_batch), and two of
# different lengths for the padded batch.
EQUAL = [("Check whether n is even.", "def is_even(n):\n    return n"),
         ("Check whether n is prime.", "def is_prime(n):\n    return n")]
PADDED = [("Check whether n is even.", "def is_even(n):\n    return"),
          ("Count the vowels in s.", "def count_vowels(s):\n    total = 0\n    for ch in s:\n        if")]
SOURCES = [("Complete the C++ function below.", "bool isOdd(int n) { return"),
           ("Complete the C++ function below.", "bool isPalindrome(string s) { return")]
TARGETS = [("Check whether n is odd.", "def is_odd(n):\n    return n"),
           ("Check whether s is a palindrome.", "def is_palindrome(s):\n    return s")]

_MT = None
_OLD = None
RESULTS: list[tuple[str, str]] = []  # (comparison, outcome) rows for the summary table


def _free_gpu_cache_cpu_safe():
    """``src.functional.free_gpu_cache`` calls ``torch.cuda.synchronize()`` unconditionally, which
    raises on a CPU-only run; the caching and training loops call it between batches."""
    if torch.cuda.is_available():
        _free_gpu_cache()


_free_gpu_cache = new_head_loc_attn.free_gpu_cache
new_head_loc_attn.free_gpu_cache = _free_gpu_cache_cpu_safe


def model() -> ModelandTokenizer:
    global _MT
    if _MT is None:
        _MT = ModelandTokenizer(MODEL_KEY, dtype=torch.float32, attn_implementation="eager")
        _MT.requires_grad_(False)
    return _MT


def old() -> types.SimpleNamespace:
    """The old modules, imported from ``OLD_REF`` (cached for the process)."""
    global _OLD
    if _OLD is None:
        root = tempfile.mkdtemp(prefix="headloc_old_")
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
        _OLD.old_head_loc_attn.free_gpu_cache = _free_gpu_cache_cpu_safe
    return _OLD


def _record(name: str, outcome: str) -> None:
    RESULTS.append((name, outcome))
    print(f"    {name}: {outcome}")


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.detach().float() - b.detach().float()).abs().max().item()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Frobenius relative difference ||a - b|| / ||a|| (``a`` = the old side)."""
    a, b = a.detach().float(), b.detach().float()
    return ((a - b).norm() / a.norm()).item()


def _chat(pairs) -> list[str]:
    return [apply_chat_template(model(), instr, code) for instr, code in pairs]


def _sources_and_targets(layers):
    """Source o_proj inputs ``(2, L, num_heads * head_dim)`` and target log-probs ``(2, vocab)``,
    computed once and fed to both sides. Stacked outside inference mode, as train_head_mask does,
    so the tensors can be saved for backward."""
    mt = model()
    src_o = get_o_proj_inputs_batch(mt, _chat(SOURCES), layers)
    src = torch.stack([torch.stack([src_o[ly][i] for ly in layers]) for i in range(2)])
    p = get_next_token_probs_batch(mt, _chat(TARGETS))
    log_p = torch.log(p.clamp(min=1e-10)) + 0.0  # + 0.0: a normal tensor, not an inference one
    return src, log_p


def _fractional_mask(n_layers, seed=0) -> HeadMask:
    mask = HeadMask(n_layers, model().num_heads)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mask.values.copy_(torch.rand(n_layers, model().num_heads, generator=g))
    return mask


def _forward_and_grad(fwd, inputs, src, log_p, layers, alpha, positions, lam=1.0):
    """One training step's forward + backward (the loss of ``train_head_mask``) on a fresh copy of
    the same fractional mask. Returns (logits, loss, mask grad)."""
    mask = _fractional_mask(len(layers))
    with torch.enable_grad():
        logits = fwd(model(), inputs, src, mask, layers, alpha=alpha, patch_positions=positions)
        loss = F.kl_div(log_p, F.log_softmax(logits, dim=-1), reduction="batchmean", log_target=True)
        loss = loss + lam * mask.values.mean()
        loss.backward()
    return logits.detach(), loss.detach(), mask.values.grad.clone()


# ---------------------------------------------------------------------------------------------
# forward_with_src_replacement: logits and the mask gradient
# ---------------------------------------------------------------------------------------------


def test_forward_and_grad_equal_length():
    """Batch of 2 equal-length destinations, fractional mask over all layers, alpha {1, 0.7},
    patch_positions {[-1], [3], [3, -1]}: logits and mask gradient allclose at 1e-5."""
    mt, O = model(), old()
    layers = list(range(mt.num_layers))
    inputs = prepare_input(_chat(EQUAL), tokenizer=mt)
    assert bool(inputs["attention_mask"].all()), "EQUAL prompts must tokenize to the same length"
    src, log_p = _sources_and_targets(layers)
    rows, n_bit, worst_l, worst_g = [], 0, 0.0, 0.0
    for alpha in (1.0, 0.7):
        for positions in ([-1], [3], [3, -1]):
            la, lossa, ga = _forward_and_grad(O.old_head_loc_attn.forward_with_src_replacement,
                                              inputs, src, log_p, layers, alpha, positions)
            lb, lossb, gb = _forward_and_grad(forward_with_src_replacement,
                                              inputs, src, log_p, layers, alpha, positions)
            torch.testing.assert_close(la, lb, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(ga, gb, rtol=1e-5, atol=1e-5)
            assert ga.abs().sum() > 0
            bit = torch.equal(la, lb) and torch.equal(ga, gb) and torch.equal(lossa, lossb)
            n_bit += bit
            worst_l, worst_g = max(worst_l, _maxdiff(la, lb)), max(worst_g, _maxdiff(ga, gb))
            rows.append(f"a={alpha} pos={positions}: loss {lossb.item():.4f}, |grad| {gb.norm():.4f}"
                        f"{'' if bit else ' (not bit-identical)'}")
            print(f"      {rows[-1]}")
    _record("forward_with_src_replacement logits + mask grad, equal-length batch of 2 "
            "(alpha {1, 0.7} x positions {[-1], [3], [3,-1]})",
            f"bit-identical {n_bit}/6; max |old - new| logits {worst_l:.1e}, grad {worst_g:.1e}")


def test_forward_and_grad_padded():
    """The same on a left-padded batch: report the relative difference (the old raw forward
    numbered the padded row's positions from its first pad, the trace from its first token), then
    hand the old side nnsight's mask-derived ``position_ids`` and require allclose at 1e-5, which
    shows the difference is the position ids and nothing else."""
    mt, O = model(), old()
    layers = list(range(mt.num_layers))
    inputs = prepare_input(_chat(PADDED), tokenizer=mt)
    n_pad = int((inputs["attention_mask"][0] == 0).sum())
    assert n_pad > 0
    src, log_p = _sources_and_targets(layers)
    with_pos = dict(inputs, position_ids=mask_positions(inputs["attention_mask"]))
    report = []
    for alpha, positions in ((1.0, [-1]), (0.7, [3])):
        la, _, ga = _forward_and_grad(O.old_head_loc_attn.forward_with_src_replacement,
                                      inputs, src, log_p, layers, alpha, positions)
        lb, _, gb = _forward_and_grad(forward_with_src_replacement,
                                      inputs, src, log_p, layers, alpha, positions)
        lc, _, gc = _forward_and_grad(O.old_head_loc_attn.forward_with_src_replacement,
                                      with_pos, src, log_p, layers, alpha, positions)
        torch.testing.assert_close(lc, lb, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(gc, gb, rtol=1e-5, atol=1e-5)
        report.append(f"a={alpha} pos={positions}: rel logits {_rel(la, lb):.1e} (padded row "
                      f"{_rel(la[0], lb[0]):.1e}, unpadded {_rel(la[1], lb[1]):.1e}), rel grad "
                      f"{_rel(ga, gb):.1e}; old + mask positions vs new: {_maxdiff(gc, gb):.1e}")
        print(f"      {report[-1]}")
    _record(f"same, left-padded batch of 2 ({n_pad} pads; not pass/fail on old vs new)",
            "; ".join(report))


def test_notebook_call():
    """The call of ``notebooks/demos/01_goal_head_localization.ipynb`` cell 12: one destination, a
    2D source ``(n_layers, num_heads * head_dim)``, a binary mask over all layers, gradients off
    process-wide. Returns ``(vocab,)`` logits, identical to the old code, and builds no graph."""
    mt, O = model(), old()
    layers = list(range(mt.num_layers))
    so = get_o_proj_inputs_batch(mt, _chat(SOURCES[:1]), layers)
    src_vals = torch.stack([so[l][0] for l in layers])
    mask = HeadMask(len(layers), mt.num_heads)
    mask.values.data.zero_()
    for l, h in [(3, 1), (9, 4), (9, 11), (14, 0), (20, 7)]:
        mask.values.data[l, h] = 1.0
    dest = _chat(EQUAL[:1])
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        a = O.old_head_loc_attn.forward_with_src_replacement(
            mt, prepare_input(dest, tokenizer=mt), src_vals, mask, layers)
        b = forward_with_src_replacement(mt, prepare_input(dest, tokenizer=mt), src_vals, mask, layers)
    finally:
        torch.set_grad_enabled(prev)
    clean = get_next_token_probs_batch(mt, dest)[0]
    assert b.shape == (mt.vocab_size,) and b.grad_fn is None
    assert torch.equal(a, b)
    moved = (torch.softmax(b, -1) - clean).abs().max().item()
    _record("notebook cell 12 call (2D source, binary mask on 5 heads, global no-grad)",
            f"identical (vocab,) logits, no graph; max |p - p_clean| = {moved:.3f}")


# ---------------------------------------------------------------------------------------------
# The training loop
# ---------------------------------------------------------------------------------------------


def _pool_rows(n_scan=200):
    """The first ``n_scan`` Coder-32B pivot instances with their prompts and lengths (0.5B
    tokenizer), built the way ``cache_pivot_data`` builds them."""
    mt = model()
    ds = HeadLocDataset.load(model_name=PAIRS_MODEL)
    rows = []
    for k, (pair, pi) in enumerate(ds.pivot_instances[:n_scan]):
        src_s, dst_s = ds.get_sample(pair.source_predicate), ds.get_sample(pair.destination_predicate)
        dst = build_python_prompt(mt, dst_s, pivot_prefix=pi.pivot_prefix, include_docstring=True)
        rows.append(dict(k=k, pair=pair, pi=pi, n_dst=len(mt.tokenizer(dst)["input_ids"]),
                         src=build_cpp_prompt(mt, src_s, include_docstring=True)))
    return ds, rows


def _training_instances():
    """5 instances: two train batches of 2 whose destinations have equal lengths within the
    batch (no padding in the patched forward), then 1 val instance (train_head_mask's 80/20 split
    of 5 = 4 train + 1 val)."""
    ds, rows = _pool_rows()
    by_len: dict[int, list[dict]] = {}
    for r in rows:
        by_len.setdefault(r["n_dst"], []).append(r)
    groups = [g[:2] for g in by_len.values() if len(g) >= 2][:2]
    assert len(groups) == 2, "no two equal-length destination pairs in the pool"
    chosen = [r for g in groups for r in g]
    val = next(r for r in rows if r not in chosen)
    picked = chosen + [val]
    return ds, [(r["pair"], r["pi"]) for r in picked], [r["k"] for r in picked]


def _train(fn_cache, fn_train, ds, insts, *, augment=False, grad_enabled=True):
    mt = model()
    layers = list(range(mt.num_layers))
    caches = fn_cache(mt, insts, ds, layers=layers, precompute=not augment, cache_batch_size=64,
                      include_docstring=True)
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(grad_enabled)
    try:
        mask, hist = fn_train(mt, caches, layers=layers, n_epochs=1, lr=1e-2, lambda_sparse=1.0,
                              batch_size=2, checkpoint_every=1, checkpoint_dir=None, augment=augment,
                              augment_include_docstring=True, augment_cache_batch_size=64,
                              augment_seed=123456)
    finally:
        torch.set_grad_enabled(prev)
    return mask.values.detach().clone(), hist


def test_train_two_steps():
    """``train_head_mask`` (precomputed path), 4 train instances in 2 equal-length batches of 2,
    1 epoch = 2 optimizer steps, lambda 1, lr 1e-2, then val recall on the 5th instance: mask values
    allclose at 1e-5, the same losses and val recall. Then the new loop again with gradients
    turned off process-wide (as the eval scripts do), which the old loop could not run."""
    O = old()
    ds, insts, ks = _training_instances()
    ma, ha = _train(O.old_head_loc_attn.cache_pivot_data, O.old_head_loc_attn.train_head_mask, ds, insts)
    mb, hb = _train(cache_pivot_data, train_head_mask, ds, insts)
    torch.testing.assert_close(ma, mb, rtol=1e-5, atol=1e-5)
    assert (ma != 1.0).any()  # the steps moved the mask
    torch.testing.assert_close(torch.tensor(ha.loss_causal), torch.tensor(hb.loss_causal), rtol=1e-5, atol=1e-6)
    assert ha.val_recall == hb.val_recall
    bit = torch.equal(ma, mb) and ha.loss_causal == hb.loss_causal

    mc, hc = _train(cache_pivot_data, train_head_mask, ds, insts, grad_enabled=False)
    assert torch.equal(mb, mc) and hb.loss_causal == hc.loss_causal
    try:
        _train(O.old_head_loc_attn.cache_pivot_data, O.old_head_loc_attn.train_head_mask, ds, insts,
               grad_enabled=False)
        old_no_grad = "old loop also ran"
    except RuntimeError as e:
        old_no_grad = f"old loop raises {str(e).splitlines()[0][:60]!r}"
    _record(f"train_head_mask, 2 steps (instances {ks[:4]}, val {ks[4]}; batch 2 equal-length, "
            "lambda 1, lr 1e-2)",
            f"{'bit-identical' if bit else 'allclose 1e-5'} mask (max |old - new| {_maxdiff(ma, mb):.1e}); "
            f"KL {hb.loss_causal[0]:.4f}, val recall {hb.val_recall[0][:2]}; under global no-grad: "
            f"new identical, {old_no_grad}")


def test_train_augment_smoke():
    """Not pass/fail on numbers unless every batch is unpadded: the augment path (per-epoch
    augmentation, source o_proj inputs recomputed per batch, augmented val) for one epoch, old vs
    new from the same seed. Augmentation can change destination lengths, so a batch may be padded,
    and the padded row then differs by the position ids (see test_forward_and_grad_padded)."""
    O = old()
    ds, insts, _ = _training_instances()
    ma, ha = _train(O.old_head_loc_attn.cache_pivot_data, O.old_head_loc_attn.train_head_mask, ds, insts,
                    augment=True)
    mb, hb = _train(cache_pivot_data, train_head_mask, ds, insts, augment=True)
    # The same augmentations on both sides (same seed): rebuild them to see whether a batch was padded.
    import random

    caches = cache_pivot_data(model(), insts, ds, layers=[0], precompute=False, include_docstring=True)
    aug = augment_pivot_caches(model(), caches[:4], include_docstring=True, rng=random.Random(123456))
    padded = any(not bool(prepare_input([c.dest_prompt for c in aug[i:i + 2]], tokenizer=model())
                          ["attention_mask"].all()) for i in (0, 2))
    if not padded:
        torch.testing.assert_close(ma, mb, rtol=1e-5, atol=1e-5)
    _record("train_head_mask augment path, 1 epoch (smoke; old vs new)",
            f"runs; train batches padded: {padded}; max |old - new| mask {_maxdiff(ma, mb):.1e}, "
            f"KL {ha.loss_causal[0]:.4f} vs {hb.loss_causal[0]:.4f}, val recall {ha.val_recall[0][:2]} "
            f"vs {hb.val_recall[0][:2]}")


def test_cache_pivot_data():
    """``cache_pivot_data`` on 2 instances of different lengths, precompute True and False: equal
    log-probs, source head values, prompts and target tokens."""
    O = old()
    mt = model()
    ds, rows = _pool_rows(40)
    a, b = rows[0], next(r for r in rows[1:] if r["n_dst"] != rows[0]["n_dst"])
    insts = [(a["pair"], a["pi"]), (b["pair"], b["pi"])]
    layers = list(range(mt.num_layers))
    n_eq = 0
    for precompute in (True, False):
        ca = O.old_head_loc_attn.cache_pivot_data(mt, insts, ds, layers=layers, precompute=precompute,
                                                  include_docstring=True)
        cb = cache_pivot_data(mt, insts, ds, layers=layers, precompute=precompute, include_docstring=True)
        for x, y in zip(ca, cb):
            assert torch.equal(x.log_p_target, y.log_p_target)
            assert (x.src_head_vals is None) == (y.src_head_vals is None) == (not precompute)
            if precompute:
                assert torch.equal(x.src_head_vals, y.src_head_vals)
            assert x.pivot_instance == y.pivot_instance
            assert (x.dest_prompt, x.src_prompt) == (y.dest_prompt, y.src_prompt)
            n_eq += 1
    _record(f"cache_pivot_data (instances {a['k']}, {b['k']}; dest lengths {a['n_dst']}, {b['n_dst']}; "
            "precompute True / False)",
            f"equal (torch.equal) {n_eq}/{n_eq} caches")


def test_forward_timing():
    """Old vs new forward + backward on the equal-length batch of 2 over all layers. CPU timing is
    only indicative."""
    mt, O = model(), old()
    layers = list(range(mt.num_layers))
    inputs = prepare_input(_chat(EQUAL), tokenizer=mt)
    src, log_p = _sources_and_targets(layers)
    fns = {"old": O.old_head_loc_attn.forward_with_src_replacement, "new": forward_with_src_replacement}
    times = {"old": [], "new": []}
    for i in range(6):  # interleaved, the first round a warm-up
        for name, fn in fns.items():
            t = time.perf_counter()
            _forward_and_grad(fn, inputs, src, log_p, layers, 1.0, [-1])
            if i:
                times[name].append(time.perf_counter() - t)
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    _record("forward + backward timing (batch 2, all layers, CPU)",
            f"old {med['old']:.2f} s, new {med['new']:.2f} s per step (median of 5, interleaved)")


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
