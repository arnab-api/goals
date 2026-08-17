"""head_additive_generate == the inline additive path in apply_goalscope, for both strategies.

Runs on `hf-internal-testing/tiny-random-LlamaForCausalLM` (CPU, random weights): equivalence is
token-for-token agreement between two code paths, so model quality is irrelevant.

`_ref_inline` is a verbatim copy of the additive branch inside apply_goalscope as it stands BEFORE
the refactor, so this is a genuine old-vs-new check.
"""
import sys, torch
sys.path.append(".")

from src.functional import PatchSpec, generate_with_patch
from src.goals.goalscope import head_additive_generate
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils.typing import TokenizerOutput

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _ref_inline(mt, dest_inputs, steering_vecs, steer_layers, max_new_tokens,
                strategy, n_steer, alpha):
    """Verbatim apply_goalscope: normalise up front, tile, build PatchSpecs, generate."""
    sv = steering_vecs
    if strategy == "additive_norm":
        sv = sv / sv.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).to(sv.dtype)
    B = sv.shape[0]
    tiled = TokenizerOutput(dict(
        input_ids=dest_inputs["input_ids"].expand(B, -1).clone(),
        attention_mask=dest_inputs["attention_mask"].expand(B, -1).clone()))
    sv_batch = alpha * sv
    patches = [
        PatchSpec(location=(mt.layer_name_format.format(li), -1), patch=sv_batch,
                  strategy="add" if strategy == "additive" else "add_norm", on_pass="all")
        for li in steer_layers
    ]
    return generate_with_patch(mt, tiled, max_new_tokens=max_new_tokens, patches=patches,
                               n_steer_tokens=n_steer, do_sample=False,
                               patch_interface="baukit", remove_prefix=True,
                               skip_special_tokens=True)


def main():
    mt = ModelandTokenizer(model_key=MODEL, dtype=torch.float32, device_map="cpu")
    print(f"{MODEL}  n_layer={mt.n_layer}  n_embd={mt.n_embd}")
    inp = prepare_input(prompts="def check(x):\n    a = ", tokenizer=mt)
    dest = dict(inp)
    torch.manual_seed(0)
    layers = list(range(0, mt.n_layer))
    sv1 = torch.randn(mt.n_embd)
    sv3 = torch.randn(3, mt.n_embd) * 5.0        # non-unit norms, so add vs add_norm differ

    fails = 0
    for strategy in ("additive", "additive_norm"):
        for n_steer in (1, None):
            for alpha in (0.2, 2.0):
                kw = dict(max_new_tokens=6, strategy=strategy,
                          n_steer_tokens=n_steer, alpha=alpha)
                got = head_additive_generate(mt, dest, sv3, layers, **kw)
                want = _ref_inline(mt, dest, sv3, layers, 6, strategy, n_steer, alpha)
                ok_b = list(got) == list(want) and isinstance(got, list)
                got1 = head_additive_generate(mt, dest, sv1, layers, **kw)
                want1 = _ref_inline(mt, dest, sv1.unsqueeze(0), layers, 6, strategy,
                                    n_steer, alpha)[0]
                ok_s = got1 == want1 and isinstance(got1, str)
                fails += (not ok_b) + (not ok_s)
                print(f"  {strategy:14s} n_steer={str(n_steer):4s} a={alpha:<4} | "
                      f"2D->list {'ok' if ok_b else 'FAIL'} | 1D->str {'ok' if ok_s else 'FAIL'}")
                if not ok_b:
                    print(f"     new={list(got)}\n     old={list(want)}")

    # the two strategies must actually differ (otherwise 'equivalent' is vacuous)
    a = head_additive_generate(mt, dest, sv3, layers, max_new_tokens=6,
                               strategy="additive", alpha=2.0)
    n = head_additive_generate(mt, dest, sv3, layers, max_new_tokens=6,
                               strategy="additive_norm", alpha=2.0)
    print(f"\n  additive != additive_norm on the same input: {list(a) != list(n)}")
    fails += (list(a) == list(n))

    # normalisation is internal: pre-normalising the input must not change additive_norm
    unit = sv3 / sv3.norm(dim=-1, keepdim=True)
    n2 = head_additive_generate(mt, dest, unit, layers, max_new_tokens=6,
                                strategy="additive_norm", alpha=2.0)
    print(f"  additive_norm is scale-invariant in the input: {list(n) == list(n2)}")
    fails += (list(n) != list(n2))

    for bad, exc in ((dict(strategy="replace"), ValueError),):
        try:
            head_additive_generate(mt, dest, sv1, layers, max_new_tokens=2, **bad)
            print(f"  {list(bad)[0]:22s} -> NO ERROR  <-- FAIL"); fails += 1
        except exc as e:
            print(f"  {list(bad)[0]:22s} -> {type(e).__name__}: {str(e)[:52]}")

    # An arbitrary position RUNS (prefill-only) and warns — it used to raise. The prefill edit still
    # reaches later tokens through the cached K/V, so this is a restriction on where a fresh patch
    # is written, not on how far the effect travels.
    import warnings as _w
    for n_steer, want_diverge in ((1, False), (3, True)):
        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter("always")
            out = head_additive_generate(mt, dest, sv1, layers, max_new_tokens=4,
                                         n_steer_tokens=n_steer, patch_input_positions=[-2])
        msgs = [str(x.message) for x in rec if issubclass(x.category, RuntimeWarning)]
        ok = bool(msgs) and isinstance(out, str) and (("fresh patch" in msgs[0]) == want_diverge)
        fails += (not ok)
        print(f"  pos=[-2] n_steer={n_steer:<4} -> ran, warns={bool(msgs)}, "
              f"divergence-note={'fresh patch' in (msgs[0] if msgs else '')} "
              f"(want {want_diverge}) {'ok' if ok else 'FAIL'}")

    # multiple arbitrary positions build one PatchSpec per (layer, position)
    import src.goals.goalscope as _G
    seen = {}
    _orig = _G.generate_with_patch
    def _spy(m_, i_, **kw):
        seen["locs"] = sorted({p.location[1] for p in kw["patches"]})
        seen["on_pass"] = sorted({p.on_pass for p in kw["patches"]})
        seen["n"] = len(kw["patches"])
        return _orig(m_, i_, **kw)
    _G.generate_with_patch = _spy
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        head_additive_generate(mt, dest, sv1, layers, max_new_tokens=2,
                               n_steer_tokens=1, patch_input_positions=[-3, -2])
    _G.generate_with_patch = _orig
    ok = (seen["locs"] == [-3, -2] and seen["on_pass"] == ["first"]
          and seen["n"] == 2 * len(layers))
    fails += (not ok)
    print(f"  pos=[-3,-2] -> {seen['n']} specs at {seen['locs']} on_pass={seen['on_pass']} "
          f"(expect {2*len(layers)}) {'ok' if ok else 'FAIL'}")

    # stop_ids now works (forwarded as HF generate's eos_token_id, which takes a list)
    _full = head_additive_generate(mt, dest, sv1, layers, max_new_tokens=10, alpha=0.5)
    _ids = mt.tokenizer(_full, add_special_tokens=False)["input_ids"]
    if len(_ids) > 3:
        _cut = head_additive_generate(mt, dest, sv1, layers, max_new_tokens=10, alpha=0.5,
                                      stop_ids={_ids[2]})
        ok = _full.startswith(_cut) and len(_cut) < len(_full)
        fails += (not ok)
        print(f"  stop_ids={{{_ids[2]}}} -> truncates and stays a prefix: {ok}")

    # out-of-range positions must name the culprit, not surface as a raw IndexError from the hook
    L = dest["input_ids"].shape[1]
    for bad_pos in ([-(L + 5)], [L + 1]):
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                head_additive_generate(mt, dest, sv1, layers, max_new_tokens=2,
                                       n_steer_tokens=1, patch_input_positions=bad_pos)
            print(f"  pos={bad_pos} -> NO ERROR  <-- FAIL"); fails += 1
        except ValueError as e:
            print(f"  pos={str(bad_pos):10s} -> ValueError: {str(e)[:58]}")
        except IndexError:
            print(f"  pos={bad_pos} -> raw IndexError  <-- FAIL"); fails += 1

    print("\n" + ("ALL EQUIVALENT" if not fails else f"{fails} FAILURE(S)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
