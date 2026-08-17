"""headpatched_generate: merged single/batch path == the two pre-merge implementations.

Runs on `hf-internal-testing/tiny-random-LlamaForCausalLM` (CPU, random weights) — equivalence is
about token-for-token agreement between two code paths, so model quality is irrelevant and a 32B
load would be pure waste.

The reference implementations below are verbatim copies of goalscope.headpatched_generate /
_batch as they stood before the merge (commit 6910bf8), so this is a genuine old-vs-new check
rather than the new code compared against itself.
"""
import sys, torch
sys.path.append(".")

from src.goals.goalscope import headpatched_generate, headpatched_generate_batch
from src.goals.head_loc_attn import HeadMask, forward_with_src_replacement
from src.goals.goalscope import _normalize_input_positions
from src.models import ModelandTokenizer
from src.tokens import prepare_input

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _ref_single(mt, dest_inputs, src_o_proj_inps, mask, layers, max_new_tokens,
                stop_ids=None, n_steer_tokens=None, patch_input_positions=None, alpha=1.0):
    _stop = stop_ids or set()
    if len(_stop) == 0:
        _stop = {mt.tokenizer.eos_token_id}
    input_ids = dest_inputs["input_ids"].clone()
    attention_mask = dest_inputs["attention_mask"].clone()
    L_prompt = input_ids.shape[1]
    prefill_positions = _normalize_input_positions(patch_input_positions, L_prompt)
    generated, past_key_values = [], None
    with torch.no_grad():
        for step in range(max_new_tokens):
            cur = input_ids if step == 0 else input_ids[:, -1:]
            inputs = {"input_ids": cur, "attention_mask": attention_mask,
                      "past_key_values": past_key_values, "use_cache": True}
            if n_steer_tokens is None or step < n_steer_tokens:
                pos = prefill_positions if step == 0 else [-1]
                logits, past_key_values = forward_with_src_replacement(
                    mt, inputs, src_o_proj_inps, mask, layers, alpha=alpha,
                    patch_positions=pos, return_past_key_values=True)
            else:
                out = mt._model(**inputs)
                logits = out.logits[0, -1, :].float()
                past_key_values = out.past_key_values
            next_tok = int(logits.argmax())
            generated.append(next_tok)
            if next_tok in _stop:
                break
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_tok]], device=input_ids.device)], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype,
                                            device=attention_mask.device)], dim=1)
    return mt.tokenizer.decode(generated, skip_special_tokens=True).rstrip()


def _ref_batch(mt, dest_inputs, src_batch, mask, layers, max_new_tokens,
               stop_ids=None, n_steer_tokens=None, patch_input_positions=None, alpha=1.0):
    _stop = stop_ids or set()
    if len(_stop) == 0:
        _stop = {mt.tokenizer.eos_token_id}
    B = src_batch.shape[0]
    input_ids = dest_inputs["input_ids"].expand(B, -1).clone()
    attention_mask = dest_inputs["attention_mask"].expand(B, -1).clone()
    L_prompt = input_ids.shape[1]
    prefill_positions = _normalize_input_positions(patch_input_positions, L_prompt)
    generated = [[] for _ in range(B)]
    finished = [False] * B
    past_key_values = None
    with torch.no_grad():
        for step in range(max_new_tokens):
            if all(finished):
                break
            cur = input_ids if step == 0 else input_ids[:, -1:]
            inputs = {"input_ids": cur, "attention_mask": attention_mask,
                      "past_key_values": past_key_values, "use_cache": True}
            if n_steer_tokens is None or step < n_steer_tokens:
                pos = prefill_positions if step == 0 else [-1]
                logits, past_key_values = forward_with_src_replacement(
                    mt, inputs, src_batch, mask, layers, alpha=alpha,
                    patch_positions=pos, return_past_key_values=True)
            else:
                out = mt._model(**inputs)
                logits = out.logits[:, -1, :].float()
                past_key_values = out.past_key_values
            next_toks = logits.argmax(dim=-1)
            for i in range(B):
                if not finished[i]:
                    tok = int(next_toks[i])
                    generated[i].append(tok)
                    if tok in _stop:
                        finished[i] = True
            input_ids = torch.cat([input_ids, next_toks.unsqueeze(1)], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((B, 1), dtype=attention_mask.dtype,
                                            device=attention_mask.device)], dim=1)
    return [mt.tokenizer.decode(g, skip_special_tokens=True).rstrip() for g in generated]


def main():
    mt = ModelandTokenizer(model_key=MODEL, dtype=torch.float32, device_map="cpu")
    n_layer, n_heads = mt.n_layer, mt.model.config.num_attention_heads
    print(f"{MODEL}  n_layer={n_layer} n_heads={n_heads} n_embd={mt.n_embd}")

    inp = prepare_input(prompts="def check(x):\n    a = ", tokenizer=mt)
    dest = dict(inp)
    torch.manual_seed(0)

    # a compact mask over a NON-contiguous layer subset — the case where `layers` is load-bearing
    layers = sorted(set([0, n_layer - 1]))[: max(1, n_layer)]
    mask = HeadMask(len(layers), n_heads)
    with torch.no_grad():
        mask.values.zero_()
        mask.values[0, 0] = 1.0
    src2d = torch.randn(len(layers), mt.n_embd)
    src3d = torch.stack([src2d, torch.randn(len(layers), mt.n_embd)])

    fails = 0
    for n_steer in (1, None):
        for pos in (None, [-2]):
            kw = dict(max_new_tokens=6, n_steer_tokens=n_steer,
                      patch_input_positions=pos, alpha=0.7)
            got = headpatched_generate(mt, dest, src2d, mask, layers, **kw)
            want = _ref_single(mt, dest, src2d, mask, layers, **kw)
            ok1 = got == want and isinstance(got, str)
            gotb = headpatched_generate(mt, dest, src3d, mask, layers, **kw)
            wantb = _ref_batch(mt, dest, src3d, mask, layers, **kw)
            ok2 = gotb == wantb and isinstance(gotb, list)
            # a 3D source with B==1 must still return a list
            got1 = headpatched_generate(mt, dest, src2d.unsqueeze(0), mask, layers, **kw)
            ok3 = isinstance(got1, list) and len(got1) == 1 and got1[0] == want
            fails += (not ok1) + (not ok2) + (not ok3)
            print(f"  n_steer={str(n_steer):4s} pos={str(pos):5s} | "
                  f"2D->str {'ok' if ok1 else 'FAIL'} | 3D->list {'ok' if ok2 else 'FAIL'} | "
                  f"B=1 3D->list {'ok' if ok3 else 'FAIL'}")
            if not ok1:
                print(f"     new={got!r}\n     old={want!r}")
            if not ok2:
                print(f"     new={gotb!r}\n     old={wantb!r}")

    import warnings as _w
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        alias = headpatched_generate_batch(mt, dest, src3d, mask, layers, max_new_tokens=4)
    dep = any(issubclass(x.category, DeprecationWarning) for x in rec)
    print(f"  deprecated alias -> list[{len(alias)}], warns: {dep}")
    fails += (not isinstance(alias, list)) or (not dep)

    try:
        headpatched_generate(mt, dest, torch.randn(3, 2, len(layers), mt.n_embd), mask, layers,
                             max_new_tokens=2)
        print("  4D source -> NO ERROR  <-- FAIL"); fails += 1
    except ValueError as e:
        print(f"  4D source -> ValueError: {str(e)[:60]}")

    print("\n" + ("ALL EQUIVALENT" if not fails else f"{fails} FAILURE(S)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
