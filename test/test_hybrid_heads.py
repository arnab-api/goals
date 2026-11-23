"""Smoke test: goal-head localization machinery on a HYBRID model (attention + Gated DeltaNet heads).

A tiny, randomly initialised Qwen3.5 text model (no weights downloaded): 4 layers, DeltaNet / attention /
DeltaNet / attention, with DIFFERENT head counts and widths per layer type (DeltaNet 6 heads x 8 = 48, attention
4 heads x 16 = 64), so ragged masks and padding are exercised. The tokenizer is Qwen3.8-27B's (local cache).

Checks:
  1. head specs: projection path, head count, head width per layer type
  2. per-head decomposition: the summed per-head writes (collect_head_steering_vecs) == the projection's output
  3. ragged HeadMask: valid entries, clamp keeps non-existent heads at 0, sparsity over existing heads
  4. replacement hooks: mask 0 -> clean logits; source = own inputs at mask 1 -> clean logits; a foreign source on
     an attention head and on a DeltaNet head changes the logits; gradients reach both kinds of heads, never
     the non-existent ones
  5. train_head_mask: a few epochs on synthetic pivot caches (precomputed path) run, and the invalid heads stay 0
  6. apply_goalscope: additive_norm and replace with a head set mixing attention and DeltaNet heads

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= conda run -n goals python test/test_hybrid_heads.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

TOK = "Qwen/Qwen3.8-27B"
HF_CACHE = os.environ.get("HF_HUB_CACHE", "/disk/u/models")


def build_tiny(tmp: str) -> str:
    import transformers as T
    tok = T.AutoTokenizer.from_pretrained(TOK, cache_dir=HF_CACHE)
    cfg = T.Qwen3_5TextConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_value_heads=6, linear_num_key_heads=2, linear_value_head_dim=8, linear_key_head_dim=8,
        linear_conv_kernel_dim=4, max_position_embeddings=512, tie_word_embeddings=False)
    torch.manual_seed(0)
    m = T.Qwen3_5ForCausalLM(cfg).float()
    m.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    return tmp


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def main():
    from src.goals.goalscope import apply_goalscope
    from src.goals.head_loc_attn import (HeadMask, PivotCache, collect_head_steering_vecs, forward_with_src_replacement,
                                         get_o_proj_inputs_batch, train_head_mask)
    from src.models import ModelandTokenizer
    from src.tokens import prepare_input
    import baukit

    tmp = build_tiny(tempfile.mkdtemp(prefix="tiny_qwen35_"))
    mt = ModelandTokenizer(model_key=tmp, dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    L = mt.n_layer

    print("1. head specs")
    specs = [mt.head_spec(l) for l in range(L)]
    for s in specs:
        print(f"     layer {s.layer}: {s.kind:6s} {s.proj}  {s.n_heads} x {s.head_dim} = {s.width}")
    check([s.kind for s in specs] == ["linear", "attn", "linear", "attn"], "layer kinds follow layer_types")
    check(specs[0].proj.endswith("layers.0.linear_attn.out_proj") and specs[1].proj.endswith("layers.1.self_attn.o_proj"),
          "projection paths (DeltaNet out_proj / attention o_proj)")
    check((specs[0].n_heads, specs[0].head_dim, specs[1].n_heads, specs[1].head_dim) == (6, 8, 4, 16), "head counts / dims")
    check(mt.max_heads == 6, "max_heads = 6")

    prompt = "<code>\ndef func(n: int):\n    a = n % 2 == 0\n    b ="
    other = "<code>\ndef func(s: str):\n    a = s[::-1] == s\n    b ="
    tok = prepare_input([prompt], tokenizer=mt)
    T_ = tok["input_ids"].shape[1]
    pos = [T_ - 3, T_ - 1]

    print("2. per-head decomposition (sum of per-head writes == projection output)")
    outs = {}
    hooks = [baukit.get_module(mt._model, specs[l].proj).register_forward_hook(
        lambda m, i, o, l=l: outs.__setitem__(l, o[0, pos].detach().clone())) for l in range(L)]
    with torch.no_grad():
        mt._model(**tok)
    for h in hooks:
        h.remove()
    for l in range(L):
        v = collect_head_steering_vecs(mt, tok, [(l, h) for h in range(specs[l].n_heads)], pos)[0]
        check(torch.allclose(v, outs[l], atol=1e-4), f"layer {l} ({specs[l].kind}): max |Σ heads - out| = "
              f"{(v - outs[l]).abs().max():.2e}")

    print("3. ragged HeadMask")
    layers = list(range(L))
    mask = HeadMask.for_layers(mt, layers)
    check(mask.values.shape == (4, 6) and mask.valid.sum().item() == 6 + 4 + 6 + 4, "shape (4, 6), 20 existing heads")
    with torch.no_grad():
        mask.values.fill_(0.7)
    mask.clamp_()
    check(bool((mask.values[1, 4:] == 0).all()) and bool((mask.values[0] == 0.7).all()), "clamp_ zeroes non-existent heads")
    check(abs(mask.sparsity().item() - 0.7) < 1e-6, "sparsity = mean over existing heads")
    check(len(mask.selected_heads(layers)) == 20 and mask.n_active() == 20, "selected_heads / n_active skip padding")

    print("4. replacement hooks")
    src_self = get_o_proj_inputs_batch(mt, [prompt], layers)
    src_other = get_o_proj_inputs_batch(mt, [other], layers)
    check(src_self[0].shape[-1] == 64 and src_self[1].shape[-1] == 64, "inputs padded to the widest layer (48 -> 64)")
    stack = lambda d: torch.stack([d[l][0] for l in layers])
    dest = prepare_input([prompt], tokenizer=mt)
    with torch.no_grad():
        clean = mt._model(**dest).logits[0, -1]
        m0 = HeadMask.for_layers(mt, layers)
        with torch.no_grad():
            m0.values.zero_()
        check(torch.allclose(forward_with_src_replacement(mt, dest, stack(src_other), m0, layers), clean, atol=1e-4),
              "mask 0 -> clean logits")
        m1 = HeadMask.for_layers(mt, layers)
        check(torch.allclose(forward_with_src_replacement(mt, dest, stack(src_self), m1, layers), clean, atol=1e-4),
              "own source at mask 1 -> clean logits")
        for l, name in [(2, "DeltaNet"), (1, "attention")]:   # layer 0 sees only embeddings: both prompts end alike
            mm = HeadMask.for_layers(mt, layers)
            with torch.no_grad():
                mm.values.zero_(); mm.values[l, 0] = 1.0
            d = (forward_with_src_replacement(mt, dest, stack(src_other), mm, layers) - clean).abs().max()
            check(d > 1e-5, f"a foreign source on one {name} head (layer {l}) changes the logits (max |Δ| {d:.2e})")
    mg = HeadMask.for_layers(mt, layers)
    with torch.no_grad():
        mg.values.fill_(0.5); mg.clamp_()
    logits = forward_with_src_replacement(mt, dest, stack(src_other), mg, layers)
    F.kl_div(F.log_softmax(clean, -1), F.log_softmax(logits, -1), reduction="batchmean", log_target=True).backward()
    gr = mg.values.grad
    check(bool((gr[2, :6].abs() > 0).any()) and bool((gr[1, :4].abs() > 0).any()), "gradients reach DeltaNet and attention heads")
    check(bool((gr[1, 4:] == 0).all()) and bool((gr[3, 4:] == 0).all()), "no gradient on non-existent heads")

    print("5. train_head_mask on synthetic pivot caches (precomputed path)")
    caches = []
    for p_src, p_dst in [(other, prompt), (prompt, other)] * 3:
        so = get_o_proj_inputs_batch(mt, [p_src], layers)
        with torch.no_grad():
            lp = F.log_softmax(mt._model(**prepare_input([p_src], tokenizer=mt)).logits[0, -1].float(), -1)
        caches.append(PivotCache(pivot_instance=SimpleNamespace(target_top_token=" ="), dest_prompt=p_dst,
                                 log_p_target=lp, src_sample=None, dest_sample=None,
                                 src_head_vals=torch.stack([so[l][0] for l in layers])))
    tm, hist = train_head_mask(mt, caches, layers, n_epochs=3, lr=1e-2, lambda_sparse=1.0, batch_size=2,
                               checkpoint_every=3, checkpoint_dir=None, augment=False)
    check(len(hist.loss_causal) == 3 and all(x == x for x in hist.loss_causal), f"3 epochs, KL {[f'{x:.2e}' for x in hist.loss_causal]}")
    moved = (1 - tm.values[tm.valid]).abs().max().item()
    check(moved > 0, f"mask moved from its all-ones init (max |1 - m| = {moved:.2e})")
    check(bool((tm.values[1, 4:] == 0).all()) and bool((tm.values[3, 4:] == 0).all()), "non-existent heads stay 0")
    check(bool(((tm.values >= 0) & (tm.values <= 1)).all()), "mask in [0, 1]")

    print("6. apply_goalscope with a mixed head set (DeltaNet + attention)")
    heads = [(0, 0), (0, 5), (1, 1), (2, 3), (3, 2)]
    full = prompt + " n % 3 == 0\n"
    for strat in ["additive_norm", "replace"]:
        kw = dict(steer_layer_range=(1, 3), alpha=0.1) if strat == "additive_norm" else dict(alpha=1.0)
        res = apply_goalscope(full, mt=mt, selected_heads=heads, strategy=strat, max_new_tokens=6,
                              include_noncode=True, token_positions=[T_ - 1], **kw)
        r = [x for x in res if x.get("scoped")]
        check(len(r) == 1 and isinstance(r[0]["patched_gen"], str), f"{strat}: read-out {r[0]['patched_gen']!r}")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
