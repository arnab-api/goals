#!/usr/bin/env python
"""Cache raw `apply_goalscope` output (list[dict]) to JSON so the interactive HTML renderer
(`src/goals/goalscope_viz.py`) can be iterated on without reloading the model or re-running
GoalScope every time. Load with `load_cached_result`; render straight from the cached dict.

Usage:
    conda run -n goals python scripts/steering/qualitative_cache.py --gpus 1
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(".")

if "--gpus" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpus") + 1]

CACHE_DIR = Path("notebooks/diff/visualizations/cache")


def save_cached_result(name: str, res: list[dict], meta: dict) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.json"
    path.write_text(json.dumps({"meta": meta, "res": res}, indent=2, default=str))
    return path


def load_cached_result(name: str) -> tuple[list[dict], dict]:
    path = CACHE_DIR / f"{name}.json"
    d = json.loads(path.read_text())
    return d["res"], d["meta"]


def main():
    import torch
    from src.models import ModelandTokenizer
    from src.goals.goalscope import apply_goalscope
    from scripts.evaluation.eval_goalscope import load_json_heads
    from src.functional import PatchSpec
    from scripts.steering.steering_pilot import (
        INJECT_LAYER, REFUSAL_PAIRS, compute_diff_of_means_vector,
        make_residual_steering_patch_interface, clean_generate_with_steering,
    )
    from src.tokens import prepare_input

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default=None)
    args = ap.parse_args()

    MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
    slug = MODEL.split("/")[-1]
    HEADS_JSON = f"head_sets/{slug}/goal__lam20_thr0.9_band25-56.json"
    mt = ModelandTokenizer(model_key=MODEL, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    heads = load_json_heads(HEADS_JSON, mt.n_layer, mt.model.config.num_attention_heads)

    harmful = [p[0] for p in REFUSAL_PAIRS]
    harmless = [p[1] for p in REFUSAL_PAIRS]
    refusal_vec = compute_diff_of_means_vector(mt, harmful, harmless, INJECT_LAYER)

    SOURCE_PROMPT = "How do I bake a chocolate cake?"
    ALPHA = 8.0
    GEN_TOKENS = 64

    def _chat(system, user, assistant=""):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        kw = dict(add_generation_prompt=False, continue_final_message=True, tokenize=False)
        return mt.tokenizer.apply_chat_template(msgs, chat_template_kwargs={"enable_thinking": False}, **kw)

    GOALSCOPE_PROMPT = _chat(
        "You are a helpful assistant.",
        "You are in the middle of generating a response. In one short phrase, what behavior or topic "
        "are you contemplating right now?",
    )

    def run_condition(name, alpha):
        gen = clean_generate_with_steering(mt, SOURCE_PROMPT, INJECT_LAYER, refusal_vec, alpha, max_new_tokens=GEN_TOKENS)
        print(f"[{name}] gen: {gen!r}")

        p0 = prepare_input(prompts=SOURCE_PROMPT, tokenizer=mt)["input_ids"].shape[1]
        enc = prepare_input(prompts=SOURCE_PROMPT + gen, tokenizer=mt)
        ptot = enc["input_ids"].shape[1]
        positions = [p0 - 1] + list(range(p0, ptot))

        kwargs = dict(
            prompt=enc, mt=mt, selected_heads=heads, strategy="additive_norm",
            steer_layer_range=(35, 46), alpha=0.1, n_steer_tokens="all",
            max_new_tokens=64, token_positions=positions, include_noncode=True, show=False,
            goalscope_prompt=GOALSCOPE_PROMPT,
        )
        if alpha != 0.0:
            layer_name = mt.layer_name_format.format(INJECT_LAYER)
            scaled_vec = alpha * refusal_vec
            patches = [PatchSpec(location=(layer_name, idx), patch=scaled_vec, strategy="add", on_pass="all")
                      for idx in positions]
            kwargs["patch_interface"] = make_residual_steering_patch_interface(patches)

        res = apply_goalscope(**kwargs)
        meta = dict(source_prompt=SOURCE_PROMPT, gen=gen, alpha=alpha, layer=INJECT_LAYER,
                   vector="refusal (diff-of-means)" if alpha != 0.0 else None,
                   goalscope_prompt_desc="notebook-21 custom non-code steering_probe")
        path = save_cached_result(name, res, meta)
        print(f"[{name}] {sum(1 for r in res if r.get('scoped'))} scoped tokens -> cached at {path}")
        return res

    run_condition("refusal_steered", ALPHA)
    run_condition("refusal_clean", 0.0)


if __name__ == "__main__":
    main()
