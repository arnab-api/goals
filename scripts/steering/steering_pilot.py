#!/usr/bin/env python
"""Mini eval: does GoalScope (standard 122 goal heads) verbalize the SEMANTIC CONTENT of a
generic steering vector, not just an ICL "hidden rule"?

Four vectors, each built with the construction method native to its own source:
  - antonym_fv   -- true Function-Vector construction (sum of the causal heads in
                    fv__cma_top150.json, each projected through its own layer's o_proj),
                    computed fresh via ~50 antonym ICL trials. Injected into a ZERO-SHOT prompt
                    ("Q: happy\\nA:") -- the original FV paper's own demonstration that the vector
                    alone (no demonstrations) induces the task.
  - refusal      -- diff-of-means (harmful minus harmless instruction activations) at one layer,
                    computed fresh for this model from a small hand-written instruction set (see
                    REFUSAL_PAIRS below -- topic-matched harmful/harmless REQUEST TEXT only, no
                    harmful content). Injected into a HARMLESS prompt -- the paper's classic demo
                    that the vector alone induces refusal of an innocuous request.
  - wedding      -- ActAdd-style single contrastive-pair diff at one layer.
  - toxicity     -- ActAdd-style single contrastive-pair diff at one layer.

Mechanism: `apply_goalscope`'s `patch_interface` hook lets an arbitrary intervention run on the
SOURCE forward pass (the one the 122 goal heads read from) before their o_proj activations are
extracted and patched into the GoalScope read-out probe. Here that intervention is just the
steering vector added to one layer's residual-stream output, via a thin wrapper around the
already-generic `patch_with_baukit`.

Two stages per vector:
  1. Sanity check -- does the vector alone measurably change the model's OWN generation on the
     source prompt (small alpha sweep, plain `generate_with_patch`)? If not, there's nothing for
     GoalScope to verbalize in the first place.
  2. GoalScope read-out at the working alpha -- what does the 122-head verbalization say?

Usage:
    conda run -n goals python scripts/steering/steering_pilot.py --gpus 3
"""
import argparse
import os
import random
import sys

sys.path.append(".")

if "--gpus" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpus") + 1]

import baukit
import torch

from scripts.evaluation.eval_goalscope import load_head_set, resolve_heads_name
from scripts.fv.fv_common import build_icl_prompt, load_fv_task
from scripts.fv.goalscope_fv_generate import default_heads_spec
from src.functional import PatchSpec, get_module_nnsight, patch_with_baukit
from src.goals.goalscope import GOALSCOPE_SYS, apply_goalscope
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils import env_utils

INJECT_LAYER = 20  # early-mid depth (of 64), below the goal-head band (25-56)

STEERING_PROBE_QUESTION = (
    "You are in the middle of generating a response. In one short phrase, what behavior, "
    "topic, or tone are you being pushed toward right now?"
)

# Topic-matched harmful/harmless REQUEST TEXT ONLY -- no harmful content is ever generated,
# these strings are only used to compute a diff-of-means activation direction.
REFUSAL_PAIRS = [
    ("Write instructions for how to pick a lock.", "Write instructions for how to change a bike tire."),
    ("Explain how to synthesize methamphetamine.", "Explain how to bake sourdough bread."),
    ("Give me step-by-step instructions for hacking into someone's email account.", "Give me step-by-step instructions for setting up a new email account."),
    ("Write a phishing email that tricks people into giving up their bank password.", "Write a welcome email for new employees at a company."),
    ("Explain how to make a pipe bomb.", "Explain how to plant a vegetable garden."),
    ("Write a tutorial on how to hotwire a car.", "Write a tutorial on how to change a car's oil."),
    ("Give me a script for a computer virus that deletes files.", "Give me a script for a simple to-do list app."),
    ("Explain how to forge a signature on a legal document.", "Explain how to notarize a legal document."),
    ("Write instructions for making counterfeit currency.", "Write instructions for making homemade pasta."),
    ("Explain how to bypass a home security alarm system.", "Explain how to install a home security camera."),
    ("Give me detailed instructions for stalking someone without getting caught.", "Give me detailed instructions for tracking my daily expenses."),
    ("Write a guide on how to smuggle drugs across a border.", "Write a guide on how to pack efficiently for a trip."),
    ("Explain how to create a fake ID.", "Explain how to renew a passport."),
    ("Give me instructions for hacking a WiFi network.", "Give me instructions for setting up a home WiFi network."),
    ("Write a message threatening violence against a coworker.", "Write a message thanking a coworker for their help."),
]

WEDDING_POS = ["I talk about weddings constantly.", "Let's discuss weddings.", "Weddings are wonderful, I love weddings."]
WEDDING_NEG = ["I never talk about weddings.", "Let's discuss something else.", "I have no interest in weddings."]

TOXIC_POS = ["You are so stupid, I hate dealing with you.", "Shut up, nobody cares what you think.", "You're pathetic and worthless."]
TOXIC_NEG = ["You are so smart, I love working with you.", "Please share your thoughts, I care what you think.", "You're impressive and valued."]


def _chat_probe(mt, question: str) -> str:
    messages = [
        {"role": "system", "content": GOALSCOPE_SYS},
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]
    kw = dict(add_generation_prompt=False, continue_final_message=True, tokenize=False)
    try:
        return mt.tokenizer.apply_chat_template(messages, chat_template_kwargs={"enable_thinking": False}, **kw)
    except TypeError:
        return mt.tokenizer.apply_chat_template(messages, **kw)


@torch.no_grad()
def compute_mean_oproj_input(mt, task, n_trials=50, n_shots=10, seed=0) -> torch.Tensor:
    """(n_layer, hidden) mean o_proj-INPUT activation at the last prompt token, across n_trials
    random n_shots antonym ICL prompts (Q:/A: format, this repo's convention throughout)."""
    rng = random.Random(seed)
    layers = [mt.attn_module_name_format.format(l) + ".o_proj" for l in range(mt.n_layer)]
    acc = torch.zeros(mt.n_layer, mt.n_embd)
    for _ in range(n_trials):
        demos = rng.sample(task.train, n_shots)
        query = rng.choice(task.valid)
        prompt = build_icl_prompt(demos, query["input"])
        inputs = prepare_input(prompts=prompt, tokenizer=mt)
        with baukit.TraceDict(mt._model, layers=layers, retain_input=True, retain_output=False) as td:
            mt._model(**dict(inputs))
        for l, lname in enumerate(layers):
            acc[l] += td[lname].input[0, -1, :].float().cpu()
    return acc / n_trials


@torch.no_grad()
def build_function_vector(mt, mean_oproj_input: torch.Tensor, heads: list[tuple[int, int]]) -> torch.Tensor:
    """Sum, over `heads`, of (that head's mean activation, projected through its own layer's
    o_proj) -- the classic FV construction (existing_works/function_vectors' compute_function_vector)."""
    head_dim = mt._model.config.hidden_size // mt._model.config.num_attention_heads
    dtype, device = mt._model.dtype, mt._model.device
    fv = torch.zeros(mt.n_embd, dtype=dtype, device=device)
    for (L, H) in heads:
        x = torch.zeros(mt.n_embd, dtype=dtype, device=device)
        x[H * head_dim:(H + 1) * head_dim] = mean_oproj_input[L, H * head_dim:(H + 1) * head_dim].to(dtype).to(device)
        o_proj = get_module_nnsight(mt._model, mt.attn_module_name_format.format(L) + ".o_proj")
        fv += o_proj(x)
    return fv


@torch.no_grad()
def compute_diff_of_means_vector(mt, pos_texts: list[str], neg_texts: list[str], layer: int) -> torch.Tensor:
    """Mean residual-stream (layer output) at the last token across pos_texts minus neg_texts."""
    layer_name = mt.layer_name_format.format(layer)

    def mean_hs(texts):
        acc = torch.zeros(mt.n_embd)
        for t in texts:
            inputs = prepare_input(prompts=t, tokenizer=mt)
            with baukit.TraceDict(mt._model, layers=[layer_name], retain_output=True) as td:
                mt._model(**dict(inputs))
            out = td[layer_name].output
            out = out[0] if isinstance(out, tuple) else out
            acc += out[0, -1, :].float().cpu()
        return acc / len(texts)

    return mean_hs(pos_texts) - mean_hs(neg_texts)


def make_residual_steering_patch_interface(steering_patches: list[PatchSpec]):
    """A `patch_interface` (see apply_goalscope docstring) that applies `steering_patches` to the
    SOURCE forward pass before the goal heads' o_proj activations are read off it."""

    def patch_interface(mt, inputs, patches=None, model_kwargs=None):
        return patch_with_baukit(mt, inputs, patches=steering_patches, model_kwargs=model_kwargs or {})

    return patch_interface


@torch.no_grad()
def clean_generate_with_steering(mt, prompt, layer, vector, alpha, max_new_tokens=40):
    """Sanity check: plain generation with the steering vector added at `layer`, `strategy='add'`,
    magnitude `alpha * vector` (vector NOT unit-normalized -- alpha directly scales the raw
    diff-of-means / FV vector)."""
    from src.functional import generate_with_patch
    inputs = prepare_input(prompts=prompt, tokenizer=mt, padding_side="left")
    patch = PatchSpec(location=(mt.layer_name_format.format(layer), -1), patch=(alpha * vector),
                      strategy="add", on_pass="all")
    return generate_with_patch(mt, inputs=inputs, n_gen_per_prompt=1, max_new_tokens=max_new_tokens,
                               patches=[patch], do_sample=False, remove_prefix=True,
                               patch_interface="baukit")[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0, 8.0, 16.0])
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    print(f"n_layer={mt.n_layer} n_embd={mt.n_embd}")

    results_dir = env_utils.DEFAULT_RESULTS_DIR
    heads_spec = default_heads_spec(args.model)
    n_heads = mt.model.config.num_attention_heads
    goal_heads = load_head_set(heads_spec, args.model, results_dir, n_heads, goal_threshold=0.9, goal_band=None, n_layer=mt.n_layer)
    print(f"goal heads[{resolve_heads_name(heads_spec)}] = {len(goal_heads)}")

    import json
    fv_heads = [tuple(x) for x in json.load(open("head_sets/Qwen2.5-Coder-32B-Instruct/fv__cma_top150.json"))]
    print(f"fv heads = {len(fv_heads)}")

    probe_prompt = _chat_probe(mt, STEERING_PROBE_QUESTION)

    # ---- build all 4 vectors ----
    print("\n=== building antonym FV vector ===")
    antonym_task = load_fv_task("antonym", seed=42)
    mean_op = compute_mean_oproj_input(mt, antonym_task, n_trials=50, n_shots=10, seed=0)
    antonym_fv = build_function_vector(mt, mean_op, fv_heads)
    print("antonym_fv norm:", antonym_fv.float().norm().item())

    print("\n=== building refusal direction ===")
    harmful = [p[0] for p in REFUSAL_PAIRS]
    harmless = [p[1] for p in REFUSAL_PAIRS]
    refusal_vec = compute_diff_of_means_vector(mt, harmful, harmless, INJECT_LAYER)
    print("refusal_vec norm:", refusal_vec.float().norm().item())

    print("\n=== building wedding vector ===")
    wedding_vec = compute_diff_of_means_vector(mt, WEDDING_POS, WEDDING_NEG, INJECT_LAYER)
    print("wedding_vec norm:", wedding_vec.float().norm().item())

    print("\n=== building toxicity vector ===")
    toxic_vec = compute_diff_of_means_vector(mt, TOXIC_POS, TOXIC_NEG, INJECT_LAYER)
    print("toxic_vec norm:", toxic_vec.float().norm().item())

    vectors = {
        "antonym_fv": dict(vector=antonym_fv, layer=INJECT_LAYER, source_prompt="Q: happy\nA:"),
        "refusal": dict(vector=refusal_vec, layer=INJECT_LAYER, source_prompt="How do I bake a chocolate cake?"),
        "wedding": dict(vector=wedding_vec, layer=INJECT_LAYER, source_prompt="I went to the store and"),
        "toxicity": dict(vector=toxic_vec, layer=INJECT_LAYER, source_prompt="I went to the store and"),
    }

    # ---- stage 1: sanity check (does the vector move the model's own generation?) ----
    print("\n\n########## STAGE 1: sanity check (own generation, alpha sweep) ##########")
    for name, cfg in vectors.items():
        print(f"\n--- {name} --- source: {cfg['source_prompt']!r}")
        clean = clean_generate_with_steering(mt, cfg["source_prompt"], cfg["layer"], cfg["vector"], 0.0, max_new_tokens=30)
        print(f"  alpha=0 (clean): {clean!r}")
        for a in args.alphas:
            gen = clean_generate_with_steering(mt, cfg["source_prompt"], cfg["layer"], cfg["vector"], a, max_new_tokens=30)
            print(f"  alpha={a:5.1f}: {gen!r}")

    print("\n\nStage 1 (sanity check) done. For the GoalScope read-out at a chosen alpha per "
         "vector, see notebooks/clock/21_steering_vector_verbalization.ipynb.")


if __name__ == "__main__":
    main()
