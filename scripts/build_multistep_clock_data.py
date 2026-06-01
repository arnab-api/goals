#!/usr/bin/env python
"""Build the multistep clock-signal dataset.

For each MultiStepSample x each modularity in {modular, monolithic}:
  1. Generate a code completion from the LM (greedy decoding).
  2. Label which token spans implement which step (oracle: Claude).
  3. Extract per-head attention distributions over the full sequence.

Output layout:

    results/multistep_clock/<model>/
      metadata.json
      <modularity>/
        <sample_name>/
          completion.json   {sample_name, modularity, prompt, generation}
          labels.json       LabeledGeneration.to_dict()
          attn.npz          {attn: (n_layers, n_heads, seq, seq) float16,
                             tokens: (seq,) str}

Each artifact is checked individually; only missing ones are recomputed. Pass
--overwrite to force re-run.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Make sure the project root is importable when called directly from scripts/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.attention import get_attention_matrices  # noqa: E402
from src.goals.functional import (  # noqa: E402
    apply_chat_template,
    supports_chat_template,
)
from src.goals.goalscope import greedy_continue  # noqa: E402
from src.goals.multistep_data import MultiStepDataset, MultiStepSample  # noqa: E402
from src.goals.multistep_label import label_generation  # noqa: E402
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402
from src.utils import env_utils, experiment_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample processing.
# ──────────────────────────────────────────────────────────────────────────────


def build_prompt(mt, sample: MultiStepSample, modularity: str) -> str:
    """Render the prompt for a given (sample, modularity) without mutating sample."""
    instruction = sample.render_instruction(modularity=modularity)
    code_stub = sample.render_code_stub(modularity=modularity)
    if supports_chat_template(mt):
        return apply_chat_template(
            mt=mt,
            user_query=instruction.strip(),
            incomplete_code=code_stub,
        )
    return instruction + "<code>" + code_stub


def generate_completion(mt, prompt: str, max_new_tokens: int) -> str:
    inputs = prepare_input(prompts=prompt, tokenizer=mt)
    return greedy_continue(mt, **inputs, max_new_tokens=max_new_tokens)


def extract_attention(
    mt,
    full_prompt: str,
    value_weighted: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Return the full attention tensor for all heads.

    Returns:
        attn:   (n_layers, n_heads, seq, seq) float16
        tokens: list[str]                     decoded per-token strings
    """
    attn_info = get_attention_matrices(full_prompt, mt, value_weighted=value_weighted)
    attn = attn_info.attention_matrices.astype(np.float16)
    return attn, attn_info.tokenized_prompt


def process_sample(
    mt,
    sample: MultiStepSample,
    modularity: str,
    out_dir: Path,
    max_new_tokens: int,
    value_weighted: bool,
    overwrite: bool,
) -> None:
    sample_dir = out_dir / sample.name
    sample_dir.mkdir(parents=True, exist_ok=True)
    completion_path = sample_dir / "completion.json"
    labels_path = sample_dir / "labels.json"
    attn_path = sample_dir / "attn.npz"

    if (
        not overwrite
        and completion_path.exists()
        and labels_path.exists()
        and attn_path.exists()
    ):
        logger.info("  [skip] all artifacts already exist")
        return

    # Prompt (deterministic).
    prompt = build_prompt(mt, sample, modularity)

    # 1) Completion.
    if completion_path.exists() and not overwrite:
        with open(completion_path) as f:
            generation = json.load(f)["generation"]
        logger.info("  loaded cached completion")
    else:
        logger.info(f"  generating completion (max_new_tokens={max_new_tokens})")
        generation = generate_completion(mt, prompt, max_new_tokens=max_new_tokens)
        with open(completion_path, "w") as f:
            json.dump(
                {
                    "sample_name": sample.name,
                    "modularity": modularity,
                    "prompt": prompt,
                    "generation": generation,
                },
                f,
                indent=2,
            )

    # 2) Oracle labels.
    if labels_path.exists() and not overwrite:
        logger.info("  loaded cached labels")
    else:
        logger.info("  labeling generation with oracle")
        labeled = label_generation(mt, sample, prompt, generation, use_cache=True)
        with open(labels_path, "w") as f:
            json.dump(labeled.to_dict(), f, indent=2)

    # 3) Attention — full (n_layers, n_heads, seq, seq) for downstream flexibility.
    if attn_path.exists() and not overwrite:
        logger.info("  loaded cached attention")
    else:
        logger.info(f"  extracting full attention (value_weighted={value_weighted})")
        full_prompt = prompt + generation
        attn, tokens = extract_attention(
            mt, full_prompt, value_weighted=value_weighted
        )
        logger.info(f"    attn shape: {attn.shape}, dtype: {attn.dtype}")
        np.savez_compressed(
            attn_path,
            attn=attn,
            tokens=np.array(tokens, dtype=object),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main.
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument(
        "--value_weighted",
        type=lambda x: x.lower() in {"1", "true", "yes"},
        default=True,
    )
    p.add_argument(
        "--modularities",
        nargs="+",
        choices=["modular", "monolithic"],
        default=["modular", "monolithic"],
    )
    p.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="optional subset of sample names; default = all",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dataset_path",
        default=os.path.join(env_utils.DEFAULT_DATA_DIR, "multistep", "seed.json"),
    )
    logging_utils.add_logging_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)

    # Load model.
    logger.info(f"loading model: {args.model}")
    mt = ModelandTokenizer(
        model_key=args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # Dataset.
    ds = MultiStepDataset.load_dataset(args.dataset_path)
    samples = (
        [s for s in ds if s.name in args.samples]
        if args.samples
        else list(ds.samples)
    )
    logger.info(f"processing {len(samples)} samples × {args.modularities}")

    # Output root + metadata.
    out_root = (
        Path(env_utils.DEFAULT_RESULTS_DIR)
        / "multistep_clock"
        / args.model.split("/")[-1]
    )
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "metadata.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "n_layers": mt.n_layer,
                "n_heads": mt.config.num_attention_heads,
                "max_new_tokens": args.max_new_tokens,
                "value_weighted": args.value_weighted,
                "modularities": list(args.modularities),
                "n_samples": len(samples),
                "sample_names": [s.name for s in samples],
            },
            f,
            indent=2,
        )

    # Loop.
    total = len(samples) * len(args.modularities)
    done = 0
    for sample in samples:
        for modularity in args.modularities:
            done += 1
            logger.info(
                f"[{done}/{total}] {sample.name} ({modularity})"
            )
            try:
                process_sample(
                    mt=mt,
                    sample=sample,
                    modularity=modularity,
                    out_dir=out_root / modularity,
                    max_new_tokens=args.max_new_tokens,
                    value_weighted=args.value_weighted,
                    overwrite=args.overwrite,
                )
            except Exception as e:
                logger.exception(f"  FAILED: {e}")

    logger.info(f"done. outputs at {out_root}")


if __name__ == "__main__":
    main()
