#!/usr/bin/env python
"""Build the condition-checking clock dataset (src.goals.condition_data).

For each ConditionSample (a spec under one lang × modularity):
  1. Generate a code completion (batched greedy decoding).
  2. Oracle-label which token span implements which condition (Claude).
  3. Precompute, per condition, the token ranges of its `id_substrs` (the
     informative spec sub-spans) within `prompt + generation` — so the scorer
     (`score_condition_heads.py`) can localize attention to the key spec tokens
     without re-tokenizing.
  4. Extract the full per-head attention tensor.

Output layout (mirrors build_multistep_clock_data so the scorers can share
discovery logic):

    results/multistep_clock/<version>/<model>/
      metadata.json
      <modularity>/<spec_id>/
        completion.json    {sample_name, spec_id, lang, modularity, prompt, generation}
        labels.json        LabeledGeneration.to_dict()
        key_ranges.json    {spec_id, step_key_token_ranges: [[[s,e],...], ...]}
      attn/<modularity>/<spec_id>/
        attn.npz           {attn: (n_layers, n_heads, seq, seq) f16, tokens: (seq,) str}

Stages are independent with per-file existence checks (pass --overwrite to
redo). `--skip_attention` skips stage 4 (attn tensors are large). `--limit`
processes only the first N samples.
"""

import argparse
import gc
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.attention import get_attention_matrices  # noqa: E402
from src.goals.condition_data import ConditionDataset, ConditionSample  # noqa: E402
from src.goals.functional import apply_chat_template, supports_chat_template  # noqa: E402
from src.goals.multistep_label import label_generation  # noqa: E402
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import find_token_range, prepare_input  # noqa: E402
from src.utils import env_utils, experiment_utils, logging_utils  # noqa: E402

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers.
# ──────────────────────────────────────────────────────────────────────────────


def build_prompt(mt, sample: ConditionSample) -> str:
    instr = sample.render_instruction()
    stub = sample.render_code_stub()
    if supports_chat_template(mt):
        return apply_chat_template(mt=mt, user_query=instr.strip(), incomplete_code=stub)
    return instr + "\nAnswer:\n" + stub


def log_dataset_stats(samples: list[ConditionSample]) -> None:
    """Basic distribution summary of the loaded samples."""
    logger.info(f"dataset: {len(samples)} samples, {len({s.spec.spec_id for s in samples})} unique specs")
    logger.info(f"  lang:           {dict(Counter(s.lang for s in samples))}")
    logger.info(f"  modularity:     {dict(Counter(s.modularity for s in samples))}")
    logger.info(f"  num_conditions: {dict(sorted(Counter(s.spec.num_conditions for s in samples).items()))}")
    pred = Counter(pid for s in samples for pid in s.spec.predicate_ids)
    logger.info(f"  predicate freq: {dict(pred.most_common())}")


def sample_dirs(out_root: Path, sample: ConditionSample) -> tuple[Path, Path]:
    """(labeled_dir, attn_dir) for a sample — keyed by modularity / spec_id."""
    lab = out_root / sample.modularity / sample.spec.spec_id
    attn = out_root / "attn" / sample.modularity / sample.spec.spec_id
    return lab, attn


def compute_key_token_ranges(mt, prompt: str, generation: str, sample: ConditionSample):
    """Per condition, the (start, end) token ranges (into `prompt + generation`) of its
    rendered id_substrs, pinned to the condition's spec text in the prompt. Returns
    list[list[[start, end]]] — one inner list per condition (end exclusive).

    The char→token mapping is delegated to src.tokens.find_token_range. Because short
    substrings (e.g. "10") recur, we pin the right one by computing its occurrence index
    from the known char position (`base + rel`); since these substrings don't self-overlap,
    that occurrence resolves to exactly the spec instance.
    """
    combined = prompt + generation
    offsets = [tuple(p) for p in mt.tokenizer(
        combined, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]]

    out = []
    for text, subs in zip(sample.spec.spec_texts, sample.spec.id_substrs):
        base = prompt.find(text)          # spec text occurs once, in the numbered list
        ranges = []
        if base != -1:
            for ss in subs:
                rel = text.find(ss)
                if rel == -1:
                    continue
                occ = combined.count(ss, 0, base + rel)  # 0-indexed occurrence at the spec
                try:
                    ts, te = find_token_range(combined, ss, offset_mapping=offsets, occurrence=occ)
                    ranges.append([ts, te])
                except (ValueError, AssertionError):
                    continue
        out.append(ranges)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stages.
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_batched(mt, samples, out_root, max_new_tokens, batch_size, overwrite):
    """Stage 1: batched greedy generation → completion.json (skips existing)."""
    pad_id = mt.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = mt.tokenizer.eos_token_id
    todo = [s for s in samples
            if overwrite or not (sample_dirs(out_root, s)[0] / "completion.json").exists()]
    logger.info(f"[gen] {len(todo)}/{len(samples)} samples need generation "
                f"(batch_size={batch_size}, max_new_tokens={max_new_tokens})")
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        prompts = [build_prompt(mt, s) for s in chunk]
        inputs = prepare_input(prompts=prompts, tokenizer=mt)  # left-padded
        plen = inputs["input_ids"].shape[1]
        out = mt._model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id,
        )
        for s, prompt, row in zip(chunk, prompts, out[:, plen:]):
            gen = mt.tokenizer.decode(row, skip_special_tokens=True)
            lab_dir, _ = sample_dirs(out_root, s)
            lab_dir.mkdir(parents=True, exist_ok=True)
            with open(lab_dir / "completion.json", "w") as f:
                json.dump({"sample_name": s.name, "spec_id": s.spec.spec_id,
                           "lang": s.lang, "modularity": s.modularity,
                           "prompt": prompt, "generation": gen}, f, indent=2)
        logger.info(f"[gen] {min(i + batch_size, len(todo))}/{len(todo)}")
        del out
        gc.collect()
        torch.cuda.empty_cache()


def label_and_attn(mt, samples, out_root, value_weighted, skip_attention, overwrite):
    """Stages 2-4: oracle labels, id_substr key ranges, attention — per sample."""
    for j, s in enumerate(samples, start=1):
        lab_dir, attn_dir = sample_dirs(out_root, s)
        comp_path = lab_dir / "completion.json"
        if not comp_path.exists():
            logger.warning(f"[{j}/{len(samples)}] {s.name}: no completion — skipping")
            continue
        with open(comp_path) as f:
            data = json.load(f)
        prompt, generation = data["prompt"], data["generation"]
        logger.info(f"[{j}/{len(samples)}] {s.name}")

        # Stage 2: oracle labels.
        labels_path = lab_dir / "labels.json"
        if overwrite or not labels_path.exists():
            try:
                labeled = label_generation(mt, s, prompt, generation, use_cache=True)
                with open(labels_path, "w") as f:
                    json.dump(labeled.to_dict(), f, indent=2)
            except Exception as e:
                logger.exception(f"  labeling FAILED (continuing): {e}")

        # Stage 3: id_substr key-token ranges.
        kr_path = lab_dir / "key_ranges.json"
        if overwrite or not kr_path.exists():
            try:
                kr = compute_key_token_ranges(mt, prompt, generation, s)
                with open(kr_path, "w") as f:
                    json.dump({"spec_id": s.spec.spec_id, "step_key_token_ranges": kr}, f, indent=2)
            except Exception as e:
                logger.exception(f"  key-range computation FAILED (continuing): {e}")

        # Stage 4: attention.
        if skip_attention:
            continue
        attn_path = attn_dir / "attn.npz"
        if overwrite or not attn_path.exists():
            try:
                attn_dir.mkdir(parents=True, exist_ok=True)
                attn_info = get_attention_matrices(prompt + generation, mt, value_weighted=value_weighted)
                attn = attn_info.attention_matrices.astype(np.float16)
                np.savez_compressed(attn_path, attn=attn,
                                    tokens=np.array(attn_info.tokenized_prompt, dtype=object))
            except Exception as e:
                logger.exception(f"  attention extraction FAILED (continuing): {e}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Main.
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="path to a ConditionDataset json")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    p.add_argument("--version", default="conditions", help="subdir under multistep_clock/")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--value_weighted", type=lambda x: x.lower() in {"1", "true", "yes"}, default=True)
    p.add_argument("--skip_attention", action="store_true", help="skip stage 4 (large tensors)")
    p.add_argument("--limit", type=int, default=None, help="process only the first N samples")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    logging_utils.add_logging_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging_utils.configure(args)
    experiment_utils.set_seed(args.seed)

    # Load + summarize the dataset first (fail fast before the expensive model load).
    ds = ConditionDataset.load(args.dataset)
    samples = ds.samples[:args.limit] if args.limit else ds.samples
    logger.info(f"loaded {len(ds.samples)} samples from {args.dataset}"
                + (f"; using first {len(samples)} (--limit)" if args.limit else ""))
    log_dataset_stats(samples)

    logger.info(f"loading model: {args.model}")
    mt = ModelandTokenizer(model_key=args.model, dtype=torch.bfloat16,
                           device_map="auto", attn_implementation="eager")

    out_root = Path(env_utils.DEFAULT_RESULTS_DIR) / "multistep_clock" / args.version / args.model.split("/")[-1]
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "metadata.json", "w") as f:
        json.dump({"model": args.model, "version": args.version,
                   "n_layers": mt.n_layer, "n_heads": mt.config.num_attention_heads,
                   "max_new_tokens": args.max_new_tokens, "value_weighted": args.value_weighted,
                   "dataset": str(args.dataset), "n_samples": len(samples),
                   "langs": sorted({s.lang for s in samples}),
                   "modularities": sorted({s.modularity for s in samples})}, f, indent=2)

    generate_batched(mt, samples, out_root, args.max_new_tokens, args.batch_size, args.overwrite)
    label_and_attn(mt, samples, out_root, args.value_weighted, args.skip_attention, args.overwrite)
    logger.info(f"done. outputs at {out_root}")


if __name__ == "__main__":
    main()
