"""Peak GPU memory and time of ONE head-mask training step (train_head_mask's lazy+augment step: the source
forward, then the masked destination forward + KL + backward) at several batch sizes, on the longest
destination prompts of the pivot pool (the worst batch an epoch can draw), in one model load.

    CUDA_VISIBLE_DEVICES=<ids> .venv/bin/python repro/batch_probe.py --model-key <hf id> --batch-sizes 16 24 32

Uses train_head_mask.py's pool (the model's pairs file, else the Coder-32B fallback) and the repo's own
functions. Prints one ``REPRO_BATCH {...}`` json line per batch size (``oom: true`` when it does not fit)."""
import argparse
import json
import sys
import time

sys.path.append(".")
import nnterp  # noqa: F401,E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.goals.head_loc_attn import HeadMask, cache_pivot_data, forward_with_src_replacement, get_o_proj_inputs_batch  # noqa: E402
from src.goals.head_loc_data import HeadLocDataset  # noqa: E402
from src.models import ModelandTokenizer  # noqa: E402
from src.tokens import prepare_input  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    args = ap.parse_args()
    ds = HeadLocDataset.load(model_name=args.model_key.split("/")[-1])
    if not ds.pivot_instances:
        ds = HeadLocDataset.load(model_name="Qwen2.5-Coder-32B-Instruct")
    mt = ModelandTokenizer(model_key=args.model_key, dtype=torch.bfloat16, device_map="auto",
                           attn_implementation="eager")
    layers = list(range(mt.num_layers))
    n = len(mt._module.hf_device_map) if hasattr(mt._module, "hf_device_map") else 1
    caches = cache_pivot_data(mt, ds.pivot_instances, ds, layers=layers, precompute=False, cache_batch_size=64,
                              include_docstring=True)
    lens = [len(mt.tokenizer(c.dest_prompt)["input_ids"]) for c in caches]
    order = sorted(range(len(caches)), key=lambda i: -lens[i])
    mt.requires_grad_(False)
    base = {i: torch.cuda.memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())}
    print("REPRO_BASE " + json.dumps({"resident_GiB": {k: round(v, 2) for k, v in base.items()},
                                      "dest_tokens_max": max(lens), "dest_tokens_median": sorted(lens)[len(lens) // 2],
                                      "n_devices_in_map": n}), flush=True)
    for B in args.batch_sizes:
        grp = [caches[i] for i in order[:B]]
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        rec = {"batch": B, "dest_tokens_max": lens[order[0]]}
        try:
            t = time.perf_counter()
            with torch.inference_mode():
                src_o = get_o_proj_inputs_batch(mt, [c.src_prompt for c in grp], layers)
            hv = torch.stack([torch.stack([src_o[ly][j] for ly in layers]) for j in range(B)])
            del src_o
            mask = HeadMask(len(layers), mt.num_heads)
            inputs = prepare_input([c.dest_prompt for c in grp], tokenizer=mt)
            with torch.enable_grad():
                logits = forward_with_src_replacement(mt, inputs, hv, mask, layers)
                lp = torch.stack([c.log_p_target for c in grp]).to(logits.device)
                loss = F.kl_div(lp, F.log_softmax(logits, -1), reduction="batchmean", log_target=True) + 20 * mask.values.mean()
                loss.backward()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            rec["step_s"] = round(time.perf_counter() - t, 2)
            rec["oom"] = False
            del logits, loss, hv
        except torch.OutOfMemoryError:
            rec["oom"] = True
        rec["peak_alloc_GiB"] = {i: round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())}
        rec["activation_GiB_total"] = round(sum(rec["peak_alloc_GiB"][i] - base[i] for i in base), 1)
        print("REPRO_BATCH " + json.dumps(rec), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
