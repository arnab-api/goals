"""Export a gated subspace mask (scripts/train_subspace_loc.py) as a Subspace Probe v2 checkpoint, so the GoalScope
causality / faithfulness evals (scripts/evaluation/eval_subspace_probe_v2.py) can read it out.

Exact: the mask patches with W_lᵀ diag(g_l) W_l = (√g_l ⊙ W_l)ᵀ (√g_l ⊙ W_l), so the probe gets W'_l = √g_l ⊙ W_l (rows
scaled by the root of their gate; closed gates give zero rows) and reads out o = Σ_l W'_lᵀ W'_l h_l at the mask's
layers. The steer band / α are the GoalScope eval's (additive_norm, α 0.1, layers 35-45), as for the goal heads.

    python -m scripts.evaluation.export_subspace_mask_probe --mask <.../subspace_mask.pt> [--out <probe.pt>]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(".")
from src.goals.subspace_probe_v2 import SubspaceV2Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", default=None, help="default: <mask dir>/probe.pt")
    ap.add_argument("--steer_lo", type=int, default=35)
    ap.add_argument("--steer_hi", type=int, default=45, help="inclusive (35-45 = the GoalScope band)")
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    m = torch.load(a.mask, map_location="cpu", weights_only=False)
    W, g, layers = m["W"].float(), m["g"].float().clamp(0, 1), list(m["layers"])
    Wp = W * g.sqrt().unsqueeze(-1)                      # (L, r, d): rows scaled by √gate
    cfg = SubspaceV2Config(d_model=W.shape[-1], r=W.shape[1], read_layer=layers[-1], read_layers=layers,
                           steer_layers=list(range(a.steer_lo, a.steer_hi + 1)), alpha=a.alpha)
    out = Path(a.out) if a.out else Path(a.mask).with_name("probe.pt")
    torch.save(dict(W=Wp, config=cfg, epoch=m["config"]["n_epochs"], read_layer=layers[-1], read_layers=layers,
                    steer_layers=cfg.steer_layers, model=m["config"]["model"], source_mask=str(a.mask),
                    open_directions=(g > 0.5).sum(1).tolist()), out)
    print(f"exported {a.mask} -> {out} | layers {layers[0]}-{layers[-1]} | open directions per layer "
          f"{(g > 0.5).sum(1).tolist()} (total {int((g > 0.5).sum())}) | steer {a.steer_lo}-{a.steer_hi} α {a.alpha}")


if __name__ == "__main__":
    main()
