"""Export a fitted MSE subspace (scripts/train_subspace_mse.py, <fit dir>/subspace.pt) as a Subspace Probe v2
checkpoint for the GoalScope evals: o = Σ_l W_lᵀ W_l h_l at the fit's read layers, steered with GoalScope's
additive_norm (α 0.1, layers 35-45).

    python -m scripts.evaluation.export_subspace_mse_probe <fit dir> [<fit dir> ...]
"""

import sys
from pathlib import Path

import torch

sys.path.append(".")
from src.goals.subspace_probe_v2 import SubspaceV2Config  # noqa: E402


def export(fit_dir: Path, model: str = "Qwen/Qwen2.5-Coder-32B-Instruct") -> Path:
    s = torch.load(fit_dir / "subspace.pt", map_location="cpu", weights_only=False)
    W, layers = s["W"].float(), list(s["layers"])
    cfg = SubspaceV2Config(d_model=W.shape[-1], r=W.shape[1], read_layer=layers[-1],
                           read_layers=layers if len(layers) > 1 else None, steer_layers=list(range(35, 46)), alpha=0.1)
    out = fit_dir / "probe.pt"
    torch.save(dict(W=W, config=cfg, epoch=0, read_layer=layers[-1], read_layers=layers, steer_layers=cfg.steer_layers,
                    model=model, source=f"subspace_mse:{fit_dir.name}"), out)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print("exported", export(Path(d)))
