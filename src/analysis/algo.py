"""Direction-finding & linear-algebra primitives for latent analysis.

Three groups live here:

  * SVD / low-rank        — `SVD` dataclass (moved from `src.utils.typing`),
                            `low_rank_approx`, `low_rank_pinv` (moved from `src.functional`).
  * Direction extraction  — `mean_diff_direction` (mass-mean-diff) and
                            `learn_contrastive_direction` (hinge-loss contrastive probe,
                            lifted from `scripts/goal_dir.py`).
  * Projection / PCA      — `compute_projection` + `projection_across_layers`: project a set
                            of contrastive latents onto either a *supplied* direction
                            (e.g. v_stop / v_cont) or the top principal components, with
                            ROC-AUC separability diagnostics. Generalizes the per-layer
                            projection cell of `notebooks/clock/04.3_eval.ipynb`.
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# ══════════════════════════════════════════════════════════════════════
# SVD + low-rank (moved from src.utils.typing / src.functional)
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=False, kw_only=True)
class SVD:
    U: torch.Tensor
    S: torch.Tensor
    V: torch.Tensor

    @property
    def shape(self):
        return self.U.shape, self.S.shape, self.V.shape

    def to_device(self, device: torch.device):
        """in-place device change -- to save memory"""
        self.U = self.U.to(device)
        self.S = self.S.to(device)
        self.V = self.V.to(device)
        return self

    def to_dtype(self, dtype: torch.dtype):
        """in-place dtype change -- to save memory"""
        self.U = self.U.to(dtype)
        self.S = self.S.to(dtype)
        self.V = self.V.to(dtype)
        return self

    @property
    def dtype(self):
        return self.U.dtype

    @property
    def device(self):
        return self.U.device

    @staticmethod
    def calculate(matrix: torch.Tensor) -> "SVD":
        n, d = matrix.shape
        if n >= d:
            U, S, Vh = torch.linalg.svd(
                matrix.to(dtype=torch.float32), full_matrices=False
            )
            V = Vh.mH
        else:
            U, S, Vh = torch.linalg.svd(
                matrix.to(dtype=torch.float32), full_matrices=True
            )
            V = Vh.mH

        svd = SVD(U=U, S=S, V=V)
        svd.to_dtype(matrix.dtype)
        svd.to_device(matrix.device)
        return svd


def low_rank_approx(
    matrix: torch.Tensor, rank: int, svd: SVD | None = None
) -> torch.Tensor:
    """Compute a low-rank approximation of a matrix.

    Args:
        matrix: The matrix to approximate.
        rank: The rank of the approximation.

    Returns:
        The approximation.
    """
    if svd is None:
        svd = SVD.calculate(matrix.float())
    u, s, v = svd.U, svd.S, svd.V
    matrix_approx = u[:, :rank] @ torch.diag(s[:rank]) @ v[:, :rank].T
    return matrix_approx.to(matrix.dtype)


def low_rank_pinv(
    matrix: torch.Tensor, rank: int, svd: SVD | None = None
) -> torch.Tensor:
    """Compute a low-rank pseudo-inverse of a matrix.

    Args:
        matrix: The matrix to invert.
        rank: The rank of the approximation.

    Returns:
        The pseudo-inverse.
    """
    if svd is None:
        svd = SVD.calculate(matrix.float())
    u, s, v = svd.U, svd.S, svd.V
    matrix_pinv = v[:, :rank] @ torch.diag(1 / s[:rank]) @ u[:, :rank].T
    return matrix_pinv.to(matrix.dtype)


# ══════════════════════════════════════════════════════════════════════
# small numpy helpers
# ══════════════════════════════════════════════════════════════════════
def unit(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize along `axis` (rows by default), guarding against zero norm."""
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


# ══════════════════════════════════════════════════════════════════════
# direction extraction
# ══════════════════════════════════════════════════════════════════════
def mean_diff_direction(
    z_pos: np.ndarray, z_neg: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Mass-mean-difference direction: unit( E[z_pos] - E[z_neg] ).

    With paired (minimal) contrastive inputs this is the paired-difference mean. If
    `normalize`, each sample is put on the unit sphere first (cosine geometry), matching
    the projection convention used by `compute_projection`.
    """
    z_pos = np.asarray(z_pos, dtype=np.float64)
    z_neg = np.asarray(z_neg, dtype=np.float64)
    if normalize:
        z_pos, z_neg = unit(z_pos), unit(z_neg)
    v = (z_pos - z_neg).mean(axis=0)
    return unit(v)


@torch.enable_grad()
def learn_contrastive_direction(
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    l1_lambda: float = 0.0,
    l2_lambda: float = 1e-3,
    margin: float = 1.0,
    lr: float = 1e-2,
    n_steps: int = 1000,
    patience: int = 50,
) -> tuple[torch.Tensor, dict]:
    """Learn a direction via the contrastive hinge objective (lifted from `scripts/goal_dir.py`).

    Minimizes:  l1*||v||_1 + l2*||v||_2^2 + (1/N) sum_i max(0, margin - v.(z+_i - z-_i))

    `z_pos`/`z_neg` are L2-normalized per sample before training so the margin is a
    consistent cosine-space threshold across layers (raw activation norms vary a lot).
    Returns the unit-normalized direction and a diagnostics dict.
    """
    if not torch.is_tensor(z_pos):
        z_pos = torch.as_tensor(np.asarray(z_pos), dtype=torch.float32)
    if not torch.is_tensor(z_neg):
        z_neg = torch.as_tensor(np.asarray(z_neg), dtype=torch.float32)
    z_pos = z_pos.float()
    z_neg = z_neg.float()

    device = z_pos.device
    hidden_dim = z_pos.shape[1]

    z_pos = z_pos / z_pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    z_neg = z_neg / z_neg.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    delta = z_pos - z_neg  # (n_pairs, hidden_dim)

    v = torch.randn(hidden_dim, device=device, dtype=z_pos.dtype) * 0.01
    v.requires_grad_(True)

    optimizer = torch.optim.Adam([v], lr=lr)

    best_loss = float("inf")
    patience_counter = 0
    final_step = n_steps

    for step in range(n_steps):
        optimizer.zero_grad()

        scores = delta @ v
        hinge_loss = torch.clamp(margin - scores, min=0).mean()

        reg_loss = torch.zeros(1, device=device, dtype=z_pos.dtype)
        if l1_lambda > 0:
            reg_loss = reg_loss + l1_lambda * v.abs().sum()
        if l2_lambda > 0:
            reg_loss = reg_loss + l2_lambda * (v @ v)

        loss = hinge_loss + reg_loss.squeeze()

        loss.backward()
        optimizer.step()

        curr_loss = loss.item()
        if curr_loss < best_loss - 1e-6:
            best_loss = curr_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            final_step = step + 1
            break

    with torch.no_grad():
        v_final = v.detach()
        v_norm = v_final.norm()
        v_normalized = v_final / (v_norm + 1e-8)

        scores_final = delta @ v_normalized
        constraint_sat = (scores_final > 0).float().mean().item()
        sparsity = (v_normalized.abs() < 1e-2).float().mean().item()

    return v_normalized, {
        "final_loss": best_loss,
        "steps": final_step,
        "constraint_satisfaction": constraint_sat,
        "sparsity": sparsity,
        "v_norm_pre_normalize": v_norm.item(),
    }


# ══════════════════════════════════════════════════════════════════════
# projection / PCA separability
# ══════════════════════════════════════════════════════════════════════
@dataclass
class ProjectionResult:
    """2-D scatter coordinates + separability diagnostics for one set of contrastive latents.

    `pos_*` correspond to `z_pos` (label 1), `neg_*` to `z_neg` (label 0). The `auc_signal`
    / `auc_resid` / `resid_var` diagnostics are always computed from the supplied `direction`
    (independent of which axis is plotted), so they stay comparable across `axis` modes.
    """

    pos_x: np.ndarray
    neg_x: np.ndarray
    pos_y: np.ndarray
    neg_y: np.ndarray
    xlabel: str
    ylabel: str
    auc_x: float  # AUC of whatever is on the x-axis
    auc_signal: float  # AUC of projection onto `direction`
    auc_resid: float  # AUC of the top residual PC (after removing `direction`)
    resid_var: float  # variance fraction held by that residual PC


def compute_projection(
    z_pos: np.ndarray,
    z_neg: np.ndarray,
    direction: np.ndarray,
    axis: str = "direction",
    sign: float = 1.0,
    signal_name: str = "signal",
    normalize: bool = True,
) -> ProjectionResult:
    """Project `z_pos`/`z_neg` to 2-D for a separability scatter.

    `direction` is the contrast direction (e.g. v_stop); the signal diagnostics are always
    measured against it. The plotted x-axis is chosen by `axis`:

      * ``"direction"`` — x = (`sign`) * projection onto `direction`, y = top residual PC.
        Use ``sign=-1`` to show v_cont (the mirror of v_stop).
      * ``"pca"``       — plain 2-D PCA of the pooled latents: x = PC1, y = PC2.
    """
    assert axis in ("direction", "pca"), axis
    z_pos = np.asarray(z_pos, dtype=np.float64)
    z_neg = np.asarray(z_neg, dtype=np.float64)
    direction = unit(np.asarray(direction, dtype=np.float64))
    if normalize:
        z_pos, z_neg = unit(z_pos), unit(z_neg)

    n_pos = len(z_pos)
    all_z = np.concatenate([z_pos, z_neg], axis=0)  # pos (1) then neg (0)
    labels = np.concatenate([np.ones(n_pos), np.zeros(len(z_neg))])

    # signal diagnostics from `direction` (always, independent of axis)
    proj = all_z @ direction
    z_resid = all_z - np.outer(proj, direction)
    pca_res = PCA(n_components=1).fit(z_resid)
    proj_resid = pca_res.transform(z_resid).squeeze()
    auc_signal = roc_auc_score(labels, proj)
    auc_resid = roc_auc_score(labels, proj_resid)
    resid_var = float(pca_res.explained_variance_ratio_[0])

    if axis == "direction":
        x = sign * proj
        y = proj_resid
        auc_x = roc_auc_score(labels, x)
        xlabel = f"{signal_name} projection  (AUC={auc_x:.3f})"
        ylabel = f"top residual PC  ({resid_var:.1%} var, AUC={auc_resid:.3f})"
    else:  # pca
        pca2 = PCA(n_components=2).fit(all_z)
        Z = pca2.transform(all_z)
        x, y = Z[:, 0], Z[:, 1]
        vr = pca2.explained_variance_ratio_
        auc_x = roc_auc_score(labels, x)
        xlabel = f"PC1  ({vr[0]:.1%} var, AUC={auc_x:.3f})"
        ylabel = f"PC2  ({vr[1]:.1%} var, AUC={roc_auc_score(labels, y):.3f})"

    return ProjectionResult(
        pos_x=x[:n_pos], neg_x=x[n_pos:],
        pos_y=y[:n_pos], neg_y=y[n_pos:],
        xlabel=xlabel, ylabel=ylabel,
        auc_x=auc_x, auc_signal=auc_signal,
        auc_resid=auc_resid, resid_var=resid_var,
    )


def projection_across_layers(
    H_pos: np.ndarray,
    H_neg: np.ndarray,
    layers,
    axis: str = "pca",
    sign: float = 1.0,
    signal_name: str = "signal",
    train_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    normalize: bool = True,
) -> dict[int, ProjectionResult]:
    """Per-layer `compute_projection`, fitting the mean-diff `direction` on a train split.

    `H_pos`/`H_neg` are `(n, n_layers, d)`. The direction (v_stop) is fit on `train_idx`
    and points are scattered for `test_idx` (defaults to all rows for both). Returns
    `{layer: ProjectionResult}`.
    """
    n = H_pos.shape[0]
    if train_idx is None:
        train_idx = np.arange(n)
    if test_idx is None:
        test_idx = np.arange(n)

    results: dict[int, ProjectionResult] = {}
    for L in layers:
        zp_tr, zn_tr = H_pos[train_idx, L].astype(np.float64), H_neg[train_idx, L].astype(np.float64)
        zp_te, zn_te = H_pos[test_idx, L].astype(np.float64), H_neg[test_idx, L].astype(np.float64)
        if normalize:
            zp_tr, zn_tr, zp_te, zn_te = unit(zp_tr), unit(zn_tr), unit(zp_te), unit(zn_te)
        v = mean_diff_direction(zp_tr, zn_tr, normalize=False)  # already unit-normalized
        results[int(L)] = compute_projection(
            zp_te, zn_te, v,
            axis=axis, sign=sign, signal_name=signal_name, normalize=False,
        )
    return results
