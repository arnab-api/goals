"""Plotting helpers for latent analysis.

  * `matrix_heatmap`     — generic RdBu cross-section heatmap (moved from `src.plotting`).
  * `animate_projection` — animate per-layer 2-D projection scatters into a GIF
                           (replicates the `03_probe` contrastive.gif; lifted out of
                           `notebooks/clock/04.3_eval.ipynb`).
"""

import os
from typing import Callable, Mapping, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analysis.algo import ProjectionResult
from src.utils.typing import ArrayLike, PathLike


def matrix_heatmap(
    matrix: ArrayLike,
    limit_dim: int = 100,
    canvas: plt = plt,
    save_path: PathLike | None = None,
    title: str | None = None,
    tick_gap: int | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    tick_labels: list[str] | None = None,
) -> None:
    """Plot cross section of matrix as a heatmap."""

    limit_dim = min(limit_dim, matrix.shape[0])

    matrix = torch.stack([w[:limit_dim] for w in matrix[:limit_dim]]).cpu()
    limit = max(abs(matrix.min().item()), abs(matrix.max().item()))
    img = plt.imshow(
        matrix, cmap="RdBu", interpolation="nearest", vmin=-limit, vmax=limit
    )
    canvas.colorbar(img, orientation="vertical")

    if x_label is not None:
        canvas.xlabel(x_label)
    if y_label is not None:
        canvas.ylabel(y_label)
    if tick_gap is not None:
        canvas.xticks(range(0, limit_dim, tick_gap), range(0, limit_dim, tick_gap))
        canvas.yticks(range(0, limit_dim, tick_gap), range(0, limit_dim, tick_gap))
    if tick_labels is not None:
        canvas.xticks(range(limit_dim), tick_labels, rotation=90)
        canvas.yticks(range(limit_dim), tick_labels)

    if title is not None:
        canvas.title(title)
    if save_path is not None:
        canvas.savefig(str(save_path))
    canvas.show()


def animate_projection(
    proj_results: Mapping[int, ProjectionResult],
    pos_label: str = "positive",
    neg_label: str = "negative",
    pos_color: str = "tab:red",
    neg_color: str = "tab:green",
    title_fn: Optional[Callable[[int], str]] = None,
    save_path: PathLike | None = None,
    figsize: tuple[float, float] = (8, 7),
    interval: int = 300,
    fps: int = 2,
    dpi: int = 110,
) -> animation.FuncAnimation:
    """Animate per-layer projection scatters (one frame per layer) and optionally save a GIF.

    `proj_results` maps `layer -> ProjectionResult` (from `src.analysis.algo`). Axis limits
    are held fixed across frames. `title_fn(layer)` builds each frame title (defaults to
    ``"Layer {L}"``). Returns the `FuncAnimation` (call `.to_jshtml()` to embed in a notebook).
    """
    results = dict(proj_results)
    frames = list(results.keys())
    if title_fn is None:
        title_fn = lambda L: f"Layer {L}"  # noqa: E731

    fig, ax = plt.subplots(figsize=figsize)
    # stable axis limits across all frames
    all_x = np.concatenate([np.r_[r.pos_x, r.neg_x] for r in results.values()])
    all_y = np.concatenate([np.r_[r.pos_y, r.neg_y] for r in results.values()])
    px = (all_x.max() - all_x.min()) * 0.05
    py = (all_y.max() - all_y.min()) * 0.05
    ax.set_xlim(all_x.min() - px, all_x.max() + px)
    ax.set_ylim(all_y.min() - py, all_y.max() + py)

    neg_sc = ax.scatter([], [], alpha=0.25, s=10, c=neg_color, label=neg_label)
    pos_sc = ax.scatter([], [], alpha=0.25, s=10, c=pos_color, label=pos_label)
    ax.legend(markerscale=3, loc="upper right")
    title = ax.set_title("")

    def update(fi):
        L = frames[fi]
        r = results[L]
        pos_sc.set_offsets(np.column_stack([r.pos_x, r.pos_y]))
        neg_sc.set_offsets(np.column_stack([r.neg_x, r.neg_y]))
        ax.set_xlabel(r.xlabel)
        ax.set_ylabel(r.ylabel)
        title.set_text(title_fn(L))
        return pos_sc, neg_sc, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=interval, blit=False
    )
    if save_path is not None:
        os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
        anim.save(str(save_path), writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    return anim
