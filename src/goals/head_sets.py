"""Goal-head sets: load the saved ``head_sets/<model>/*.json`` files.

Every reported experiment reads its heads from a json head set (``[[layer, head], ...]``). The default goal set of a
model is ``head_sets/<model>/goal__<tag>_thr<threshold>_band<lo>-<hi>.json``: the trained localization mask
(``scripts/train_head_mask.py``) thresholded inside the model's layer band, exported by
``scripts/data/export_head_sets.py``. When that json is missing, ``load_goal_heads`` derives the set from the mask
checkpoint under ``results/localization/<model>/<tag>/head_mask.pt`` (and says so).
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from src.utils import env_utils

logger = logging.getLogger(__name__)

HEAD_SETS_DIR = os.path.join(env_utils.PROJECT_ROOT, "head_sets")

# Layer band the goal heads are read from, per model. The trained mask covers every layer; the band restricts
# selection to the region where that model's goal heads live. A model missing here falls back to the full depth.
GOAL_BANDS = {
    "Qwen2.5-Coder-32B-Instruct": (25, 56),
    "Qwen2.5-32B-Instruct": (25, 56),  # code axis: matched to Coder (both 64 layers)
    "Llama-3.3-70B-Instruct": (18, 45),
    "Olmo-3.1-32B-Instruct": (10, 36),
    "Olmo-3.1-32B-Instruct-SFT": (10, 36),  # RL axis: matched to Olmo Instruct
}


def default_goal_band(model_name, n_layer=None):
    """`GOAL_BANDS[model]`, else the full depth (needs `n_layer`) — never a foreign model's band."""
    slug = model_name.split("/")[-1]
    if slug in GOAL_BANDS:
        return GOAL_BANDS[slug]
    if n_layer is None:
        raise ValueError(
            f"no GOAL_BANDS entry for {slug!r} — pass --goal_band explicitly (or add it to "
            f"GOAL_BANDS). Falling back to another model's band would select the wrong heads."
        )
    logger.warning(f"no GOAL_BANDS entry for {slug!r} — using the full depth (0, {n_layer})")
    return (0, n_layer)


def goal_heads_path(model_name, threshold=0.9, band=None, tag="lam20", n_layer=None) -> str:
    """Path of the default goal head-set json for ``model_name``."""
    band = band or default_goal_band(model_name, n_layer=n_layer)
    slug = model_name.split("/")[-1]
    return os.path.join(HEAD_SETS_DIR, slug, f"goal__{tag}_thr{threshold:g}_band{band[0]}-{band[1]}.json")


def goal_heads_from_mask(model_name, results_dir=None, threshold=0.9, band=None, tag="lam20", n_layer=None):
    """Threshold the trained localization mask inside the layer band -> sorted [(layer, head)]."""
    results_dir = results_dir or env_utils.DEFAULT_RESULTS_DIR
    ck = torch.load(
        os.path.join(results_dir, f"localization/{model_name.split('/')[-1]}/{tag}/head_mask.pt"),
        map_location="cpu",
    )
    layers = [int(x) for x in ck["layers"]]
    m = ck["mask_values"].numpy().copy()
    band = band or default_goal_band(model_name, n_layer=n_layer)
    m[: band[0]] = 0
    m[band[1]:] = 0
    return sorted((layers[li], int(h)) for li, h in np.argwhere(m > threshold))


def load_goal_heads(model_name, results_dir=None, threshold=0.9, band=None, tag="lam20", n_layer=None):
    """The model's goal heads: the exported json if present, else derived from the mask checkpoint."""
    p = goal_heads_path(model_name, threshold=threshold, band=band, tag=tag, n_layer=n_layer)
    if os.path.exists(p):
        return sorted(tuple(x) for x in json.load(open(p)))
    logger.warning(f"{p} not found — deriving the goal heads from the {tag} mask checkpoint")
    return goal_heads_from_mask(model_name, results_dir, threshold=threshold, band=band, tag=tag, n_layer=n_layer)


def resolve_heads_name(spec: str) -> str:
    return f"json:{Path(spec).stem}" if spec.endswith(".json") else spec


def load_json_heads(spec, n_layer, n_heads):
    """Read + validate a ``[[layer, head], ...]`` json head set.

    A hand-supplied head set is the one code path with no schema behind it, so everything is
    checked here rather than failing deep inside the patching hooks (where an out-of-range head
    silently indexes the wrong slice of the o_proj input instead of raising)."""
    p = Path(spec)
    if not p.exists():
        raise SystemExit(f"--heads {spec!r}: file not found")
    try:
        arr = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"--heads {spec!r}: not valid json ({e})") from None
    if not isinstance(arr, list) or not arr:
        raise SystemExit(
            f"--heads {spec!r}: expected a non-empty json array of [layer, head] pairs, "
            f"got {type(arr).__name__} of length {len(arr) if hasattr(arr, '__len__') else '?'}"
        )
    heads = []
    for i, item in enumerate(arr):
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise SystemExit(f"--heads {spec!r}: entry {i} is {item!r}, want a [layer, head] pair")
        try:
            l, h = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            raise SystemExit(f"--heads {spec!r}: entry {i} = {item!r} is not a pair of ints") from None
        if not (0 <= l < n_layer):
            raise SystemExit(f"--heads {spec!r}: entry {i} has layer {l}, outside [0, {n_layer}) for this model")
        if not (0 <= h < n_heads):
            raise SystemExit(f"--heads {spec!r}: entry {i} has head {h}, outside [0, {n_heads}) for this model")
        heads.append((l, h))
    dupes = {x for x in heads if heads.count(x) > 1}
    if dupes:
        raise SystemExit(
            f"--heads {spec!r}: duplicate entries {sorted(dupes)[:5]}"
            f"{' …' if len(dupes) > 5 else ''} — a repeated head would be steered twice"
        )
    return heads
