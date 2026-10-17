# set_diff localization — Qwen2.5-Coder-32B-Instruct

Query masks trained to make the **source-query transplant** behave like the **set_diff** progress rule
(`first_of(dst_task_list \ source_completed)`) and *not* like its rivals (order_id, complete_count, induction) on the
q-transfer-ALT panel (ALT1–3). Trained 2026-10-14/15 with `scripts/diff/train_alt_localize.py` (branch / worktree
`add-q-transfer-alt`, module `src/diff/alt_localize.py`).

## Files

| file | what |
|---|---|
| `set_diff__head__lam20__thr0.5.json` | **97 whole heads** — `[[layer, head], ...]` (query-head indices, 0-based), the heads with mask value > 0.5 |
| `set_diff__head__lam20__mask_values.json` | their trained (soft) mask values, `{"layer,head": m}` |
| `set_diff__head__lam20.safetensors` | the full head mask, tensor `mask_values` of shape `(64 layers, 40 heads)` in [0, 1] |
| `set_diff__svd__lam50.safetensors` | the full **sub-head (W_QK SVD component) mask**, tensor `mask_values` of shape `(64 layers, 40 heads, 128 components)` in [0, 1]; component 0 = largest singular value |
| `set_diff__svd__lam50__heads_thr0.5.json` | the 506 heads that carry ≥ 1 component with m > 0.5 |
| `set_diff__svd__lam50__summary.json` | counts: 4639 components > 0.5, active components per head, overlap with the goal sets |
| `*.metrics.json` | the training run's train / dev / test scores (per config, pairwise and triplet) |
| `*.mask.pt` | the original torch checkpoints (gitignored — present on nagoya only; the `.safetensors` hold the same tensors) |

The safetensors metadata carries `unit`, `layers` (all 64), the full training `args` (JSON) and the model name.

## The intervention the masks were trained for

Two-prompt **query transfer** (pre-RoPE): cache the source prompt's `q_proj` output at its final token ` =`; in the
destination forward, overwrite the `q_proj` output at the destination's **last** token. Keys / values are untouched.

* **Head mask** (`m[l, h]`), per query head: `q_h ← m · q_src,h + (1 − m) · q_dst,h`
  (`src/diff/query_mask_ops.py::q_blend_hooks`). The 97-head list = binarized at 0.5 (m = 1 on those, 0 elsewhere).
* **SVD mask** (`m[l, h, i]`), per query head: `q_h ← q_dst,h + (q_src,h − q_dst,h) · M_h · diag(m) · M_h⁻¹`,
  where `M_h` maps the head's query to the left-singular coordinates of its `W_QK = W_Q W_Kᵀ` (GQA: the head's key
  group), `src/diff/query_mask_ops.py::svd_bases` / `svd_blend_hooks`. `m = 1` everywhere = the whole-head transplant.
  It was trained and evaluated with the **soft** values (10,216 nonzero entries; 4,639 above 0.5).

## Loading

```python
import json, torch
from safetensors.torch import load_file, safe_open

D = "head_sets/Qwen2.5-Coder-32B-Instruct/set_diff_localization"

# whole heads
heads = [tuple(h) for h in json.load(open(f"{D}/set_diff__head__lam20__thr0.5.json"))]     # 97 (layer, head)

# SVD sub-head mask
mv = load_file(f"{D}/set_diff__svd__lam50.safetensors")["mask_values"]                     # (64, 40, 128)
with safe_open(f"{D}/set_diff__svd__lam50.safetensors", "pt") as f:
    meta = f.metadata()                                                                    # unit / layers / args
```

Per-head projectors `P = M diag(m) M⁻¹` for an intervention (needs the model for the SVD basis):

```python
from src.diff.query_mask_ops import svd_bases            # on master; identical to the training branch

def component_projectors(mt, mv, threshold=None, eps=1e-3):
    """{layer: {head: P (d_head x d_head)}}, plus the sorted (layer, head) list. threshold=None -> soft values
    (as trained); a float binarizes m > threshold. Same logic as src.diff.nrgs_svd_mask.component_projectors."""
    mv = mv.float() if threshold is None else (mv > threshold).float()
    layers = sorted({int(l) for l in torch.nonzero(mv.amax(dim=(1, 2)) > eps).flatten()})
    bases, proj, heads = svd_bases(mt, layers), {}, []
    for l in layers:
        for h in torch.nonzero(mv[l].amax(dim=1) > eps).flatten().tolist():
            m = mv[l, h].to(bases[l]["M"].device)
            proj.setdefault(l, {})[h] = ((bases[l]["M"][h] * m.unsqueeze(0)) @ bases[l]["Minv"][h]).cpu()
            heads.append((l, h))
    return proj, heads
# apply per head at the destination's last token:  q_new = q_dst + (q_src - q_dst) @ P
```

On nagoya, `src.diff.nrgs_svd_mask.component_projectors(mt, f"{D}/set_diff__svd__lam50.mask.pt")` reads the `.pt`
directly, and `scripts/diff/diff_trigger_eval.py --component-mask <that .pt>` accepts it. **Caveat:** that script builds
its run tag from the mask's directory name assuming the NRGS naming, so it would tag these runs ambiguously
(`cm-set_diff_localization-…`); pass a distinct `--tag-suffix` (or fix the parser) before running several masks.

## Training setup and scores (held-out)

Panel: `--panel disjoint` (bank split first: task-disjoint for int/str, family-disjoint for lists; 96 / 16 / 16 groups per
datatype for train / dev / test), `--fork 3way`, loss per instance `softplus(margin − (logp[T] − logp[X]))` with X = the
native token ("causal") or each rival's prediction ("spec", scored as gain over the clean margin; chance ≈ 0.5),
+ λ · mean(m). `triplet` = all three ALT configs of a group succeed (clean triplet = 0.00).

| mask | active units | causal dev / test | spec dev / test | triplet train / dev / test |
|---|---|---|---|---|
| SVD, all heads, λ = 50 | 4,639 components (506 heads) | 0.69 / 0.68 | 0.72 / 0.73 | 0.70 / 0.07 / 0.25 |
| whole heads, λ = 20 | 97 heads | 0.48 / 0.54 | 0.61 / 0.69 | 0.05 / 0.00 / 0.12 |

Overlap with the goal heads — whole-head mask: 52 / 97 in goal-122 (`goal__lam20_thr0.9_band25-56.json`), 65 / 97 in
goal-250 (`goal_topk/goal__lam20_top250.json`), vs ≈ 20 / ≈ 31 expected by chance at matched layers (the mask sits in
layers 44–51). SVD mask: 92 of its 506 heads in goal-122, 154 in goal-250.

Caveats: no random-mask baseline yet; the triplet score does not generalize (overfits train); ALT2 (complete_count
contrast) stays near chance on spec; the masks were scored only by next-token logit margins on this panel — never
with the gaze winner of `notebooks/Figures/slides/qtransfer_hypotheses/07_results` nor with gaze_select_restrict
generation (that is the eval to run).
