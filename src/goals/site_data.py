"""Write data for the GoalScope demo site (the goals-web repo, https://goals.baulab.info).

The site's source lives in goals-web/site/; this module writes the data files it renders. Set ``GOALS_WEB_SITE`` to
that directory (default: ``<goals repo>/goals-web/site``, where goals-web is checked out inside this repo, gitignored).

    site/data/<model>/<source>/<task>.json   code-task pages (``write_data``; rendered by goals-web/site/build_site.py)
    site/demos/data/icl/<id>.json            ICL tab, one few-shot prompt (``write_icl_example``)
    site/demos/data/em/<id>.json             chat / EM tab, one innocent-vs-EM prompt pair (``write_em_example``)
    site/demos/data/{icl,em}/index.json      rebuilt by ``rebuild_indexes``

Token rows use the keys ``pos`` / ``token`` / ``patched_gen`` / ``scoped`` (the prompt is shown dimmed; scoped tokens
carry a read-out), plus ``verdict`` (ICL: the LM judge's "yes" | "no") or ``tone`` (EM: "benign" | "concerning";
"concerning" underlines the token in red). ``pins`` lists the positions pinned when an example opens.
"""

import json
import os
from pathlib import Path

from src.utils import env_utils

SITE = Path(os.environ.get("GOALS_WEB_SITE", os.path.join(env_utils.PROJECT_ROOT, "goals-web", "site")))
CODE_DATA = SITE / "data"
DEMO_DATA = SITE / "demos" / "data"


def write_data(model: str, source: str, task: str, rows: list[dict], meta: dict | None = None) -> Path:
    """Store one code task's GoalScope rows (the list returned by ``apply_goalscope(..., show=False)``)."""
    out = CODE_DATA / model / source / f"{task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = [{"pos": int(r["pos"]), "token": r.get("token", ""), "patched_gen": r.get("patched_gen") or "",
             "scoped": bool(r.get("scoped", True))} for r in rows]
    out.write_text(json.dumps({"task": task, "model": model, "source": source, "meta": meta or {}, "rows": keep},
                              ensure_ascii=False, indent=1))
    return out


def _write(kind: str, ex: dict) -> Path:
    out = DEMO_DATA / kind / f"{ex['id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ex, ensure_ascii=False, indent=1))
    return out


def write_icl_example(ex_id: str, *, task: str, category: str, rule: str, rows: list[dict], pins: list[int],
                      continuation: str = "", accuracy: float | None = None, model: str = "Qwen2.5-Coder-32B-Instruct",
                      meta: dict | None = None) -> Path:
    """One few-shot prompt. ``category`` ∈ {abstractive, extractive, composition, superposition};
    ``continuation`` = text shown after the rows (empty when the generation is part of ``rows``);
    ``accuracy`` = the task's judged verbalization rate (shown next to the task name)."""
    return _write("icl", {"id": ex_id, "task": task, "category": category, "rule": rule, "accuracy": accuracy,
                          "model": model, "rows": rows, "continuation": continuation, "pins": pins,
                          "meta": meta or {}})


def write_em_example(ex_id: str, *, title: str, caption: str, sides: dict[str, dict], meta: dict | None = None) -> Path:
    """One prompt answered by both models. ``sides`` = {"innocent": {...}, "em": {...}}, each with ``model``,
    ``rows`` (prompt + response, read-outs on the scoped positions), ``continuation`` and ``pins``."""
    return _write("em", {"id": ex_id, "title": title, "caption": caption, "sides": sides, "meta": meta or {}})


def rebuild_indexes() -> None:
    """Rebuild demos/data/{icl,em}/index.json from the example files (ICL grouped by category, EM by meta.order)."""
    icl = [json.loads(f.read_text()) for f in sorted((DEMO_DATA / "icl").glob("*.json")) if f.name != "index.json"]
    order = {"abstractive": 0, "extractive": 1, "composition": 2, "superposition": 3}
    icl.sort(key=lambda e: (order.get(e["category"], 9), e["task"], e["id"]))
    (DEMO_DATA / "icl" / "index.json").write_text(json.dumps({"examples": [
        {k: e.get(k) for k in ("id", "task", "category", "rule", "accuracy")} for e in icl]}, indent=1))
    em = [json.loads(f.read_text()) for f in sorted((DEMO_DATA / "em").glob("*.json")) if f.name != "index.json"]
    em.sort(key=lambda e: e.get("meta", {}).get("order", 99))
    (DEMO_DATA / "em" / "index.json").write_text(json.dumps({"examples": [
        {"id": e["id"], "title": e["title"]} for e in em]}, indent=1))
