"""Regenerate web/manifest.json from results/well-known-tasks/goalscope/.

New structure: { model: { source: [task, ...] } }

Run from the project root:
    python web/generate_manifest.py
"""
import json
import sys
from pathlib import Path

sys.path.append(".")
from src.utils.env_utils import DEFAULT_RESULTS_DIR

goalscope_dir = Path(DEFAULT_RESULTS_DIR) / "well-known-tasks" / "goalscope"
manifest = {}

for model_dir in sorted(goalscope_dir.iterdir()):
    if not model_dir.is_dir():
        continue
    sources = {}
    for source_dir in sorted(model_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        tasks = sorted(p.stem for p in source_dir.glob("*.html"))
        if tasks:
            sources[source_dir.name] = tasks
    if sources:
        manifest[model_dir.name] = sources

out = Path(__file__).parent / "manifest.json"
out.write_text(json.dumps(manifest, indent=2))

total = sum(len(t) for m in manifest.values() for t in m.values())
print(f"Wrote {out}  ({total} tasks across {len(manifest)} models)")
for model, sources in manifest.items():
    for src, tasks in sources.items():
        print(f"  {model}/{src}: {len(tasks)} tasks")
