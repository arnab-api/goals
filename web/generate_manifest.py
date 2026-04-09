"""Regenerate web/manifest.json from results/well-known-tasks/goalscope/.

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
    tasks = sorted(p.stem for p in model_dir.glob("*.html"))
    if tasks:
        manifest[model_dir.name] = tasks

out = Path(__file__).parent / "manifest.json"
out.write_text(json.dumps(manifest, indent=2))
print(f"Wrote {out}  ({sum(len(v) for v in manifest.values())} tasks across {len(manifest)} models)")
