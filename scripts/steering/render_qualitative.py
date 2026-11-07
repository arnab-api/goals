#!/usr/bin/env python
"""Render cached GoalScope qualitative results to HTML -- no model, no GPU. Iterate on
src/goals/goalscope_viz.py freely and just re-run this.

Usage:
    conda run -n goals python scripts/steering/render_qualitative.py refusal_steered
    conda run -n goals python scripts/steering/render_qualitative.py --all
"""
import argparse
import sys
from pathlib import Path

sys.path.append(".")

from scripts.steering.qualitative_cache import CACHE_DIR, load_cached_result
from src.goals.goalscope_viz import goalscope_qualitative_flow_html

OUT_DIR = Path("notebooks/diff/visualizations")


def render(name: str):
    res, meta = load_cached_result(name)
    title = f"{name} — GoalScope (margin-snap layout)"
    out = OUT_DIR / f"{name}_flow.html"
    goalscope_qualitative_flow_html(res, title=title, save_path=str(out))
    print(f"{name}: {meta.get('gen', '')!r} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    names = [p.stem for p in CACHE_DIR.glob("*.json")] if args.all else args.names
    if not names:
        print("no cached results found under", CACHE_DIR)
        return
    for n in names:
        render(n)


if __name__ == "__main__":
    main()
