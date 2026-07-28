"""Patch existing GoalScope HTML files to add font-family to #ps-tooltip.

Adds `font-family: 'Fira Code', 'Cascadia Code', monospace;` to the
#ps-tooltip CSS rule in all HTML files under the given directory.
Skips files that already have the font set.

Usage
-----
python scripts/patch_goalscope_tooltip_font.py
python scripts/patch_goalscope_tooltip_font.py --dir results/well-known-tasks/goalscope/Qwen2.5-Coder-32B-Instruct
python scripts/patch_goalscope_tooltip_font.py --dry-run
"""

import argparse
import sys
from pathlib import Path

OLD = "    color: #e8e8e8; font-size: 12px;"
NEW = "    color: #e8e8e8; font-size: 12px;\n    font-family: 'Fira Code', 'Cascadia Code', monospace;"


def patch_file(path: Path, dry_run: bool) -> bool:
    text = path.read_text(encoding="utf-8")
    if "font-family" in text[text.find("#ps-tooltip"):text.find("#ps-tooltip") + 300]:
        return False  # already patched
    if OLD not in text:
        print(f"  SKIP (unexpected format): {path}")
        return False
    patched = text.replace(OLD, NEW, 1)
    if not dry_run:
        path.write_text(patched, encoding="utf-8")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        default="results/well-known-tasks/goalscope/Qwen2.5-Coder-32B-Instruct",
        help="Root directory to search for HTML files.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would change without writing.")
    args = p.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"Directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    files = sorted(root.rglob("*.html"))
    print(f"Found {len(files)} HTML files under {root}")

    patched = skipped = 0
    for f in files:
        if patch_file(f, args.dry_run):
            print(f"  {'[dry] ' if args.dry_run else ''}patched: {f.relative_to(root)}")
            patched += 1
        else:
            skipped += 1

    print(f"\nDone. patched={patched}  already-ok/skipped={skipped}")


if __name__ == "__main__":
    main()
