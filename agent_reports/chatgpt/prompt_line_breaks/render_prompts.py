"""Make prompt illustrations with literal \\n at each actual newline.

The raw prompt files are preserved. A final line without a newline receives
no marker, including the source prompt ending immediately after the pivot '='.
Run this script after changing the raw prompt files.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROMPTS = ROOT / "Overleaf-draft/Goals-Overleaf/Text/Appendix/prompts"


def main():
    display = PROMPTS / "display"
    display.mkdir(exist_ok=True)
    for source in sorted(PROMPTS.glob("*.txt")):
        raw = source.read_bytes()
        illustrated = raw.replace(b"\n", b"\\n\n")
        (display / source.name).write_bytes(illustrated)


if __name__ == "__main__":
    main()
