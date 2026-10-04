"""Build the GoalScope demo site (web/) with the paper-style qualitative visualization.

Data and rendering are decoupled:

    web/data/<model>/<source>/<task>.json    raw GoalScope rows (what apply_goalscope returns):
                                             {"task", "model", "source", "meta": {...},
                                              "rows": [{"pos", "token", "patched_gen", "scoped"}, ...]}
    web/pages/<model>/<source>/<task>.html   rendered by src.goals.goalscope_viz.goalscope_qualitative_html
                                             (the renderer of notebooks/diff/13_goalscope_qualitative.ipynb)
    web/manifest.json                        {model: {source: [task, ...]}} for web/index.html

A new GoalScope run only needs to write data JSON (see ``write_data``), then re-run this script.

    python web/build_site.py                      # render every data JSON + rebuild the manifest
    python web/build_site.py --import-legacy      # first, recover data from the old per-task pages in
                                                  # results/well-known-tasks/goalscope/<model>/<task>.html

Run from the project root.
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path

sys.path.append(".")
from src.goals.goalscope_viz import goalscope_qualitative_html  # noqa: E402

WEB = Path(__file__).resolve().parent
DATA, PAGES = WEB / "data", WEB / "pages"
LEGACY = Path("results/well-known-tasks/goalscope")

# one old token span: <span class="ps-tok ...">TOKEN<span class="ps-tip">pos N['tok']: READOUT</span></span>
_LEGACY_TOK = re.compile(
    r'<span class="ps-tok[^"]*">(.*?)<span class="ps-tip">pos (\d+)\[(.*?)\]: (.*?)</span></span>', re.S)


def write_data(model: str, source: str, task: str, rows: list[dict], meta: dict | None = None) -> Path:
    """Store one task's GoalScope rows (the list returned by ``apply_goalscope(..., show=False)``)."""
    out = DATA / model / source / f"{task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = [{"pos": int(r["pos"]), "token": r.get("token", ""), "patched_gen": r.get("patched_gen") or "",
             "scoped": bool(r.get("scoped", True))} for r in rows]
    out.write_text(json.dumps({"task": task, "model": model, "source": source, "meta": meta or {}, "rows": keep},
                              ensure_ascii=False, indent=1))
    return out


def _tokenizer(model: str):
    """Local (offline) tokenizer for ``model`` — from the HF cache or the shared model dir."""
    import glob
    from transformers import AutoTokenizer
    org = {"Llama": "meta-llama", "Qwen": "Qwen", "Olmo": "allenai"}
    repo = f"models--{next(v for k, v in org.items() if model.startswith(k))}--{model}"
    for snap in (glob.glob(str(Path.home() / ".cache/huggingface/hub" / repo / "snapshots/*/"))
                 + glob.glob(f"/disk/u/models/{repo}/snapshots/*/")):
        try:
            return AutoTokenizer.from_pretrained(snap)
        except Exception:  # noqa: BLE001 — a snapshot without tokenizer files; try the next
            continue
    raise FileNotFoundError(f"no local tokenizer for {model}")


def import_legacy(source: str = "ds_and_algo") -> int:
    """Recover rows from the old pages (they hold only the generated code's tokens, each with its read-out),
    and add the prompt back as dimmed non-scoped rows from results/well-known-tasks/generations/<model>/<task>.json
    (prompt + generation re-tokenized; the old runs counted Llama's BOS, Qwen has none — every saved position is
    checked against the re-tokenization before the page is written)."""
    n, toks = 0, {}
    for page in sorted(LEGACY.glob("*/*.html")):
        model, task = page.parent.name, page.stem
        s = page.read_text(encoding="utf-8")
        scoped = {int(pos): {"pos": int(pos), "token": html.unescape(tok).replace("\u21b5\n", "\n"),
                             "patched_gen": html.unescape(ro), "scoped": True}
                  for tok, pos, _, ro in _LEGACY_TOK.findall(s)}   # old pages showed "\n" as the glyph + "\n"
        if not scoped:
            print(f"  skip {page}: no token spans")
            continue
        rows, meta = list(scoped.values()), {"origin": "legacy page", "legacy_page": str(page)}
        gen_fp = LEGACY.parent / "generations" / model / f"{task}.json"
        if gen_fp.exists():
            g = json.loads(gen_fp.read_text())
            tok = toks.setdefault(model, _tokenizer(model))
            ids = tok(g["prompt"] + g["generation"], add_special_tokens=model.startswith("Llama"))["input_ids"]
            text = [tok.decode([i]) for i in ids]
            if all(p < len(text) and text[p] == r["token"] for p, r in scoped.items()):
                rows = [scoped.get(p, {"pos": p, "token": t, "patched_gen": "", "scoped": False})
                        for p, t in enumerate(text) if not (p == 0 and t == tok.bos_token)]
                meta.update(prompt=g["prompt"], generation=g["generation"])
            else:
                print(f"  {model}/{task}: prompt re-tokenization does not match the saved positions — code only")
        write_data(model, source, task, rows, meta=meta)
        n += 1
    return n


BANK = Path("data/well-known-tasks")
PINS = WEB / "pins.json"   # {"<model>/<source>/<task>": [pos, ...]} read-outs pinned when a page opens


def task_label(source: str, task: str) -> str:
    """Sidebar / title label. HumanEval / MBPP ids become ``<number>_<function name>`` (e.g. ``125_find_length``);
    ds_and_algo tasks keep their name. ``source`` may carry a suffix (``mbpp-v2-smoke``) — the bank dir is its prefix."""
    base = source.split("-", 1)[0]
    m = re.fullmatch(rf"{base}_(\d+)", task)
    if m and (BANK / base / f"{task}.json").exists():
        return f"{m.group(1)}_{json.loads((BANK / base / f'{task}.json').read_text())['function_name']}"
    return task


def build() -> dict:
    manifest, labels = {}, {}
    pins = json.loads(PINS.read_text()) if PINS.exists() else {}
    for fp in sorted(DATA.glob("*/*/*.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        model, source, task = fp.parent.parent.name, fp.parent.name, fp.stem
        out = PAGES / model / source / f"{task}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        # read-outs on the LEFT (the paper layout: they come before the code they anticipate); the site's nav
        # sidebar sits on the right (web/index.html)
        label = labels.setdefault(f"{source}/{task}", task_label(source, task))
        want = pins.get(f"{model}/{source}/{task}", [])
        scoped = {r["pos"] for r in d["rows"] if r["scoped"] and (r.get("patched_gen") or "").strip()}
        if set(want) - scoped:     # positions go stale if the task's data is regenerated
            print(f"  WARNING {model}/{source}/{task}: pins {sorted(set(want) - scoped)} are not scoped tokens")
        goalscope_qualitative_html(d["rows"], title=f"{label.replace('_', ' ')}  ·  {model}", save_path=str(out),
                                   initial_pins=want)
        manifest.setdefault(model, {}).setdefault(source, []).append(task)
    for model, srcs in manifest.items():          # numbered tasks (HumanEval / MBPP) in numeric order
        for source, tasks in srcs.items():
            lab = lambda t: labels[f"{source}/{t}"]
            tasks.sort(key=lambda t: (0, int(lab(t).split("_", 1)[0]), "") if lab(t).split("_", 1)[0].isdigit()
                       else (1, 0, lab(t)))
    (WEB / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (WEB / "labels.json").write_text(json.dumps(labels, indent=1))      # {"<source>/<task>": display label}
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--import-legacy", action="store_true", help="recover data JSON from the old per-task pages")
    args = ap.parse_args()
    if args.import_legacy:
        print(f"imported {import_legacy()} legacy pages -> {DATA}")
    m = build()
    for model, srcs in m.items():
        for src, tasks in srcs.items():
            print(f"  {model}/{src}: {len(tasks)} pages")
    print(f"manifest -> {WEB / 'manifest.json'}")


if __name__ == "__main__":
    main()
