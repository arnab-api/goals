"""Re-score stored transfer results so a BOTH verdict is NOT a hit (src/diff/eval.py, 2026-10-13).

Before the fix, ``judge_causal`` counted ``label in {src, both}`` as triggered and ``eval_readouts``
counted ``patched in {src, both}`` as shifted (and ``valid`` accepted both on the clean readouts).
This recomputes those flags from the stored oracle labels -- no model, no oracle calls -- and the
derived summary rates, in place. Originals are kept next to them as ``*_incl_both`` (idempotent:
re-running never overwrites a saved original).

  python scripts/diff/rescore_exclude_both.py            # all of results/diff/<model>/diff_trigger
  python scripts/diff/rescore_exclude_both.py --dry-run  # report what would change
"""
import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _keep(d, key, old):
    if key + "_incl_both" not in d:
        d[key + "_incl_both"] = old


def rescore_record(r):
    """Recompute triggered / valid / shifted / wrote_c from labels. Returns #flags changed."""
    n = 0
    c = r.get("causal")
    if isinstance(c, dict) and "label" in c and "triggered" in c:
        new = c["label"] == "src"
        _keep(c, "triggered", c["triggered"]); n += c["triggered"] != new; c["triggered"] = new
    ro = r.get("readout")
    if isinstance(ro, dict) and "patched" in ro:
        new_s = ro["patched"] == "src"
        new_v = ro.get("source") == "src" and ro.get("dest") == "dst"
        for k, new in (("shifted", new_s), ("valid", new_v)):
            if k in ro:
                _keep(ro, k, ro[k]); n += ro[k] != new; ro[k] = new
    lc = r.get("literal_c")
    if isinstance(lc, dict) and "label" in lc and "wrote_c" in lc:
        new = lc["label"] == "src"
        _keep(lc, "wrote_c", lc["wrote_c"]); n += lc["wrote_c"] != new; lc["wrote_c"] = new
    return n


def _rate(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def cell_rates(recs):
    """the summary fields that depend on the rescored flags (mirrors diff_trigger's _summarize)."""
    ro = [r["readout"] for r in recs if isinstance(r.get("readout"), dict) and "shifted" in r["readout"]]
    return dict(
        causal_trigger_rate=_rate([r["causal"]["triggered"] for r in recs
                                   if isinstance(r.get("causal"), dict) and "triggered" in r["causal"]]),
        readout_shift_rate=_rate([x["shifted"] for x in ro]),
        literal_c_rate=_rate([r["literal_c"]["wrote_c"] for r in recs
                              if isinstance(r.get("literal_c"), dict) and "wrote_c" in r["literal_c"]]),
    )


def update_summary(path, rates):
    """per-cell summary.json (maybe a 1-element list)."""
    if not path.exists():
        return False
    d = json.loads(path.read_text())
    e = d[0] if isinstance(d, list) else d
    changed = False
    for k, v in rates.items():
        if k in e and v is not None:
            _keep(e, k, e[k]); changed |= e[k] != v; e[k] = v
    return d, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    base = ROOT / "results" / "diff" / a.model
    files = sorted(glob.glob(str(base / "diff_trigger" / "**" / "results.json"), recursive=True))
    cell_meta = {}          # cell dir -> new rates (for the top-level summaries)
    n_flags = n_cells = 0
    for f in files:
        f = Path(f)
        recs = json.loads(f.read_text())
        k = sum(rescore_record(r) for r in recs)
        if not any(isinstance(r.get("causal"), dict) or isinstance(r.get("readout"), dict) for r in recs):
            continue                                   # kv-design records (kv_judge, `__kv__` arms) are unaffected
        rates = cell_rates(recs)
        cell_meta[f.parent] = rates
        n_flags += k
        res = update_summary(f.parent / "summary.json", rates)
        if res and not a.dry_run:
            (f.parent / "summary.json").write_text(json.dumps(res[0], indent=2))
        if k:
            n_cells += 1
            print(f"{f.relative_to(base)}: {k} flag(s) changed · causal {rates['causal_trigger_rate']} "
                  f"readout {rates['readout_shift_rate']}")
        if not a.dry_run:
            f.write_text(json.dumps(recs, indent=2))
    # top-level summary lists (diff_trigger: [{variant, datatype, src_lang, dst_lang, ...}])
    for top in sorted(glob.glob(str(base / "diff_trigger" / "*" / "summary.json"))):
        top = Path(top)
        d = json.loads(top.read_text())
        if not isinstance(d, list):
            continue
        hit = 0
        for e in d:
            if not all(k in e for k in ("variant", "datatype", "src_lang", "dst_lang")):
                continue
            tag = f"{e['datatype'].replace('[', '(').replace(']', ')')}__{e['src_lang']}-{e['dst_lang']}"
            rates = cell_meta.get(top.parent / e["variant"] / tag)
            if rates:
                for k2, v in rates.items():
                    if k2 in e and v is not None:
                        _keep(e, k2, e[k2]); e[k2] = v
                hit += 1
        if hit and not a.dry_run:
            top.write_text(json.dumps(d, indent=2))
    print(f"\n{len(files)} result files scanned · {n_cells} cells changed · {n_flags} flags flipped"
          + (" (dry run, nothing written)" if a.dry_run else ""))


if __name__ == "__main__":
    main()
