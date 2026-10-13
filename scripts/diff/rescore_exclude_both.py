"""Re-score stored transfer results so a BOTH verdict is NOT a hit (src/diff/eval.py, 2026-10-13).

Before the fix, ``judge_causal`` counted ``label in {src, both}`` as triggered and ``eval_readouts``
counted ``patched in {src, both}`` as shifted (and ``valid`` accepted both on the clean readouts).
This recomputes those flags from the stored oracle labels -- no model, no oracle calls -- and the
derived summary rates, in place. Originals are kept next to them as ``*_incl_both`` (idempotent:
re-running never overwrites a saved original).

  python scripts/diff/rescore_exclude_both.py            # all of results/diff/<model>/{kv_transfer,diff_trigger}
  python scripts/diff/rescore_exclude_both.py --dry-run  # report what would change
"""
import argparse
import glob
import json
import os
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
    """the summary fields that depend on the rescored flags (mirrors diff_trigger / kv / hier _summarize)."""
    ro = [r["readout"] for r in recs if isinstance(r.get("readout"), dict) and "shifted" in r["readout"]]
    out = dict(
        causal_trigger_rate=_rate([r["causal"]["triggered"] for r in recs
                                   if isinstance(r.get("causal"), dict) and "triggered" in r["causal"]]),
        readout_shift_rate=_rate([x["shifted"] for x in ro]),
        readout_valid_rate=_rate([x.get("valid") for x in ro]),
        readout_shift_rate_valid=_rate([x["shifted"] for x in ro if x.get("valid")]),
        literal_c_rate=_rate([r["literal_c"]["wrote_c"] for r in recs
                              if isinstance(r.get("literal_c"), dict) and "wrote_c" in r["literal_c"]]),
    )
    co = [r for r in recs if r.get("clean_oracle") and isinstance(r.get("causal"), dict) and "triggered" in r["causal"]]
    if co:
        ok = [r for r in co if r["clean_oracle"]["src"] == "src" and r["clean_oracle"]["dst"] == "dst"]
        out["causal_trigger_rate_clean_valid"] = _rate([r["causal"]["triggered"] for r in ok])
    pairs = [(r["causal"]["label"], r["rule"]["label"]) for r in recs if isinstance(r.get("causal"), dict)
             and "label" in r["causal"] and r.get("rule")]
    if pairs:
        out["oracle_rule_agree_trigger"] = sum((o == "src") == (ru == "src") for o, ru in pairs) / len(pairs)
    return out


def update_summary(path, rates):
    """per-cell summary.json (diff_trigger, maybe a 1-element list) or meta.json (kv / hier)."""
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
    ap.add_argument("--skip", nargs="*", default=[], help="skip result paths containing any of these substrings (e.g. a run still writing)")
    a = ap.parse_args()
    base = ROOT / "results" / "diff" / a.model
    files = sorted(glob.glob(str(base / "kv_transfer" / "**" / "results.json"), recursive=True)
                   + glob.glob(str(base / "diff_trigger" / "**" / "results.json"), recursive=True))
    cell_meta = {}          # cell dir -> new rates (for the top-level summaries)
    n_flags = n_cells = 0
    files = [f for f in files if not any(x in f for x in a.skip)]
    for f in files:
        f = Path(f)
        recs = json.loads(f.read_text())
        recs_list = recs if isinstance(recs, list) else recs.get("records", [])
        k = sum(rescore_record(r) for r in recs_list)
        if not any(isinstance(r.get("causal"), dict) or isinstance(r.get("readout"), dict) for r in recs_list):
            continue                                   # kv-design records (kv_judge) are unaffected
        rates = cell_rates(recs_list)
        cell_meta[f.parent] = rates
        n_flags += k
        for name in ("summary.json", "meta.json"):
            res = update_summary(f.parent / name, rates)
            if res and not a.dry_run:
                (f.parent / name).write_text(json.dumps(res[0], indent=2))
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
    print("note: per-run aggregate files other than diff_trigger/<run>/summary.json (e.g. kv multi_hier "
          "summary__*.json) are NOT recomputed; the figure generators read the per-cell files.")


if __name__ == "__main__":
    main()
