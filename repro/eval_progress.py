"""Progress of the evaluation-wave groups from their logs (repro/eval_lib.sh). Per command of a group: units done of
the full run, seconds per unit over the last 20 minutes (judge included) and a naive ETA of that command. Per
group: judge calls (cache hits, API calls, failed attempts by status), tokens, mean API latency, the spend at
claude-sonnet-4-6's list price ($3 / $15 per million input / output tokens) and the lowest rate-limit budgets left.

    python repro/eval_progress.py [<slug>-<group> ...]        default: every logs/eval-<slug>-<group>.log
"""
import collections
import glob
import json
import os
import re
import sys
from datetime import datetime, timedelta

LOGS = os.environ.get("GOALS_LOGS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"))
PRICE_IN, PRICE_OUT = 3.0, 15.0
TS = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)"
WINDOW = timedelta(minutes=20)


def ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def total_units(cmd):
    """Units of a full run of this command (None: unknown)."""
    if "eval_goalscope" in cmd or "eval_ablation" in cmd:
        return 253
    if "obstacle_persistence_eval" in cmd:
        return 409
    if "kv_transfer_eval" in cmd:
        return 512
    if "q_transfer_eval" in cmd:
        return 128 if "c_in_dst" in cmd else 1024
    return None


def unit_times(cmd, seg):
    """Timestamps of the unit-closing lines of one command's log segment."""
    if "eval_goalscope" in cmd:
        pat = TS + r" \S+ INFO +\[\d+/\d+\] \S+: \d+ pivot read-outs scored"
    elif "eval_ablation" in cmd:
        pat = TS + r" \S+ INFO +\[\d+/\d+\] \S+: base="
    elif "kv_transfer_eval" in cmd:
        pat = TS + r" \S+ INFO +\[[^\]]+\]   \(\d+s\)"
    elif "q_transfer_eval" in cmd:
        pat = TS + r" \S+ INFO +\[[^\]]+\]   => "
    else:
        return None
    return [ts(m.group(1)) for m in re.finditer(pat, seg)]


def bars(seg, name):
    """Last state of each tqdm bar called ``name [<cell>]``: {cell: (done, total, elapsed_s, eta_s)}."""
    out = {}
    pat = re.escape(name) + r" \[([^\]]+)\]: +\d+%\|[^|]*\| (\d+)/(\d+) \[(?:(\d+):)?(\d+):(\d+)<(?:(\d+):)?(\d+|\?):?(\d+)?"
    for m in re.finditer(pat, seg):
        h, mi, s = int(m.group(4) or 0), int(m.group(5)), int(m.group(6))
        el = h * 3600 + mi * 60 + s
        eta = None
        if m.group(8) != "?":
            eh, em, es = int(m.group(7) or 0), int(m.group(8)), int(m.group(9) or 0)
            eta = eh * 3600 + em * 60 + es
        out[m.group(1)] = (int(m.group(2)), int(m.group(3)), el, eta)
    return out


def command_progress(cmd, seg, now):
    times = unit_times(cmd, seg)
    if times is not None:
        n, tot = len(times), total_units(cmd)
        recent = [t for t in times if t >= now - WINDOW]
        rate = None
        if len(recent) >= 2:
            rate = (recent[-1] - recent[0]).total_seconds() / (len(recent) - 1)
        elif len(times) >= 2:
            rate = (times[-1] - times[0]).total_seconds() / (len(times) - 1)
        eta = (now + timedelta(seconds=rate * (tot - n))).strftime("%H:%M") if rate and tot and n < tot else "-"
        return f"{n}/{tot} units, {rate:.1f} s/unit, ETA {eta}" if rate else f"{n}/{tot} units"
    if "obstacle_persistence_eval" in cmd:
        b = bars(seg, "obstacle trials")
        if not b:
            return "loading"
        done, tot, el, eta = next(iter(b.values()))
        end = (now + timedelta(seconds=eta)).strftime("%H:%M") if eta is not None else "-"
        return f"{done}/{tot} trials, {el / max(done, 1):.1f} s/trial, ETA {end}"
    if "temporal_persistence_eval" in cmd:
        b = bars(seg, "contexts")
        if not b:
            return "loading"
        cells = list(b.items())
        done_cells = sum(1 for _, (d, t, _, _) in cells if d == t)
        cur, (d, t, el, eta) = cells[-1]
        spent = sum(v[2] for _, v in cells)
        frac = done_cells + (d / t if d < t else 0)
        n_cells = 8
        end = (now + timedelta(seconds=spent / max(frac, 1e-9) * (n_cells - frac))).strftime("%H:%M") if frac else "-"
        ctx = sum(v[0] for _, v in cells)
        return (f"cell {len(cells)}/{n_cells} {cur} {d}/{t}, {ctx} contexts, {spent / max(ctx, 1):.1f} s/context, "
                f"ETA {end} (cells assumed equal)")
    return "?"


def judge_stats(path):
    if not os.path.exists(path):
        return "no judge log"
    hits = api = 0
    tin = tout = 0
    lat = []
    errs = collections.Counter()
    low = {}
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "err" in d:
            errs[f"{d['err']}({d.get('status')})"] += 1
        elif d.get("hit"):
            hits += 1
        else:
            api += 1
            tin += d.get("input_tokens", 0)
            tout += d.get("output_tokens", 0)
            lat.append(d.get("s", 0))
            for k in ("requests-remaining", "input-tokens-remaining", "output-tokens-remaining"):
                if k in d:
                    low[k] = min(low.get(k, 1 << 60), int(d[k]))
    cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    s = (f"judge: {api} API calls + {hits} cache hits, {tin / 1e6:.2f}M in / {tout / 1e6:.3f}M out tokens, "
         f"${cost:.2f}, mean latency {sum(lat) / len(lat) if lat else 0:.2f} s")
    if errs:
        s += f", failed attempts {dict(errs)}"
    if low:
        s += ", lowest left " + " ".join(f"{k.split('-')[0]}={v}" for k, v in low.items())
    return s


def main():
    names = sys.argv[1:] or sorted(
        os.path.basename(p)[len("eval-"):-len(".log")] for p in glob.glob(f"{LOGS}/eval-*-*.log")
        if ".aborted" not in p and "-lane-" not in p)
    now = datetime.now()
    for name in names:
        text = open(f"{LOGS}/eval-{name}.log", errors="replace").read().replace("\r", "\n")
        runs = list(re.finditer(TS + r" RUN (.*)", text))
        print(f"== {name}")
        for i, m in enumerate(runs):
            seg = text[m.end(): runs[i + 1].start() if i + 1 < len(runs) else len(text)]
            ex = re.search(TS + r" EXIT (\d+) wall=(\d+)s", seg)
            cmd = m.group(2)
            short = re.sub(r"\s+--(model|cuda)[= ]\S+", "", cmd)[:110]
            if ex:
                print(f"  [{m.group(1)[11:16]}] {short}\n      EXIT {ex.group(2)} wall {int(ex.group(3)) / 60:.0f} min")
            else:
                print(f"  [{m.group(1)[11:16]}] {short}\n      running: {command_progress(cmd, seg, now)}")
        print("  " + judge_stats(f"{LOGS}/judge-{name}.jsonl"))


if __name__ == "__main__":
    main()
