"""Progress of scripts.mine_pivot_instances from its log: Stage 2 position (by pair, weighted by candidates).

    python repro/mining_progress.py <log>
"""
import re
import sys
from datetime import datetime

text = open(sys.argv[1], errors="replace").read().replace("\r", "\n")
order, cands = [], {}
for m in re.finditer(r"Stage 1 \[(\S+)\]: (\d+) raw → (\d+) deduped", text):
    order.append(m.group(1))
    cands[m.group(1)] = int(m.group(3))
val = list(re.finditer(r"^(\S+ \S+) .*Validated: (\S+) prefix", text, re.M))
t1 = re.search(r"^(\S+ \S+) .*Stage 1 \[", text, re.M)
total = sum(cands.values())
out = {"pairs": len(order), "candidates": total, "validated": len(val)}
if val and t1:
    cur = val[-1].group(2)
    i = order.index(cur)
    done = sum(cands[p] for p in order[:i]) + cands[cur] / 2
    t0 = datetime.strptime(t1.group(1), "%Y-%m-%d %H:%M:%S")
    tn = datetime.strptime(val[-1].group(1), "%Y-%m-%d %H:%M:%S")
    el = (tn - t0).total_seconds()
    out.update(pair_index=i + 1, frac_stage2=round(done / total, 3), elapsed_s=el,
               stage2_eta_s=round(el / max(done, 1) * (total - done)))
s2 = re.search(r"^(\S+ \S+) .*Stage 2: (\d+) validated", text, re.M)
if s2:
    out["stage2_done"] = s2.group(1) + f" ({s2.group(2)} validated)"
s3 = re.search(r"^(\S+ \S+) .*Stage 3: (\d+) augmented", text, re.M)
if s3:
    out["stage3_done"] = s3.group(1) + f" ({s3.group(2)} augmented)"
print(out)
