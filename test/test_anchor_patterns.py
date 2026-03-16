"""Minimal test for Phase 1: anchor pattern refactor.

Tests:
1. New dataclass fields (anchor_patterns, short_description, augmentation)
2. convert_anchor_tokens_to_patterns utility
3. Regex matching in label_predicate_spans / _label_leaf
4. extract_predicate_clauses with regex patterns
5. Dataset load + roundtrip
"""
import re
import sys
sys.path.insert(0, ".")

from src.goals.anchor import (
    convert_anchor_tokens_to_patterns,
    label_predicate_spans,
    extract_predicate_clauses,
    show_span_annotation,
)
from src.goals.predicate_data import (
    MultiPredicateDataset,
    PredicateSpec,
    MultiPredicateSample,
)
from src.goals.data import IncompleteCodeSample

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    results.append(condition)

# ── 1. New dataclass fields ───────────────────────────────────────────────────
print("\n1. Dataclass fields")

ps = PredicateSpec(description="check palindrome")
check("PredicateSpec has anchor_patterns", hasattr(ps, "anchor_patterns"))
check("PredicateSpec.anchor_patterns default is []", ps.anchor_patterns == [])
check("PredicateSpec has short_description", hasattr(ps, "short_description"))
check("PredicateSpec.short_description default is ''", ps.short_description == "")
check("PredicateSpec no anchor_tokens", not hasattr(ps, "anchor_tokens"))

cs = IncompleteCodeSample(instruction="x", code="y")
check("IncompleteCodeSample has augmentation", hasattr(cs, "augmentation"))
check("IncompleteCodeSample.augmentation default is None", cs.augmentation is None)

# MultiPredicateSample (inherits IncompleteCodeSample, adds p_minus + augmentation)
ms = MultiPredicateSample(instruction="x", code="y")
check("MultiPredicateSample has augmentation", hasattr(ms, "augmentation"))
check("MultiPredicateSample.augmentation default is None", ms.augmentation is None)

# ── 2. convert_anchor_tokens_to_patterns ─────────────────────────────────────
print("\n2. convert_anchor_tokens_to_patterns")

patterns = convert_anchor_tokens_to_patterns(
    ["s[::-1]", "len(s)", "[::-1]", "n & (n-1)", "reversed"],
    ["s", "n"],
)
# s[::-1] -> variable stripped, brackets/dash escaped
check("s[::-1] strips var", re.search(patterns[0], "text[::-1]") is not None)
check("s[::-1] agnostic", re.search(patterns[0], "val[::-1]") is not None)
check("s[::-1] no false pos", re.search(patterns[0], "text[::1]") is None)
# len(s) -> len(\w+)
check("len(s) strips var", re.search(patterns[1], "len(text)") is not None)
check("len(s) agnostic", re.search(patterns[1], "len(x)") is not None)
# [::-1] already agnostic
check("[::-1] still matches", re.search(patterns[2], "text[::-1]") is not None)
# n & (n-1)
check("n & (n-1) strips var", re.search(patterns[3], "num & (num-1)") is not None)
check("n & (n-1) agnostic", re.search(patterns[3], "x & (x-1)") is not None)
# reversed - no var, just re.escape'd
check("reversed unchanged", re.search(patterns[4], "reversed(s)") is not None)
# All patterns compile without error
for p in patterns:
    try:
        re.compile(p)
        ok = True
    except re.error:
        ok = False
    check(f"pattern {p!r} compiles", ok)

# ── 3. label_predicate_spans with regex ──────────────────────────────────────
print("\n3. label_predicate_spans with regex patterns")

# Build a tiny fake MCTS tree manually
class FakeNode:
    def __init__(self, token_str, depth, color=None, children=None, parent=None):
        self.token_str = token_str
        self.depth = depth
        self.color = color
        self.children = children or []
        self.parent = parent
        self._terminal = False

    def is_leaf(self): return len(self.children) == 0
    def is_root(self): return self.parent is None
    @property
    def is_terminal(self): return self._terminal
    def get_path_text(self):
        nodes, n = [], self
        while n and not n.is_root():
            nodes.append(n)
            n = n.parent
        return "".join(node.token_str for node in reversed(nodes))

# Tree: root → " text" → "[::-1]" → " and" → " len" → "(text)" [green leaf]
root = FakeNode("", 0)
n1 = FakeNode(" text", 1, parent=root)
n2 = FakeNode("[::-1]", 2, parent=n1)
n3 = FakeNode(" and ", 3, parent=n2)
n4 = FakeNode(" len", 4, parent=n3)
n5 = FakeNode("(text)", 5, color="green", parent=n4)
root.children = [n1]
n1.children = [n2]; n2.children = [n3]; n3.children = [n4]; n4.children = [n5]

# P+ = [::-1] pattern (palindrome), P- = len( pattern (length check)
# Use the conversion utility so patterns are correct (tests both together)
p_plus = convert_anchor_tokens_to_patterns(["[::-1]"], [])
p_minus = convert_anchor_tokens_to_patterns(["len("], [])

prefix = "return "
labels = label_predicate_spans(root, p_plus, p_minus, prefix=prefix)

leaf_label = labels.get(n5)
check("leaf is labeled", leaf_label is not None)
if leaf_label:
    check("leaf p_plus_start_depth != -1", leaf_label.p_plus_start_depth != -1,
          f"got {leaf_label.p_plus_start_depth}")
    check("leaf p_minus_start_depth != -1", leaf_label.p_minus_start_depth != -1,
          f"got {leaf_label.p_minus_start_depth}")
    check("p_plus before p_minus (palindrome first)",
          leaf_label.p_plus_start_depth <= leaf_label.p_minus_start_depth,
          f"p+={leaf_label.p_plus_start_depth} p-={leaf_label.p_minus_start_depth}")

# ── 4. Dataset load + all patterns compile ────────────────────────────────────
print("\n4. Dataset load + regex validity")

ds = MultiPredicateDataset.load_dataset()
check("loaded 6 groups", len(ds.groups) == 6)

invalid = []
for tp in ds.groups:
    for p in tp.p_plus.anchor_patterns:
        try: re.compile(p)
        except re.error as e: invalid.append((tp.name, "p_plus", p, str(e)))
    for sample in tp.samples:
        if sample.p_minus:
            for p in sample.p_minus.anchor_patterns:
                try: re.compile(p)
                except re.error as e: invalid.append((tp.name, sample.function_name, p, str(e)))

check("all loaded patterns compile", len(invalid) == 0,
      f"{len(invalid)} invalid" if invalid else "")

# Spot-check: is_palindrome p+ patterns contain variable-agnostic forms
pal = ds.get_group("is_palindrome")
# After migration, '[::-1]' became r'\[::-1\]' or similar — verify it still matches
matched = any(re.search(p, "text[::-1]") for p in pal.p_plus.anchor_patterns)
check("is_palindrome p+ patterns match 'text[::-1]'", matched)

matched_n = any(re.search(p, "val[::-1]") for p in pal.p_plus.anchor_patterns)
check("is_palindrome p+ patterns match 'val[::-1]'", matched_n)

# ── 5. JSON roundtrip preserves new fields ────────────────────────────────────
print("\n5. JSON roundtrip")

tp = ds.groups[0]
d = tp.to_dict()
check("p_plus has anchor_patterns key", "anchor_patterns" in d["p_plus"])
check("p_plus has no anchor_tokens key", "anchor_tokens" not in d["p_plus"])
check("p_plus has short_description key", "short_description" in d["p_plus"])
check("samples[0] has augmentation key", "augmentation" in d["samples"][0])

# ── Summary ───────────────────────────────────────────────────────────────────
total = len(results)
passed = sum(results)
print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed" + (" ✓" if passed == total else " ✗"))
if passed < total:
    sys.exit(1)
