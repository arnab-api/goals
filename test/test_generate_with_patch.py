#!/usr/bin/env python
"""
Test script for generate_with_patch: all combinations of strategy × on_pass.

Tests:
1. Baseline (no patch) runs without error
2. strategy="replace", on_pass="first"  — zeroes last-token activation on prefill only
3. strategy="replace", on_pass="all"    — zeroes last-token activation at every step
4. strategy="add",     on_pass="first"  — adds large shift on prefill only
5. strategy="add",     on_pass="all"    — adds large shift at every step
6. ValueError raised when on_pass="all" with an interior token index (not 0 or -1)
7. baukit interface produces the same output as nnsight for the same patch

Expected:
- All four patched outputs differ from the baseline (patches observable).
- nnsight and baukit agree on the same patch.
- The ValueError guard fires correctly.

Usage:
    conda run -n goals python test/test_generate_with_patch.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from rich.console import Console
from rich.panel import Panel

from fixtures import get_test_model

console = Console(highlight=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    console.print(f"\n[bold cyan]{'─' * 60}[/bold cyan]")
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print(f"[bold cyan]{'─' * 60}[/bold cyan]\n")


def ok(msg: str) -> None:
    console.print(f"  [bold green]✓[/bold green]  {msg}")


def fail(msg: str) -> None:
    console.print(f"  [bold red]✗[/bold red]  {msg}")


# ── Main test ─────────────────────────────────────────────────────────────────

def run_tests() -> int:
    """Run all tests. Returns the number of failures."""
    from src.functional import PatchSpec, generate_with_patch

    # ── Setup ─────────────────────────────────────────────────────────────────
    section("Setup")
    mt = get_test_model()
    console.print(f"  Model : [cyan]{mt.name}[/cyan]")
    console.print(f"  Layers: {mt.n_layer}   Hidden: {mt.n_embd}")

    prompt  = "def is_odd(n: int) -> bool:\n    return "
    layer   = mt.layer_name_format.format(mt.n_layer // 2)
    hidden  = mt.n_embd
    device  = mt.device

    zero  = torch.zeros(hidden, dtype=torch.bfloat16, device=device)
    large = torch.full((hidden,), 30.0, dtype=torch.bfloat16, device=device)

    gen_kwargs = dict(max_new_tokens=12, do_sample=False, remove_prefix=True)

    failures = 0

    # ── Test 1: baseline ──────────────────────────────────────────────────────
    section("Test 1 — baseline (no patch)")
    try:
        baseline = generate_with_patch(mt, prompt, patches=None, **gen_kwargs)
        ok(f"baseline output: {baseline[0]!r}")
    except Exception as e:
        fail(f"baseline raised: {e}")
        failures += 1
        baseline = [""]

    # ── Test 2: replace + first ───────────────────────────────────────────────
    section("Test 2 — replace + first")
    try:
        out_rf = generate_with_patch(mt, prompt, patches=[
            PatchSpec(location=(layer, -1), patch=zero, strategy="replace", on_pass="first"),
        ], **gen_kwargs)
        console.print(f"  output : {out_rf[0]!r}")
        console.print(f"  baseline: {baseline[0]!r}")
        if out_rf[0] != baseline[0]:
            ok("output differs from baseline (patch observable)")
        else:
            fail("output identical to baseline — patch may not have fired")
            failures += 1
    except Exception as e:
        fail(f"raised: {e}")
        failures += 1
        out_rf = [""]

    # ── Test 3: replace + all ─────────────────────────────────────────────────
    section("Test 3 — replace + all  (index must be -1 or 0)")
    try:
        out_ra = generate_with_patch(mt, prompt, patches=[
            PatchSpec(location=(layer, -1), patch=zero, strategy="replace", on_pass="all"),
        ], **gen_kwargs)
        console.print(f"  output  : {out_ra[0]!r}")
        console.print(f"  baseline: {baseline[0]!r}")
        if out_ra[0] != baseline[0]:
            ok("output differs from baseline (patch observable)")
        else:
            fail("output identical to baseline — patch may not have fired")
            failures += 1
    except Exception as e:
        fail(f"raised: {e}")
        failures += 1
        out_ra = [""]

    # ── Test 4: add + first ───────────────────────────────────────────────────
    section("Test 4 — add + first")
    try:
        out_af = generate_with_patch(mt, prompt, patches=[
            PatchSpec(location=(layer, -1), patch=large, strategy="add", on_pass="first"),
        ], **gen_kwargs)
        console.print(f"  output  : {out_af[0]!r}")
        console.print(f"  baseline: {baseline[0]!r}")
        if out_af[0] != baseline[0]:
            ok("output differs from baseline (patch observable)")
        else:
            fail("output identical to baseline — patch may not have fired")
            failures += 1
    except Exception as e:
        fail(f"raised: {e}")
        failures += 1
        out_af = [""]

    # ── Test 5: add + all ─────────────────────────────────────────────────────
    section("Test 5 — add + all  (index must be -1 or 0)")
    try:
        out_aa = generate_with_patch(mt, prompt, patches=[
            PatchSpec(location=(layer, -1), patch=large, strategy="add", on_pass="all"),
        ], **gen_kwargs)
        console.print(f"  output  : {out_aa[0]!r}")
        console.print(f"  baseline: {baseline[0]!r}")
        if out_aa[0] != baseline[0]:
            ok("output differs from baseline (patch observable)")
        else:
            fail("output identical to baseline — patch may not have fired")
            failures += 1
    except Exception as e:
        fail(f"raised: {e}")
        failures += 1
        out_aa = [""]

    # ── Test 6: ValueError guard ───────────────────────────────────────────────
    section("Test 6 — ValueError on on_pass='all' with interior index")
    try:
        generate_with_patch(mt, prompt, patches=[
            PatchSpec(location=(layer, 5), patch=zero, strategy="replace", on_pass="all"),
        ], **gen_kwargs)
        fail("no ValueError raised — guard is missing")
        failures += 1
    except ValueError as e:
        ok(f"correctly raised ValueError: {e}")
    except Exception as e:
        fail(f"raised wrong exception type {type(e).__name__}: {e}")
        failures += 1

    # ── Test 7: nnsight vs baukit agreement ───────────────────────────────────
    section("Test 7 — nnsight and baukit agree on the same patch")
    patch = PatchSpec(location=(layer, -1), patch=zero, strategy="replace", on_pass="first")
    try:
        out_nnsight = generate_with_patch(
            mt, prompt, patches=[patch], patch_interface="nnsight", **gen_kwargs
        )
        out_baukit = generate_with_patch(
            mt, prompt, patches=[patch], patch_interface="baukit", **gen_kwargs
        )
        console.print(f"  nnsight: {out_nnsight[0]!r}")
        console.print(f"  baukit : {out_baukit[0]!r}")
        if out_nnsight[0] == out_baukit[0]:
            ok("nnsight and baukit outputs match")
        else:
            fail("nnsight and baukit outputs differ — interface mismatch")
            failures += 1
    except Exception as e:
        fail(f"raised: {e}")
        failures += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Summary")
    if failures == 0:
        console.print(Panel("[bold green]All tests passed.[/bold green]", expand=False))
    else:
        console.print(
            Panel(f"[bold red]{failures} test(s) failed.[/bold red]", expand=False)
        )

    return failures


if __name__ == "__main__":
    n_failures = run_tests()
    sys.exit(0 if n_failures == 0 else 1)
