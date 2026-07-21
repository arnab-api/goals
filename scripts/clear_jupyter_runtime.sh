#!/usr/bin/env bash
# clear_jupyter_runtime.sh — clean up wedged Jupyter kernels + stale connection files.
#
# Why: VS Code's Jupyter extension intermittently wedges the kernel it starts for a
# *never-before-connected* notebook (shell half-answers, heartbeat never comes up), so the
# notebook "connects but nothing runs". The env + kernelspec are fine — a CLI-started kernel is
# healthy in <1s. The fix is to clear the wedged kernel(s) and the stale connection files that
# pile up in the tmpfs runtime dir, then reopen the notebook. See memory: vscode-kernel-wedge.
#
# Usage:
#   scripts/clear_jupyter_runtime.sh            # dry-run: report only
#   scripts/clear_jupyter_runtime.sh --kill     # also SIGKILL *idle* ipykernel procs (no torch)
#   scripts/clear_jupyter_runtime.sh --kill --force-all   # kill ALL your ipykernels (careful)
#
# Safe by default: never touches connection files still held by a live process, and only kills
# kernels whose RSS is small (idle — model not yet imported) unless --force-all is given.
set -euo pipefail

KILL=0; FORCE_ALL=0
for a in "$@"; do
  case "$a" in
    --kill) KILL=1 ;;
    --force-all) FORCE_ALL=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

UID_=$(id -u)
RT="/run/user/${UID_}/jupyter/runtime"
IDLE_RSS_KB=$((300 * 1024))   # <300 MB => kernel hasn't imported a big model, treat as idle

echo "== runtime dir: ${RT}"

# ── 1. running ipykernel processes ────────────────────────────────────────────
# Collect connection files still in use by a live kernel (from each proc's cmdline `--f=<file>`),
# so we never delete a running kernel's file. fuser is unreliable here: the .json is read once at
# launch and not held open, so a live kernel's file would otherwise look "stale".
declare -A LIVE_CF=()
mapfile -t PIDS < <(pgrep -u "${UID_}" -f ipykernel_launcher || true)
if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "   no ipykernel processes running"
else
  echo "   ipykernel processes:"
  for pid in "${PIDS[@]}"; do
    [ -r "/proc/${pid}/cmdline" ] || continue
    cf=$(tr '\0' '\n' < "/proc/${pid}/cmdline" | sed -n 's/^--f=//p' | head -1)
    rss=$(awk '/VmRSS/{print $2}' "/proc/${pid}/status" 2>/dev/null || echo 0)
    et=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ' || echo '?')
    tag="idle"; [ "$rss" -ge "$IDLE_RSS_KB" ] && tag="BUSY(model-loaded)"
    printf "     pid=%-8s rss=%sMB elapsed=%s  %s\n" "$pid" "$((rss/1024))" "$et" "$tag"
    killed=0
    if [ "$KILL" -eq 1 ]; then
      if [ "$FORCE_ALL" -eq 1 ] || [ "$rss" -lt "$IDLE_RSS_KB" ]; then
        kill -9 "$pid" 2>/dev/null && { echo "       -> killed"; killed=1; }
      else
        echo "       -> kept (busy; use --force-all to kill anyway)"
      fi
    fi
    # a still-running kernel's connection file must be preserved
    [ "$killed" -eq 0 ] && [ -n "$cf" ] && LIVE_CF["$cf"]=1
  done
fi

# ── 2. stale connection files (not held by any live process) ──────────────────
if [ ! -d "$RT" ]; then
  echo "   (no runtime dir; nothing to clean)"; exit 0
fi
shopt -s nullglob
files=("$RT"/kernel-*.json)
total=${#files[@]}
removed=0; kept=0
for f in "${files[@]}"; do
  if [ -n "${LIVE_CF[$f]:-}" ]; then
    kept=$((kept+1))
  else
    rm -f "$f" && removed=$((removed+1))
  fi
done
echo "   connection files: ${total} total -> removed ${removed} stale, kept ${kept} live"
echo "== done. Reopen the notebook in VS Code and select the kernel."
