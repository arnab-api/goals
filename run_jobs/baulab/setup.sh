#!/usr/bin/env bash
# Minimal setup for the baulab job dispatcher (run_jobs/baulab).
#
# Installs the CLI tools the dispatcher shells out to — `sshpass` (feed the ssh password
# non-interactively) and `screen` (detached per-job sessions) — into the project conda env,
# and seeds a local MACHINES.py from the template if you don't have one yet.
#
# Usage:
#   bash run_jobs/baulab/setup.sh            # uses the `goals` env
#   ENV_NAME=myenv bash run_jobs/baulab/setup.sh
#
# Requirements:
#   - conda (the same one that owns the project env)
#   - the python deps (pyyaml, rich) — already in the `goals` env via the root setup_env.sh
#
# Note: `screen` must also exist on every *remote* worker; `sshpass` is only needed on the
# box you launch the scheduler from (this one).

set -euo pipefail

ENV_NAME="${ENV_NAME:-goals}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Ensuring CLI tools (sshpass, screen) in conda env '${ENV_NAME}'"
need=()
conda run -n "${ENV_NAME}" bash -lc 'command -v sshpass >/dev/null' || need+=(sshpass)
conda run -n "${ENV_NAME}" bash -lc 'command -v screen  >/dev/null' || need+=(screen)
if [ "${#need[@]}" -gt 0 ]; then
    echo "    installing: ${need[*]}"
    conda install -n "${ENV_NAME}" -y -c conda-forge "${need[@]}"
else
    echo "    already present — nothing to install"
fi

echo "==> Seeding MACHINES.py (per-client machine inventory; gitignored)"
if [ -f "${HERE}/MACHINES.py" ]; then
    echo "    ${HERE}/MACHINES.py already exists — leaving it untouched"
else
    cp "${HERE}/MACHINES.example.py" "${HERE}/MACHINES.py"
    echo "    created ${HERE}/MACHINES.py from MACHINES.example.py — EDIT IT to list your machines"
fi

echo ""
echo "Done. Next:"
echo "  1. Edit run_jobs/baulab/MACHINES.py  (MACHINES + SAFE_MACHINES; use name=\"local\" for this box)"
echo "  2. Make sure env.yml has BAULAB_USER / BAULAB_PASS"
echo "  3. Plan a batch:  conda run -n ${ENV_NAME} python -m run_jobs.baulab.dispatch --jobs <jobs.json> --dry-run"
