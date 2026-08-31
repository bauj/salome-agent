#!/usr/bin/env bash
set -euo pipefail
: "${SALOME_ROOT:?set SALOME_ROOT to your SALOME install root}"
# The agent's own venv python lacks psutil + SALOME's sys.path; strip it so
# `salome shell` picks its bundled python3.
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '\.venv' | paste -sd:)"
unset VIRTUAL_ENV
cd "$(dirname "$0")/.."
exec "$SALOME_ROOT/salome" shell -- python3 -m agent_server "$@"
