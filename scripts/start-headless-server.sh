#!/usr/bin/env bash
#
# Start the headless SALOME agent server: a full SALOME session (no GUI) that
# executes stage scripts, geom tool calls and introspection for the CLI. This
# is the unattended path (CI / batch); the GUI plugin is the interactive one.
#
# Requires: SALOME_ROOT pointing at your SALOME install root ($SALOME_ROOT/salome
#           must exist). One server at a time — a second start bails.
#
# Runs in the foreground and stays up until Ctrl-C. Startup takes ~10-20s (the
# SALOME session bootstrap). The server is ready once ~/.salome_agent/server.json
# appears; the CLI discovers it from that file.
#
# Logs: the server redirects its own stdout/stderr to ~/.salome_agent/server.log
# once it is up. Anything printed to this terminal before that is the SALOME
# shell bootstrap.
#
set -euo pipefail
: "${SALOME_ROOT:?set SALOME_ROOT to your SALOME install root}"

handshake="$HOME/.salome_agent/server.json"
log="$HOME/.salome_agent/server.log"

# The agent's own venv python lacks psutil + SALOME's sys.path; strip it so
# `salome shell` picks its bundled python3.
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '\.venv' | paste -sd:)"
unset VIRTUAL_ENV
cd "$(dirname "$0")/.."

echo "starting headless SALOME agent server (SALOME_ROOT=$SALOME_ROOT)"
echo "  ready when   $handshake  appears (~10-20s)"
echo "  server log   $log"
echo "  stop         Ctrl-C"
# --serve-forever: keep accepting CLI runs until Ctrl-C. Without it the
# server exits after the first client disconnects.
exec "$SALOME_ROOT/salome" shell -- python3 -m agent_server --serve-forever "$@"
