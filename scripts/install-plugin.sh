#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLUGDIR="${HOME}/.config/salome/Plugins"
mkdir -p "$PLUGDIR"
PLUGFILE="$PLUGDIR/salome_plugins.py"
LINE='import salome_agent_plugin  # noqa  -- SALOME-agent server launcher'
touch "$PLUGFILE"
grep -qF 'salome_agent_plugin' "$PLUGFILE" || echo "$LINE" >> "$PLUGFILE"
echo "Plugin registered in $PLUGFILE"
echo
echo "Add this to the environment that launches SALOME:"
echo "  export SALOME_PLUGINS_PATH=\"$REPO_ROOT/plugin:$REPO_ROOT\""
