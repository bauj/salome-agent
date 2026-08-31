#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLUGDIR="${HOME}/.config/salome/Plugins"
mkdir -p "$PLUGDIR"
PLUGFILE="$PLUGDIR/salome_plugins.py"
MARKER="# --- salome-agent server launcher ---"
touch "$PLUGFILE"
if ! grep -qF "$MARKER" "$PLUGFILE"; then
  {
    echo "$MARKER"
    echo "import sys as _sys"
    echo "_sys.path[:0] = [r'$REPO_ROOT/plugin', r'$REPO_ROOT']"
    echo "import salome_agent_plugin  # noqa"
    echo "# --- end salome-agent ---"
  } >> "$PLUGFILE"
fi
echo "Plugin registered in $PLUGFILE (repo = $REPO_ROOT)"
echo "No env vars needed — the plugin file bootstraps its own sys.path."
