#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLUGDIR="${HOME}/.config/salome/Plugins"
mkdir -p "$PLUGDIR"
PLUGFILE="$PLUGDIR/salome_plugins.py"
MARKER="# --- salome-agent server launcher ---"
touch "$PLUGFILE"
if grep -qF "$MARKER" "$PLUGFILE"; then
  echo "Already registered in $PLUGFILE — remove the salome-agent block to reinstall."
  exit 0
fi
# Registered inline (an `import salome_agent_plugin` indirection here does not
# reliably reach the plugin manager's current_plugins_manager; a direct
# AddFunction call in salome_plugins.py does).
cat >> "$PLUGFILE" <<EOF

$MARKER
import sys as _sys
_sys.path[:0] = [r'$REPO_ROOT/plugin', r'$REPO_ROOT']
try:
    import salome_pluginsmanager as _spm
    from agent_server.gui_plugin import launch as _launch
    _spm.AddFunction('Launch SALOME agent server',
                     'Start (or restart) the SALOME-agent server in this GUI process',
                     _launch)
except Exception:
    import traceback as _tb
    _tb.print_exc()
# --- end salome-agent ---
EOF
echo "Registered in $PLUGFILE (repo = $REPO_ROOT)"
echo "No env vars needed. Restart the SALOME GUI."
