#!/usr/bin/env bash
#
# Screenshot a run's result in the SALOME viewer: opens the SALOME GUI,
# rebuilds result.py, shows the result objects and saves the view as a png
# (SalomePyQt.dumpView), then closes the GUI again.
#
# usage: scripts/dump-view.sh <run_id | run dir | result.py> [-o out.png]
#                             [-v top|front|left|iso] [-n name,name,...]
#
# Requires SALOME_ROOT ($SALOME_ROOT/salome must exist) and a display
# (runs with QT_QPA_PLATFORM=xcb). The png defaults to <run dir>/view-<view>.png.
# Only the SALOME processes this script started are stopped at the end, so a
# running headless agent server is left alone.
#
set -euo pipefail
: "${SALOME_ROOT:?set SALOME_ROOT to your SALOME install root}"

usage() { sed -n '7,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[ $# -ge 1 ] || usage
target=$1; shift
out="" view=iso names=""
while [ $# -gt 0 ]; do
  case $1 in
    -o) out=$2; shift 2 ;;
    -v) view=$2; shift 2 ;;
    -n) names=$2; shift 2 ;;
    *) usage ;;
  esac
done

runs="${XDG_DATA_HOME:-$HOME/.local/share}/salome-agent/runs"
if [ -f "$target" ]; then result=$target
elif [ -f "$target/result.py" ]; then result=$target/result.py
else result=$runs/$target/result.py
fi
[ -f "$result" ] || { echo "no result.py for $target" >&2; exit 1; }
result=$(realpath "$result")
out=$(realpath -m "${out:-$(dirname "$result")/view-$view.png}")
log=$(mktemp --suffix=.dump-view.log)
here=$(cd "$(dirname "$0")" && pwd)

salome_pids() {
  ps -eo pid=,args= | grep -iE "salome|omniNames" | grep -v grep | awk '{print $1}' | sort
}
before=$(salome_pids)

# SALOME crashes at startup when the agent's venv python is first on PATH.
path=$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '\.venv' | paste -sd:)
rm -f "$out"
(cd "$SALOME_ROOT" && env -u VIRTUAL_ENV PATH="$path" QT_QPA_PLATFORM=xcb \
  SA_DUMP_RESULT="$result" SA_DUMP_OUT="$out" SA_DUMP_VIEW="$view" \
  SA_DUMP_NAMES="$names" SA_DUMP_LOG="$log" \
  setsid ./salome -g "$here/dump_view.py" > "$log.salome" 2>&1 &)

for _ in $(seq 1 180); do
  grep -q '^DONE' "$log" 2>/dev/null && break
  grep -q SIGSEGV "$log.salome" && break
  sleep 1
done

started=$(comm -13 <(echo "$before") <(salome_pids))
[ -n "$started" ] && kill -TERM $started 2>/dev/null || true
sleep 3
left=$(comm -12 <(echo "$started") <(salome_pids))
[ -n "$left" ] && kill -KILL $left 2>/dev/null || true

cat "$log" 2>/dev/null | grep -v '^DONE' || true
if [ -s "$out" ]; then
  echo "$out"
else
  echo "no image written; SALOME output: $log.salome" >&2
  exit 1
fi
rm -f "$log" "$log.salome"
