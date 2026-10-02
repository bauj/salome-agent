# Runs INSIDE the SALOME GUI (launched by scripts/dump-view.sh): rebuilds a
# run's result.py, displays its result objects in an OCC view and saves a
# screenshot with SalomePyQt.dumpView. Driven by environment variables:
#   SA_DUMP_RESULT  path to result.py
#   SA_DUMP_OUT     png to write
#   SA_DUMP_VIEW    top | front | left | iso (default iso)
#   SA_DUMP_NAMES   comma-separated variables to show (default: the run's
#                   result_names from build_graph.json, else the last published)
#   SA_DUMP_LOG     log file; ends with a DONE line when finished
import json
import os
import re
import traceback

from PyQt5.QtCore import QTimer

RESULT = os.environ["SA_DUMP_RESULT"]
OUT = os.environ["SA_DUMP_OUT"]
VIEW = os.environ.get("SA_DUMP_VIEW", "iso")
LOG = os.environ["SA_DUMP_LOG"]

log = open(LOG, "w")


def _finish():
    log.write("DONE\n")
    log.close()


def _names(src: str) -> list[str]:
    if os.environ.get("SA_DUMP_NAMES"):
        return [n.strip() for n in os.environ["SA_DUMP_NAMES"].split(",") if n.strip()]
    graph = os.path.join(os.path.dirname(RESULT), "build_graph.json")
    if os.path.exists(graph):
        with open(graph) as f:
            names = [n for b in json.load(f).get("blocks", []) for n in b.get("result_names") or []]
        if names:
            return names
    return re.findall(r"addToStudy\((\w+),", src)[-1:]


def _show():
    # A -g script runs before the desktop has any view, so this is deferred.
    try:
        try:
            from salome.kernel import salome
        except ImportError:
            import salome
        salome.salome_init()
        import SalomePyQt
        sg = SalomePyQt.SalomePyQt()
        sg.activateModule("Geometry")
        sg.activateView(sg.createView("OCCViewer"))
        gui = salome.ImportComponentGUI("GEOM")
        with open(RESULT) as f:
            src = f.read()
        ns = {}
        exec(src, ns)
        geompy = ns["geompy"]
        for name in _names(src):
            if name not in ns:
                log.write(f"not in result.py: {name}\n")
                continue
            entry = geompy.addToStudy(ns[name], name)
            gui.createAndDisplayGO(entry)
            # shading; objects keep their own colour (set_color) or GEOM's default
            gui.setDisplayMode(entry, 1)
            log.write(f"shown: {name}\n")
        salome.sg.updateObjBrowser()
        {"top": salome.sg.ViewTop, "front": salome.sg.ViewFront,
         "left": salome.sg.ViewLeft}.get(VIEW, lambda: None)()
        salome.sg.FitAll()
        QTimer.singleShot(1500, _dump)
    except Exception:
        log.write(traceback.format_exc())
        _finish()


def _dump():
    try:
        import SalomePyQt
        log.write(f"dump={SalomePyQt.SalomePyQt().dumpView(OUT)}\n")
    except Exception:
        log.write(traceback.format_exc())
    _finish()


QTimer.singleShot(5000, _show)
