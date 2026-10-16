# Runs INSIDE SALOME (`salome shell -- python3 scripts/cadsmith_export.py ...`),
# in its own process, not in the agent server: rebuilds a run's result.py and
# exports the run's final geometry to STL.
#
# usage: cadsmith_export.py <result.py> <out.stl> <out.json>
#
# The final geometry is the result_names of the last geom or shaper block in
# the run's build_graph.json; several names are exported as one compound.
# out.json gets the names used, the exact volume, face and solid counts and
# check_shape, or the error that stopped the export.
import json
import os
import sys
import traceback


def _final_names(run_dir: str) -> list[str]:
    with open(os.path.join(run_dir, "build_graph.json")) as f:
        blocks = json.load(f).get("blocks", [])
    for b in reversed(blocks):
        if b.get("module") in ("geom", "shaper") and b.get("result_names"):
            return list(b["result_names"])
    return []


def _export_geom(ns: dict, names: list, out_stl: str, info: dict) -> None:
    geompy = ns["geompy"]
    shapes = [ns[n] for n in names]
    shape = shapes[0] if len(shapes) == 1 else geompy.MakeCompound(shapes)
    _length, _area, info["volume"] = geompy.BasicProperties(shape)
    info["faces"] = geompy.NumberOfFaces(shape)
    info["solids"] = geompy.NumberOfSolids(shape)
    info["check_shape"] = bool(geompy.CheckShape(shape))
    geompy.ExportSTL(shape, out_stl, False, 0.001, True)


def _export_shaper(ns: dict, names: list, out_stl: str, info: dict) -> None:
    import GeomAPI
    from GeomAlgoAPI import GeomAlgoAPI_ShapeTools as T
    model, doc = ns["model"], ns["shaper_doc"]
    model.end()
    model.begin()
    results = [r for n in names for r in ns[n].feature().results() if r.shape() is not None]
    if not results:
        raise LookupError(f"no SHAPER result shape for {', '.join(names)}")
    shapes = [r.shape() for r in results]
    info["volume"] = sum(T.volume(s) for s in shapes)
    info["faces"] = sum(len(s.subShapes(GeomAPI.GeomAPI_Shape.FACE)) for s in shapes)
    info["solids"] = sum(len(s.subShapes(GeomAPI.GeomAPI_Shape.SOLID)) for s in shapes)
    info["check_shape"] = all(T.isShapeValid(s) for s in shapes)
    sels = [model.selection(r.shape().shapeTypeStr(), r.data().name()) for r in results]
    if len(sels) == 1:
        sel = sels[0]
    else:
        compound = model.addCompound(doc, sels)
        model.do()
        r = compound.feature().firstResult()
        sel = model.selection(r.shape().shapeTypeStr(), r.data().name())
    model.exportToSTL(doc, out_stl, sel, 0.001, 0.5, True, False)
    model.end()
    if not os.path.exists(out_stl):
        raise RuntimeError("SHAPER's STL export wrote no file")


def main(result_py: str, out_stl: str, out_json: str) -> int:
    info: dict = {"result_py": result_py}
    try:
        try:
            from salome.kernel import KernelBasis
        except ImportError:
            import KernelBasis
        KernelBasis.setSSLMode(True)
        names = _final_names(os.path.dirname(result_py))
        info["names"] = names
        if not names:
            raise LookupError("no geom or shaper block with result names in build_graph.json")
        with open(result_py) as f:
            src = f.read()
        ns: dict = {"__name__": "__main__"}
        exec(compile(src, result_py, "exec"), ns)
        missing = [n for n in names if n not in ns]
        if missing:
            raise LookupError(f"not in result.py: {', '.join(missing)}")
        if "from salome.shaper import model" in src:
            _export_shaper(ns, names, out_stl, info)
        else:
            _export_geom(ns, names, out_stl, info)
        info["ok"] = True
    except Exception:
        info["ok"] = False
        info["error"] = traceback.format_exc()
    with open(out_json, "w") as f:
        json.dump(info, f, indent=1)
    return 0 if info["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
