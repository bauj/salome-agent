import math

from executor.smesh.state import MAX_ELEMENTS, mesh_of, record, shape_ref, state, tool


def geom_copy(st, obj):
    """A GEOM copy of a SHAPERSTUDY object (geompy refuses sub-shape operations
    on the SHAPERSTUDY object itself); a GEOM object is returned as is. A group's
    own stream does not restore: its sub-shapes are taken from a copy of its
    main shape, which numbers them the same way."""
    kind = type(obj).__name__
    if not kind.startswith("_objref_SHAPER"):
        return obj
    geompy = st.ns["geompy"]
    if kind == "_objref_SHAPER_Group":
        main = geompy.RestoreShape(obj.GetMainShape().GetShapeStream())
        subs = [geompy.GetSubShape(main, [i]) for i in obj.GetSubShapeIndices()]
        return subs[0] if len(subs) == 1 else geompy.MakeCompound(subs)
    return geompy.RestoreShape(obj.GetShapeStream())


def shape_props(executor, name: str) -> dict:
    """Type, area, volume and face count of a published shape or group, measured
    in the module that owns it: a SHAPER shape is never copied to GEOM."""
    expr, owner = shape_ref(executor, name)
    if owner == "shaper":
        from executor.shaper import state as shaper
        if shaper.consumed(executor, name):
            raise LookupError(f"{name} was used up by a later step")
        m = shaper.measure(executor, name)
        return {"type": shaper.shape_type(executor, name), "area": m["area"],
                "volume": m["volume"], "faces": m["faces"]}
    st = state(executor)
    geompy = st.ns["geompy"]
    obj = eval(expr, st.ns)
    p = geompy.BasicProperties(obj)
    return {"type": str(obj.GetShapeType()).lower(), "area": round(p[1], 4),
            "volume": round(p[2], 4), "faces": geompy.NumberOfFaces(obj)}


# criterion -> (SMESH functor, element type it applies to: 3D first, else 2D)
QUALITY = {
    "aspect_ratio": ("FT_AspectRatio3D", "FT_AspectRatio"),
    "minimum_angle": (None, "FT_MinimumAngle"),
    "scaled_jacobian": ("FT_ScaledJacobian", None),
    "skew": (None, "FT_Skew"),
    "warping": ("FT_Warping3D", "FT_Warping"),
    "volume": ("FT_Volume3D", None),
    "area": (None, "FT_Area"),
}
_REPLY_QUALITY = ("aspect_ratio", "minimum_angle", "scaled_jacobian")


def volume_off(mesh_value: float, shape_value: float):
    if not shape_value:
        return None
    return abs(mesh_value - shape_value) / shape_value


def _spaced(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def cap_refusal(n: int, cap: int) -> str:
    return (f"not computed: about {_spaced(n)} elements predicted, above the "
            f"{_spaced(cap)} limit. Use a larger size, or larger local sizes.")


def failure_line(f: dict) -> str:
    c = ", ".join(str(x) for x in f["centre"])
    return (f"{f['algorithm']} failed on a {f['type']} at ({c}), size {f['size']}: "
            f"{f['reason']}")


def _mesh(st, name):
    return st.ns[name]


def counts(st, m) -> dict:
    return {"nodes": m.NbNodes(), "edges": m.NbEdges(), "faces": m.NbFaces(),
            "volumes": m.NbVolumes(), "triangles": m.NbTriangles(),
            "quadrangles": m.NbQuadrangles(), "tetra": m.NbTetras(), "hexa": m.NbHexas(),
            "prism": m.NbPrisms(), "pyramid": m.NbPyramids(),
            "polyhedra": m.NbPolyhedrons()}


_FACE_KINDS = ("Triangle", "Quadrangle", "Polygon")
_VOLUME_KINDS = ("Tetra", "Pyramid", "Hexa", "Penta", "Prism", "Polyhedra")


def evaluated(st, m) -> dict:
    """Element counts per entity type that SMESH's Evaluate() predicts, nodes left out."""
    items = st.ns["SMESH"].EntityType._items
    return {str(items[i]).removeprefix("Entity_"): n
            for i, n in enumerate(m.Evaluate()) if n and str(items[i]) != "Entity_Node"}


def count(counts: dict, dim, rule, bbox) -> "int | None":
    """Elements the mesh will have, or None when nothing can say. Evaluate() counts
    nothing for body fitting and only the end face for Prism: their presets count
    by their own rule. Any other setup that leaves its own dimension uncounted is
    unpredictable."""
    if rule and rule[0] == "grid":
        step = rule[1]
        cells = 1
        for lo, hi in zip(bbox[::2], bbox[1::2]):
            cells *= max(1, math.ceil((hi - lo) / step))
        return cells
    faces = sum(n for k, n in counts.items() if k.endswith(_FACE_KINDS))
    total = sum(counts.values())
    if rule and rule[0] == "layers":
        return total + faces * rule[1]
    volumes = sum(n for k, n in counts.items() if k.endswith(_VOLUME_KINDS))
    if (dim == 3 and not volumes) or (dim == 2 and not faces):
        return None
    return total


def shape_bbox(executor, name: str) -> list:
    expr, owner = shape_ref(executor, name)
    if owner == "shaper":
        from executor.shaper import state as shaper
        return shaper.measure(executor, name)["bbox"]
    st = state(executor)
    return list(st.ns["geompy"].BoundingBox(eval(expr, st.ns)))


def predicted(executor, name: str) -> "int | None":
    st = state(executor)
    info = mesh_of(executor, name)
    rule = info.count_rule
    bbox = shape_bbox(executor, info.shape) if rule and rule[0] == "grid" else None
    return count(evaluated(st, _mesh(st, name)), info.dim, rule, bbox)


UNPREDICTED = "SMESH cannot predict the element count of this setup before meshing"


def failures(st, info, m) -> list:
    geompy = st.ns["geompy"]
    shape = None
    out = []
    for e in m.GetComputeErrors():
        if e.subShapeID <= 0:
            out.append({"algorithm": e.algoName, "reason": e.comment or str(e.code),
                        "type": "shape", "centre": [], "size": 0})
            continue
        # Only GEOM resolves SMESH's sub-shape numbering: copied for a failure only.
        if shape is None:
            shape = geom_copy(st, m.GetShape())
        sub = geompy.GetSubShape(shape, [e.subShapeID])
        t = str(sub.GetShapeType()).lower()
        props = geompy.BasicProperties(sub)
        size = {"edge": props[0], "face": props[1], "solid": props[2]}.get(t, 0.0)
        centre = geompy.PointCoordinates(geompy.MakeCDG(sub))
        out.append({"algorithm": e.algoName, "reason": e.comment or str(e.code), "type": t,
                    "centre": [round(x, 3) + 0.0 for x in centre], "size": round(size, 4)})
    return out


def quality(st, m, criterion: str) -> "dict | None":
    SMESH = st.ns["SMESH"]
    f3, f2 = QUALITY[criterion]
    functor = f3 if (f3 and m.NbVolumes()) else (f2 if m.NbFaces() else None)
    if functor is None:
        return None
    lo, hi = m.GetMinMax(getattr(SMESH, functor))
    return {"min": round(lo, 4), "max": round(hi, 4)}


def measure_mesh(executor, name: str) -> dict:
    st = state(executor)
    info = mesh_of(executor, name)
    m = _mesh(st, name)
    smesh = st.ns["smesh"]
    out = counts(st, m)
    props = shape_props(executor, info.shape)
    out["shape_area"], out["shape_volume"] = props["area"], props["volume"]
    out["volume"] = round(smesh.GetVolume(m), 4) if out["volumes"] else 0.0
    out["area"] = round(smesh.GetArea(m), 4) if out["faces"] else 0.0
    out["quality"] = {k: q for k in _REPLY_QUALITY if (q := quality(st, m, k))}
    out["groups"] = {g: st.ns[f"{name}_{g}"].Size() for g in info.groups}
    out["errors"] = [failure_line(f) for f in failures(st, info, m)]
    return out


def mesh_status(info: dict, dim) -> dict:
    def bad(detail):
        return {"status": "not_meshed", "detail": detail}
    if info["nodes"] == 0:
        return bad("it is not computed: call compute")
    if info["errors"]:
        return bad("it has meshing errors: " + "; ".join(info["errors"]))
    if dim == 3 and info["volumes"] == 0:
        return bad("it has no volume elements")
    if dim == 2 and info["faces"] == 0:
        return bad("it has no face elements")
    empty = [g for g, n in info["groups"].items() if n == 0]
    if empty:
        return bad(", ".join(f"{g} is empty" for g in empty))
    if dim == 3:
        off = volume_off(info["volume"], info["shape_volume"])
        if off is not None and off > 0.01:
            return bad(f"its volume {info['volume']} is {off:.2%} off the shape's "
                       f"{info['shape_volume']}")
    return {"status": "ok"}


_MESH = {"type": "object", "properties": {"mesh": {"type": "string"}}, "required": ["mesh"]}


@tool("evaluate", category="compute",
      description=("The number of elements `mesh` would have, without meshing; null when "
                   "SMESH cannot predict its setup."),
      schema=_MESH)
def tool_evaluate(executor, args: dict):
    mesh_of(executor, args.get("mesh"))
    n = predicted(executor, args["mesh"])
    out = {"ok": True, "records": [], "mesh": args["mesh"], "predicted_elements": n}
    if n is None:
        out["note"] = UNPREDICTED
    return out


@tool("compute", category="compute",
      description=("Build `mesh`. The reply gives the counts per element type, the mesh "
                   "volume or area next to the shape's, the worst quality values, each "
                   "group's size, and every part that failed: algorithm, reason, its type, "
                   "centre and size."),
      schema=_MESH)
def tool_compute(executor, args: dict):
    st = state(executor)
    name = args.get("mesh")
    info = mesh_of(executor, name)
    if info.setup is None:
        raise ValueError(f"{name} has no setup yet: call mesh_tetra, mesh_hexa, ... first")
    m = _mesh(st, name)
    n = predicted(executor, name)
    if n is not None and n > MAX_ELEMENTS:
        return {"ok": False, "error": cap_refusal(n, MAX_ELEMENTS)}
    done = bool(m.Compute())
    out = {"ok": True, "records": [record("Compute", f"{name}.Compute()", recv=name)],
           "mesh": name, "computed": done, **measure_mesh(executor, name)}
    if n is None:
        out["note"] = UNPREDICTED + f": computed without the {_spaced(MAX_ELEMENTS)} element limit"
    return out


@tool("mesh_info", category="query",
      description="Counts per element type of `mesh` and of each of its groups.",
      schema=_MESH)
def tool_mesh_info(executor, args: dict):
    return {"ok": True, "records": [], "mesh": args.get("mesh"),
            **measure_mesh(executor, args.get("mesh"))}


@tool("check_quality", category="query",
      description=("The min and max of `criterion` (aspect_ratio, minimum_angle, "
                   "scaled_jacobian, skew, warping, volume, area) over the elements of "
                   "`mesh`, and how many pass `threshold`."),
      schema={"type": "object", "properties": {
          "mesh": {"type": "string"},
          "criterion": {"type": "string", "enum": list(QUALITY)},
          "threshold": {"type": "number"}}, "required": ["mesh", "criterion"]})
def tool_check_quality(executor, args: dict):
    st = state(executor)
    name, criterion = args.get("mesh"), args.get("criterion")
    mesh_of(executor, name)
    if criterion not in QUALITY:
        raise ValueError(f"criterion must be one of {', '.join(QUALITY)}")
    m = _mesh(st, name)
    q = quality(st, m, criterion)
    if q is None:
        raise ValueError(f"{criterion} does not apply to the elements of {name}")
    out = {"ok": True, "records": [], "mesh": name, "criterion": criterion, **q}
    t = args.get("threshold")
    if t is not None:
        SMESH, smesh = st.ns["SMESH"], st.ns["smesh"]
        f3, f2 = QUALITY[criterion]
        functor, kind = ((f3, SMESH.VOLUME) if f3 and m.NbVolumes() else (f2, SMESH.FACE))
        flt = smesh.GetFilter(kind, getattr(SMESH, functor), SMESH.FT_MoreThan, t)
        out["above_threshold"] = len(m.GetIdsFromFilter(flt))
    return out
