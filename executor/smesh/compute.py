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
