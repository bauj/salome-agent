"""The run's meshes, the named geometry groups they use, and what every SMESH
tool shares: resolving names, running SMESH code, recording it.

Every change is made by executing the same source line that goes into
result.py, so the live study and the generated script cannot drift."""
from executor.geom.helpers import _next_name
from executor.registry import ToolRegistry
from executor.shaper.state import record  # noqa: F401 — the record format is shared

registry = ToolRegistry(valid_categories={
    "setup", "naming", "local", "group", "compute", "edit", "export", "query"})
tool = registry.tool

MAX_ELEMENTS = 1_000_000
COMPUTE_TIMEOUT_S = 600

_INIT = '''try:
    from salome.kernel import SMESH, StdMeshers
except ImportError:
    import SMESH, StdMeshers
'''

_HELPERS = '''def _shaper_shape(name):
    sos = salome.myStudy.FindObjectByName(name, "SHAPERSTUDY")
    if not sos:
        raise LookupError(f"no SHAPER result {name!r} in the study")
    return sos[-1].GetObject()


def _ids(group):
    return list(group.GetSubShapeIndices())
'''

PREAMBLE = ("from salome.smesh import smeshBuilder\nsmesh = smeshBuilder.New()\n"
            + _INIT + _HELPERS)

# A SHAPER group made inside a SMESH stage: SMESH only sees it once published.
REPUBLISH = "model.end()\nmodel.publishToShaperStudy()\nmodel.begin()"


class GeoGroup:
    def __init__(self, shape: str, owner: str, sub_type: str):
        self.shape, self.owner, self.sub_type = shape, owner, sub_type


class MeshInfo:
    def __init__(self, shape: str, owner: str):
        self.shape, self.owner = shape, owner
        self.dim: int | None = None
        self.setup: str | None = None
        # var of the NETGEN parameters when a NETGEN preset meshes it
        self.netgen: str | None = None
        # (dimension, group or None) -> algorithm var
        self.algos: dict = {}
        # group -> algorithm var of its local setup (sub-mesh)
        self.submeshes: dict = {}
        # mesh group name -> geometry group name
        self.groups: dict = {}
        # group -> local size set on the NETGEN parameters (local_size)
        self.local_sizes: dict = {}


class SmeshState:
    def __init__(self, ns: dict):
        self.ns = ns
        self.meshes: dict[str, MeshInfo] = {}
        self.geo_groups: dict[str, GeoGroup] = {}


def state(executor) -> SmeshState:
    st = executor.__dict__.get("smesh_state")
    if st is None:
        from salome.smesh import smeshBuilder
        ns = {"salome": executor.namespace["salome"], "smesh": executor.namespace["smesh"],
              "smeshBuilder": smeshBuilder, "geompy": executor.namespace["geompy"]}
        exec(_INIT + _HELPERS, ns)
        st = executor.__dict__["smesh_state"] = SmeshState(ns)
    return st


def _shaper(executor):
    return executor.__dict__.get("shaper_state")


def shape_ref(executor, name) -> tuple:
    st = state(executor)
    if not isinstance(name, str):
        raise ValueError(f"shape must be one published name, got {name!r}")
    if name in st.geo_groups:
        return group_ref(executor, name)[0], st.geo_groups[name].owner
    sh = _shaper(executor)
    if sh is not None and name in sh.objects:
        return f'_shaper_shape("{name}")', "shaper"
    if name in executor.geom_bindings:
        st.ns[name] = executor.geom_bindings[name]
        return name, "geom"
    known = sorted([*((sh.objects if sh else {})), *executor.geom_bindings, *st.geo_groups])
    raise ValueError(f"no published shape or group named {name!r}; published: "
                     f"{', '.join(known) or 'none'}")


def group_ref(executor, name) -> tuple:
    st = state(executor)
    g = st.geo_groups.get(name)
    if g is None:
        sh = _shaper(executor)
        if sh is not None and name in getattr(sh, "groups", {}):
            g = st.geo_groups[name] = GeoGroup(sh.groups[name], "shaper", "")
        else:
            raise ValueError(f"no named group {name!r}; name faces or edges with "
                             "name_subshapes first")
    return (f'_shaper_shape("{name}")' if g.owner == "shaper" else name), g


def mesh_of(executor, name) -> MeshInfo:
    m = state(executor).meshes.get(name)
    if m is None:
        raise ValueError(f"no mesh named {name!r}; make one with create_mesh")
    return m


def run(st: SmeshState, code: str) -> None:
    exec(code, st.ns)


def new_name(executor, prefix: str) -> str:
    return _next_name(executor, prefix)


def reset(executor) -> None:
    executor.__dict__.pop("smesh_state", None)
