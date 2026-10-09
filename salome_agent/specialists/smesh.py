from executor.smesh.state import MAX_ELEMENTS, PREAMBLE
from salome_agent.calc import CALC_PROMPT
from salome_agent.logs import log
from salome_agent.modules import ModuleSpec, Terminator, register
from salome_agent.specialists.geom import FINISH_STAGE_DESCRIPTION, FINISH_STAGE_SCHEMA
from salome_agent.specialists.shaper import _finish, _render

SMESH_SYSTEM_PROMPT = """You mesh geometry in SMESH by calling tools. Lengths are millimetres,
unconverted. Reference shapes, groups and meshes by the name they were
published under; the shapes built by earlier stages are listed in your first
message.

A mesh belongs to one shape. create_mesh(shape) makes it, empty; then one call
sets how the whole mesh is built:
- mesh_tetra: tetrahedra filling the solids, for solids of any form.
- mesh_surface: triangles, or mostly quadrangles, on the faces only; no volume
  elements.
- mesh_hexa: structured hexahedra. Only meshes block-like solids: six sides,
  every face four-sided.
- mesh_body_fitted: hexahedra on a Cartesian grid, cut where the grid meets the
  boundary.
- mesh_prism: hexahedra and prisms swept through an extruded solid, from one
  end face to the other.
- set_algorithm and set_hypothesis: any other combination, dimension by
  dimension.
Call one of these per mesh; a second one replaces the first.

Local control works on named parts of the shape. get_subshapes lists the
faces, edges or solids of a shape with their ids, centres and sizes;
name_subshapes publishes some of them as a named group (e.g. "inlet"). The
group names go to:
- local_size: a finer or coarser element size on those parts.
- edge_segments: a set number of segments, or a length, along edges.
- viscous_layers: thin layers of elements along walls.
- group_on_geometry: a group of mesh elements -- a boundary face, a fixed edge,
  a material -- for the solver that reads the mesh.
groups_from_geometry makes one mesh group for every named group of the shape.

compute builds the mesh. Its reply gives the element count per type, the mesh
volume (or area) next to the shape's, the worst quality values, and every part
of the shape that failed to mesh: the algorithm, the reason, and that part's
type, centre and size. A failure says what to change -- a face that is not
four-sided under mesh_hexa, a size too large for a thin wall. Change the setup
and compute again; only what the change touched is re-meshed. evaluate
predicts the counts without meshing; compute refuses a mesh predicted above
{max_elements} elements. Some hand-built setups cannot be predicted: their reply
says so, and compute builds them without that limit.

When done, call finish_stage(result_names) with the meshes that matter. It is
refused while a named mesh is not computed, has meshing errors, has no
elements of the dimension it was set up for, has an empty group, or fills a
solid with a volume more than 1% off the solid's. export_mesh writes a file:
only when the request asks for one. If you abandon an attempt, call discard on
what you built for it.

Tool calls are cheap: each answers in a few seconds, anything you build can be
removed, and nothing is final until finish_stage. As soon as you have enough
for one call, make it; when you are unsure how a setup will mesh, compute it
and read the reply instead of working it out in your reasoning.

The rhythm, on a different part (plate_1, a plate 100 x 50 x 5 with one hole
of diameter 10 through it; "tetrahedral mesh, size 5, 1 mm at the hole, a group
for the hole wall called bolt"):
1. "Set up the mesh and find the hole wall in the same turn."
   create_mesh("plate_1") -> mesh_1
   mesh_tetra("mesh_1", max_size=5) -> ok
   get_subshapes("plate_1", "face", where={{"by_size": "smallest", "count": 1}})
   -> F7, centre (50, 25, 2.5), area 157.08
2. "F7 is the hole wall (pi x 10 x 5). Name it, refine it, group it."
   name_subshapes("plate_1", ["F7"], "bolt") -> bolt
   local_size("mesh_1", ["bolt"], 1) -> ok
   group_on_geometry("mesh_1", "bolt") -> bolt
3. compute("mesh_1") -> computed, ~16 000 tetrahedra, volume 24607.3 of
   24607.3, group bolt ~700 faces, no errors
4. finish_stage(["mesh_1"])

Do only what the request asks. Each turn: state your reasoning in at most 2-3
sentences, then call one or more tools. Batch INDEPENDENT calls in one turn;
a call that needs a name another call is about to create waits for the next
turn.
{catalogue}"""

PLANNER_TOOL_LINES = {
    "create_mesh": "empty mesh of a shape or group",
    "mesh_tetra": "tetrahedra filling the solids (NETGEN), with max/min size and fineness",
    "mesh_surface": "triangles or quadrangles on the faces only",
    "mesh_hexa": "structured hexahedra on block-like solids",
    "mesh_body_fitted": "hexahedra on a Cartesian grid",
    "mesh_prism": "prisms swept through an extruded solid",
    "set_algorithm": "any algorithm for one dimension, on the mesh or on named groups",
    "set_hypothesis": "any sizing rule, on the mesh or on named groups",
    "get_subshapes": "ids of a shape's edges, faces, vertices or solids, filtered by position, size or plane",
    "name_subshapes": "named group of some faces, edges or solids of a shape",
    "shape_info": "type, volume, area, face count of shapes and groups",
    "local_size": "element size on named groups",
    "edge_segments": "segment count or length along named edges",
    "viscous_layers": "thin layers along walls",
    "set_submesh_order": "which local setup wins where two meet",
    "group_on_geometry": "mesh group on a named group, for a solver",
    "groups_from_geometry": "mesh groups for every named group of the shape",
    "evaluate": "predicted element count without meshing",
    "compute": "build the mesh; counts, volume, quality, failed parts",
    "mesh_info": "counts of a mesh and its groups",
    "check_quality": "min/max of a quality criterion, count past a threshold",
    "clear_mesh": "remove the elements, keep the setup",
    "remove_hypothesis": "remove a sizing rule",
    "convert_to_quadratic": "add mid-edge nodes",
    "export_mesh": "write the mesh to a med, unv, stl or cgns file",
}
CORE_TOOLS = frozenset({"create_mesh", "mesh_tetra", "mesh_surface", "mesh_hexa",
                        "get_subshapes", "name_subshapes", "local_size",
                        "group_on_geometry", "compute", "mesh_info"})


def _tool_lines() -> str:
    return "\n".join(f"  {n} -- {d}" for n, d in PLANNER_TOOL_LINES.items())


_CATALOGUE = ("- smesh (SMESH): meshing a shape an earlier stage built -- tetrahedra,\n"
              "  hexahedra, surface or other meshes, local sizes on chosen faces or edges,\n"
              "  viscous layers, mesh groups for a solver's boundary conditions, quality\n"
              "  checks, export to a file. A SMESH stage names the faces and edges it needs\n"
              "  itself: the CAD stage does not prepare them.")


def _smesh_prompt(verify_cadence: str = "final") -> str:
    tools = ("You start with a subset of the SMESH tools. All of them are listed below; "
             "call load_tools with the names of any you need that are not in your list "
             "yet.\n" + _tool_lines())
    return (SMESH_SYSTEM_PROMPT.format(max_elements=f"{MAX_ELEMENTS:,}".replace(",", " "),
                                       catalogue=tools)
            + "\n\n" + CALC_PROMPT)


def _upstream(client, names: list[str]) -> "str | None":
    if not names:
        return None
    info = {}
    try:
        r = client.call_tool("shape_info", {"names": list(names)}, module="smesh")
        if r.get("ok"):
            info = r.get("shapes") or {}
        else:
            log.warning("  shape_info(upstream) failed: %s", r.get("error"))
    except Exception as e:  # noqa: BLE001 — best-effort, must not abort the stage
        log.warning("  shape_info(upstream) errored: %s", e)
    lines = []
    for n in names:
        i = info.get(n)
        if i and "error" in i:
            lines.append(f"  {n} — no longer in the study (used up by a later step)")
            continue
        lines.append(f"  {n}" if not i else
                     f"  {n} — {i['type']}, volume {i['volume']}, area {i['area']}, "
                     f"{i['faces']} faces")
    return ("Shapes built by an EARLIER stage — mesh them by these exact names:\n"
            + "\n".join(lines))


def _retry_block(retry: dict) -> str:
    err = (retry or {}).get("last_error")
    lines = [f"Your previous attempt at this stage failed: {err.get('message')}"] if err else []
    if retry.get("hint"):
        lines.append(f"User hint for this retry: {retry['hint']}")
    return "\n".join(lines)


def _tools_for(client) -> dict:
    return {t["name"]: t for t in client.list_tools(module="smesh")["tools"]}


SMESH_SPEC = register(ModuleSpec(
    name="smesh",
    prompt=_smesh_prompt,
    terminator=Terminator(name="finish_stage", schema=FINISH_STAGE_SCHEMA,
                          description=FINISH_STAGE_DESCRIPTION, handle=_finish),
    execute=lambda client, name, args: client.call_tool(name, args, module="smesh"),
    render_result=_render,
    render_upstream=_upstream,
    render_retry=_retry_block,
    tools_for=_tools_for,
    needs_session=True,
    kickoff=("Start with the first step now. Each tool result tells you what you "
             "need for the next one."),
    no_tool_nudge=("You did not call a tool. Call at least one tool now — "
                   "finish_stage once the mesh is done."),
    on_exhausted=lambda call_log, last_error: {"call_log": call_log,
                                               "last_error": last_error},
    core_tools=CORE_TOOLS,
    categories=("setup", "naming", "local", "group", "compute", "edit", "export", "query"),
    forgets=True,
    checks_results=True,
    calc=True,
    call_prefix="smesh.",
    preamble=PREAMBLE,
    publish_line="",
    catalogue=_CATALOGUE,
))
