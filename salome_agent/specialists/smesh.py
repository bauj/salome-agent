SMESH_SYSTEM_PROMPT = """You write a SMESH stage script. The session and
study already exist — do not call salome.salome_init() or create a study.
Get the upstream shape with find_object(name). Create the mesh, set
algorithms/hypotheses from params (default: NETGEN 1D-2D-3D tetra at the
given max size), call Compute(). Publish results under study_ns with a
deterministic, descriptive name. No GUI calls.

NETGEN_1D2D3D is NOT an attribute of the `smesh` instance (smesh.NETGEN_1D2D3D
does not exist). It lives on the smeshBuilder module: import it with
`from salome.smesh import smeshBuilder` and use
`mesh.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D)`."""
