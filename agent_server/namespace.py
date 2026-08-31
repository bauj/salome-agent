import os
import sys

# salome_init()'s import chain mutates sys.path (KERNEL's extend_path), which
# would drop a later sys.path.insert — put the repo root on the path here, at
# module load, before anything salome_init-adjacent runs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_object_factory(salome_kernel):
    def find_object(name: str):
        def walk(sobj):
            it = salome_kernel.myStudy.NewChildIterator(sobj)
            while it.More():
                child = it.Value()
                if child.GetName() == name:
                    return child.GetObject()
                found = walk(child)
                if found is not None:
                    return found
                it.Next()
            return None
        result = walk(salome_kernel.myStudy.FindObjectID("0:1"))
        if result is None:
            raise LookupError(f"no study object named {name!r}")
        return result
    return find_object


def build_namespace(live: bool) -> dict:
    if not live:
        from salome.kernel import KernelBasis
        KernelBasis.setSSLMode(True)
    from salome.kernel import salome
    salome.salome_init()   # guarded — no-op if the GUI already ran it (live=True)
    from salome.geom import geomBuilder
    geompy = geomBuilder.New()
    from salome.smesh import smeshBuilder
    smesh = smeshBuilder.New()
    try:
        import pvsimple
    except ImportError:
        pvsimple = None
    return {
        "salome": salome, "geompy": geompy, "smesh": smesh, "pvsimple": pvsimple,
        "study_ns": "salome_agent_run",
        "find_object": _find_object_factory(salome),
    }
