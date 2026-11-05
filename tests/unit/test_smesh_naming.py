# Copyright (C) 2026  salome-agent authors
# LGPL 2.1 or later
from executor.smesh.naming import same_ids_note
from executor.smesh.state import GeoGroup


def test_same_ids_note_names_a_group_over_the_same_sub_shapes():
    groups = {"appui_geom": GeoGroup("flange", "shaper", "face", ("F4",)),
              "trous_geom": GeoGroup("flange", "shaper", "face", ("F1", "F11", "F2")),
              "other": GeoGroup("plate", "shaper", "face", ("F4",))}
    note = same_ids_note(groups, "flange", ["F4"])
    assert note.startswith("the same sub-shapes of flange are already the group appui_geom")
    assert "trous_geom" in same_ids_note(groups, "flange", ["F2", "F1", "F11"])
    assert same_ids_note(groups, "flange", ["F1"]) == ""
    assert same_ids_note({"lid": GeoGroup("box", "geom", "face", ("3",))}, "box", [3])
