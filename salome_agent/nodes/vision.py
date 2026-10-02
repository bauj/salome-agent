import base64
import binascii
import json
import struct

from salome_agent.llm import with_images
from salome_agent.logs import log
from salome_agent.nodes.vision_spec import attach_conflicts, format_errors, validate_spec

VISION_SYSTEM_PROMPT = """You are the image-analysis agent of a CAD assistant. You
receive one or more reference images (photos, sketches, technical drawings, plans,
renders) plus the user's text request. Your job is to turn what is visible into a
structured spec that a planner and the SALOME GEOM tools rebuild. You do NOT plan
or write CAD code.

First decide whether the TARGET is 2D or 3D -- every image is flat, so this is
about the object to build, not the picture:
- "2d": a planar shape with no thickness to build -- an outline, a profile, a
  cross-section, a floor plan, a flat sketch of contours. It is rebuilt as a
  planar face.
- "3d": a solid or an assembly with depth -- a photo, a render, an isometric or
  perspective view, shading, several orthographic views (front/top/side) of the
  same part, or any thickness/depth/height dimension. A technical drawing of a
  3D part is "3d" even though the drawing itself is flat.
The text request wins over the image: "extrude", "plate 5 mm thick", "solid" make
it 3d; "profile", "outline", "face", "2D", "sketch only" make it 2d. When the
image(s) alone cannot tell (a single outline with no depth cue and a request
that does not say), choose "2d", set "assumed": true and list the missing depth
under uncertainties.

Several images are usually different views of the SAME object (front, top, side,
isometric...): read them together and reconcile them into one spec instead of
describing each picture as a separate object.

Respond as JSON with exactly these keys, in this order, and output only the JSON
object:
{
  "dimensionality": {"value": "2d" or "3d", "assumed": true only when nothing
                     told you, else false, "reason": the cue that decided it},
  "summary": one or two sentences: what the object is overall,
  "views": [{"image": 1-based image number, "view": "front|top|side|isometric|
             section|plan|detail|photo|unknown", "shows": what it contributes}],
  "dimensions_read": [{"text": a dimension, annotation or scale exactly as
                       written, with its unit, "means": what it measures or the
                       feature it labels; for a linear dimension also
                       "from": what the first extension line sits on and
                       "to": what the second one sits on (a centre, an edge,
                       a tangent point)}],
  "datum": the origin and axes every coordinate below uses, e.g. "O = centre of
           the big bore; +X right, +Y up, +Z towards the viewer",
  "parts": [one entry per part, see below],
  "uncertainties": ["anything ambiguous, hidden or not inferable from the image(s)"]
}

Each part:
{
  "id": "P1", "P2", ... (unique),
  "name": short label,
  "base": exactly one of
    {"kind": "profile", "plane": "XY" | "YZ" | "XZ",
     "profile": {"start": "x,y,z", "segments": [...], "holes": [...]},
     "extrude": {"height": h, "direction": "x,y,z" (optional)} or null,
     "revolve": {"point": "x,y,z", "direction": "x,y,z", "angle": degrees} or null}
      -- an outline the image gives exactly (a dimensioned drawing or sketch); at
      most one of extrude / revolve, neither for a "2d" target;
    {"kind": "primitive", "shape": "box" | "cylinder" | "sphere" | "cone" | "torus",
     "dims": {"<name>": number, ...}};
    {"kind": "approx", "description": what the shape is,
     "dims": {"<name>": number, ...}}
      -- a shape you can only estimate (photos, renders): its overall sizes,
  "placement": {"origin": "x,y,z", "normal": "x,y,z", "x_dir": "x,y,z" (optional)}
      -- where the part sits in the datum frame and which way its base faces,
  "features": [features cut into or added to this part, in order:
    {"type": "hole", "center": "x,y,z", "diameter": d, "depth": d or "through",
     "axis": "x,y,z" (optional)},
    {"type": "fillet", "radius": r, "where": which edges, in words},
    {"type": "chamfer", "size": s, "where": which edges, in words},
    {"type": "pattern", "of": 1-based index of the feature it repeats,
     "kind": "circular" | "linear", "count": n, plus its axis or step as drawn}],
  "relation": how it sits against the other parts, in words, or null,
  "ops": [{"op": "cut" | "fuse", "with": another part's id}] (optional)
}

A profile is "start" then "segments", each {"to": "x,y,z"} -- a straight line --
plus at most one of "center": "x,y,z" (a circular arc about that centre, the
shorter way round; add "reverse": true for the longer), "through": "x,y,z" (a
point on the arc; use it for a half circle) or "via": ["x,y,z", ...] (a smooth
curve), and optionally "fillet": r, which rounds the corner at that segment's
"to": give the sharp corner as drawn, the tool finds the tangent points. If the
last "to" is not "start", a straight line closes the outline. "holes" are
outlines of the same form, {"center": "x,y,z", "radius": r} circles,
{"slot": {"from": "x,y,z", "to": "x,y,z", "width": w}} straight slots (the centre
line's ends) or {"slot": {"center": "x,y,z", "radius": r, "width": w,
"from_angle": a, "to_angle": b}} slots along an arc (degrees from +X). All points
are "x,y,z" in the datum frame and lie in the profile's plane.

Rules:
- Write "dimensions_read" first. Every number in "parts" is one of those values,
  derived from them, or an estimate written as a string with "~", e.g. "~40"
  (also inside points: "~40,0,0").
- Place centres from their dimensions: when a linear dimension runs from one
  centre to another, it is centre-to-centre, so the second centre is exactly that
  far from the first; size the outline around the centres afterwards.
- Millimetres throughout; convert a drawing in another unit and say so in
  dimensions_read.
- For "2d", never invent a thickness or height; for "3d", give the depth of every
  part, estimated with "~" if no view shows it.
- Never invent detail that is not visible; list it under uncertainties instead.
- Reading technical drawings:
  - A linear dimension measures between its two extension lines: say what each one
    sits on (a centre line, an edge, a tangent point). Ending on a centre line means
    the distance is to that centre, not to an edge.
  - A radius or diameter arrowed to a chain-dotted arc or circle is a pitch/centre-
    line radius (bolt circle, slot centre line), not an edge.
  - A curved slot with R ends is as wide as twice that R, centred on its pitch arc.
  - "nX" counts identical features; "TYP" applies the value to every like feature.
  - Give the centre of every arc and circle as coordinates from the datum.
- Before answering, check that the outline closes: consecutive arcs and lines must
  meet or be tangent at the centres and radii you give. If they do not, re-read the
  dimensions; if a conflict remains, name the conflicting values in uncertainties.
- Keep it factual and compact: no CAD tool names, no step-by-step build order."""


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    i = 2
    while i + 9 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0x01, *range(0xD0, 0xDA)):
            i += 2
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return None


def _image_size(data: bytes) -> tuple[int, int] | None:
    if data.startswith(b"\x89PNG") and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"\xff\xd8"):
        return _jpeg_size(data)
    if data[:4] in (b"GIF8",):
        return struct.unpack("<HH", data[6:10])
    if data.startswith(b"BM"):
        w, h = struct.unpack("<ii", data[18:26])
        return w, abs(h)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8X":
            return (1 + int.from_bytes(data[24:27], "little"),
                    1 + int.from_bytes(data[27:30], "little"))
        if chunk == b"VP8L":
            bits = int.from_bytes(data[21:25], "little")
            return 1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF)
        if chunk == b"VP8 ":
            w, h = struct.unpack("<HH", data[26:30])
            return w & 0x3FFF, h & 0x3FFF
    return None


def image_size(uri: str) -> tuple[int, int] | None:
    """(width, height) in pixels of a base64 data: URI image, read from its
    header; None when it cannot tell."""
    head, _, payload = uri.partition(",")
    if not head.startswith("data:"):
        return None
    try:
        size = _image_size(base64.b64decode(payload))
    except (binascii.Error, struct.error, ValueError):
        return None
    return size if size and all(size) else None


def describe_images(request: str, images: list[str], llm, fix_retries: int = 2) -> dict:
    """Run the vision agent: one multimodal call that turns the reference image(s)
    into a structured spec, re-asked with the validator's errors up to
    fix_retries times; errors still left are attached as conflicts."""
    text = ("User request (context only -- describe the image(s), do not plan):\n"
            + request)
    # A model that measures in pixels otherwise guesses the image size, and a
    # wrong guess skews every scale it derives.
    sizes = [f"image {i}: {s[0]} x {s[1]} px"
             for i, s in enumerate(map(image_size, images), 1) if s]
    if sizes:
        text += "\n\nImage size (width x height): " + "; ".join(sizes) + "."
    messages = [{"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": with_images(text, images)}]
    spec = llm.chat(messages=messages, module="vision")
    for attempt in range(fix_retries + 1):
        errors = validate_spec(spec)
        if not errors:
            return spec
        if attempt == fix_retries:
            log.warning("vision spec still has %d problem(s); passing them on as conflicts",
                        len(errors))
            return attach_conflicts(spec, errors)
        log.info("vision spec has %d problem(s), asking for a fix (%d/%d):\n%s",
                 len(errors), attempt + 1, fix_retries, format_errors(errors))
        messages = messages + [
            {"role": "assistant", "content": json.dumps(spec, ensure_ascii=False)},
            {"role": "user", "content":
                "The spec has these problems:\n" + format_errors(errors)
                + "\nRe-read the image(s) where needed, fix them, and answer with the "
                  "whole corrected JSON spec only."}]
        spec = llm.chat(messages=messages, module="vision")
    return spec
