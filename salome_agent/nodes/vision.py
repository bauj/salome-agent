import json

from salome_agent.llm import with_images

VISION_SYSTEM_PROMPT = """You are the image-analysis agent of a CAD assistant. You
receive one or more reference images (photos, sketches, technical drawings, plans,
renders) plus the user's text request. Your job is to extract and describe what is
visible so that a planner can rebuild it in SALOME GEOM. You do NOT plan or write
CAD code.

First decide whether the TARGET is 2D or 3D -- every image is flat, so this is
about the object to build, not the picture:
- "2d": a planar shape with no thickness to build -- an outline, a profile, a
  cross-section, a floor plan, a flat sketch of contours. It is rebuilt as wires
  or planar faces in one plane.
- "3d": a solid or an assembly with depth -- a photo, a render, an isometric or
  perspective view, shading, several orthographic views (front/top/side) of the
  same part, or any thickness/depth/height dimension. A technical drawing of a
  3D part is "3d" even though the drawing itself is flat.
The text request wins over the image: "extrude", "plate 5 mm thick", "solid" make
it 3d; "profile", "outline", "face", "2D", "sketch only" make it 2d. When the
image(s) alone cannot tell (a single outline with no depth cue and a request
that does not say), choose "2d" and list the missing depth under uncertainties.

Several images are usually different views of the SAME object (front, top, side,
isometric...): read them together and reconcile them into one description instead
of describing each picture as a separate object.

Respond as JSON with exactly these keys, and output only the JSON object:
{
  "dimensionality": "2d" or "3d",
  "dimensionality_reason": one short sentence: the cue that decided it,
  "summary": one or two sentences: what the object is overall,
  "views": [{"image": 1-based image number, "view": "front|top|side|isometric|
             section|plan|detail|unknown", "shows": what this image contributes}],
  "plane": for "2d" only, the plane the shape lies in ("XY" unless the image or
           request says otherwise); null for "3d",
  "parts": [{"name": short label,
             "shape": for "2d": a planar element (rectangle, circle, arc, polygon,
                      polyline, slot, hole, fillet, chamfer...);
                      for "3d": a primitive or feature (box, cylinder, sphere, cone,
                      torus, plate, hole, fillet, chamfer, slot, extrusion...),
             "dimensions": {"<name>": number in millimetres, ...} or {} -- in-plane
                           sizes only for "2d" (no thickness or height),
             "position": placement relative to the other parts / the origin,
             "operations": booleans/transforms applied (cut, fuse, pattern...)}],
  "dimensions_read": ["every dimension, annotation or scale explicitly written
                      in the image(s), with its unit as written"],
  "proportions": approximate relative proportions between the main parts,
  "symmetry": axes/planes of symmetry or repetition patterns, if any,
  "uncertainties": ["anything ambiguous, hidden or not inferable from the image(s)"]
}

Rules:
- Distinguish a dimension READ from the image (put it in dimensions_read and use it
  in parts[].dimensions) from one you ESTIMATED from proportions. Prefix estimated
  values in parts[].dimensions with "~" by giving them as strings, e.g. "~40".
- Convert to millimetres when the drawing states another unit, and say so in
  dimensions_read.
- For "2d", never invent a thickness, height or extrusion; for "3d", give the depth
  of every part, estimated with "~" if no view shows it.
- Never invent detail that is not visible; list it under uncertainties instead.
- Keep it factual and compact: no CAD tool names, no step-by-step build order."""


def describe_images(request: str, images: list[str], llm) -> dict:
    """Run the vision agent: one multimodal call that turns the reference
    image(s) into a structured description for the planner."""
    text = ("User request (context only -- describe the image(s), do not plan):\n"
            + request)
    return llm.chat(
        messages=[{"role": "system", "content": VISION_SYSTEM_PROMPT},
                  {"role": "user", "content": with_images(text, images)}],
        module="vision",
    )


def format_description(desc: dict) -> str:
    """Render the vision agent's JSON as a readable text block for the planner."""
    if not isinstance(desc, dict) or not desc:
        return json.dumps(desc, ensure_ascii=False)
    lines = []
    dim = str(desc.get("dimensionality") or "").lower()
    if dim in ("2d", "3d"):
        target = (f"Target: {dim.upper()}"
                  + (f" ({desc['dimensionality_reason']})" if desc.get("dimensionality_reason") else ""))
        if dim == "2d":
            target += (f" -- a planar shape in the {desc.get('plane') or 'XY'} plane, "
                       "build it without thickness")
        lines.append(target)
    if desc.get("summary"):
        lines.append(f"Summary: {desc['summary']}")
    for v in desc.get("views") or []:
        lines.append(f"Image {v.get('image', '?')} ({v.get('view', 'unknown')}): "
                     f"{v.get('shows', '')}")
    parts = desc.get("parts") or []
    if parts:
        lines.append("Parts:")
    for p in parts:
        bits = [f"{p.get('name', '?')} -- {p.get('shape', '?')}"]
        if p.get("dimensions"):
            bits.append(f"dimensions {json.dumps(p['dimensions'], ensure_ascii=False)}")
        if p.get("position"):
            bits.append(f"position: {p['position']}")
        if p.get("operations"):
            bits.append(f"operations: {p['operations']}")
        lines.append("  - " + "; ".join(bits))
    if desc.get("dimensions_read"):
        lines.append("Dimensions read from the image(s): "
                     + "; ".join(str(d) for d in desc["dimensions_read"]))
    if desc.get("proportions"):
        lines.append(f"Proportions: {desc['proportions']}")
    if desc.get("symmetry"):
        lines.append(f"Symmetry / patterns: {desc['symmetry']}")
    if desc.get("uncertainties"):
        lines.append("Uncertainties: " + "; ".join(str(u) for u in desc["uncertainties"]))
    known = {"dimensionality", "dimensionality_reason", "plane", "summary", "views", "parts", "dimensions_read", "proportions",
             "symmetry", "uncertainties"}
    for k, v in desc.items():
        if k not in known:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)
