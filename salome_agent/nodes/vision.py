import json

from salome_agent.llm import with_images

VISION_SYSTEM_PROMPT = """You are the image-analysis agent of a CAD assistant. You
receive one or more reference images of a mechanical part or assembly (photos,
sketches, technical drawings, plans) plus the user's text request. Your job is to
extract and describe what is visible so that a planner can rebuild it as 2D
geometry in SALOME GEOM if 2D and rebuilt it as 3D geometry in SALOME GEOM if 3D.
You do NOT plan or write CAD code.

Several images are usually different views of the SAME object (front, top, side,
isometric...): read them together and reconcile them into one (2D or 3D) description
instead of describing each picture as a separate object. 

Respond as JSON with exactly these keys, and output only the JSON object:
{
  "summary": one or two sentences: what the object is overall,
  "views": [{"image": 1-based image number, "view": "front|top|side|isometric|
             section|detail|unknown", "shows": what this image contributes}],
  "parts": [{"name": short label,
             "shape": primitive or feature type (box, cylinder, sphere, cone,
                      torus, plate, hole, fillet, chamfer, slot, extrusion...),
             "dimensions": {"<name>": number in millimetres, ...} or {},
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
    known = {"summary", "views", "parts", "dimensions_read", "proportions",
             "symmetry", "uncertainties"}
    for k, v in desc.items():
        if k not in known:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)
