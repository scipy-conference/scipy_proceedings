# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow"]
# ///
"""Compose Figure 1: top-down dispatch flow (left) beside the
rasterized key-derivation listing (right).

Inputs (both rendered at 300 ppi):
  fig1_dispatch_flow.png  — `dot -Tpng -Gdpi=300 fig1_dispatch.dot`
  alg_hash.png            — `typst compile --format png --ppi 300 alg_hash.typ`

Output: fig1_dispatch.png, transparent background, panels scaled to a
common height with a small gutter.
"""

from PIL import Image

GUTTER = 90  # px at 300 ppi ≈ 7.6 mm

flow = Image.open("fig1_dispatch_flow.png").convert("RGBA")
alg = Image.open("alg_hash.png").convert("RGBA")

# Scale the listing to the flow's height (flow sets the height budget).
h = flow.height
alg = alg.resize((round(alg.width * h / alg.height), h), Image.LANCZOS)

canvas = Image.new("RGBA", (flow.width + GUTTER + alg.width, h), (0, 0, 0, 0))
canvas.paste(flow, (0, 0), flow)
canvas.paste(alg, (flow.width + GUTTER, 0), alg)
canvas.save("fig1_dispatch.png")
print(f"fig1_dispatch.png: {canvas.width}x{canvas.height}")
