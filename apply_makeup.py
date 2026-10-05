"""Apply a rose-gold glam makeup look to the face UV texture.

Only colour layers are added on top of the original texture; geometry, skin
detail and lighting are preserved by modulating each makeup colour with the
local high-frequency luminance of the source skin.

Usage: python3 apply_makeup.py <input.jpg> <output.png>
"""
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

SRC = sys.argv[1] if len(sys.argv) > 1 else "face_texture.jpg"
DST = sys.argv[2] if len(sys.argv) > 2 else "face_texture_makeup.png"

img = np.asarray(Image.open(SRC).convert("RGB")).astype(np.float32) / 255.0
H, W, _ = img.shape
rng = np.random.default_rng(7)

lum = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
# Skin micro-detail ratio (pores, wrinkles) so makeup keeps the texture.
detail = np.clip(lum / (gaussian_filter(lum, 6) + 1e-4), 0.6, 1.5)[..., None]


def mask(draw_fn, blur):
    m = Image.new("L", (W, H), 0)
    draw_fn(ImageDraw.Draw(m))
    a = np.asarray(m).astype(np.float32) / 255.0
    return gaussian_filter(a, blur) if blur else a


def rgb(*c):
    return np.array(c, dtype=np.float32) / 255.0


def tint(base, color, alpha, keep_shading=0.6):
    """Blend a pigment colour while keeping texture and part of the shading."""
    shade = (lum / (gaussian_filter(lum, 40) + 1e-4))[..., None]
    shade = 1.0 + (np.clip(shade, 0.7, 1.3) - 1.0) * keep_shading
    layer = np.clip(color * detail * shade, 0, 1)
    a = alpha[..., None]
    return base * (1 - a) + layer * a


def screen(base, color, alpha):
    a = alpha[..., None]
    layer = 1 - (1 - base) * (1 - color)
    return base * (1 - a) + layer * a


def shimmer(base, region, density, size, strength, color):
    """Sprinkle tiny metallic/glitter specks inside a region."""
    n = rng.random((H, W)).astype(np.float32)
    specks = (n > 1 - density).astype(np.float32)
    specks = gaussian_filter(specks, size) * (size * 6) ** 2 / 4
    specks = np.clip(specks, 0, 1) * region
    return screen(base, color, specks * strength)


def P(pts, ox=0, oy=0, s=1.0):
    return [(ox + x * s, oy + y * s) for x, y in pts]


out = img.copy()

# ------------------------------------------------------------------ EYES ---
# (outer corner, inner corner) of each eye's lash line in texture pixels.
eyes = {
    "L": dict(outer=(660, 1117), inner=(830, 1110), dir=-1),
    "R": dict(outer=(1330, 1117), inner=(1162, 1112), dir=1),
}

for e in eyes.values():
    (ox, oy), (ix, iy), d = e["outer"], e["inner"], e["dir"]
    cx = (ox + ix) / 2
    w = abs(ox - ix)

    def lash(t):  # point along the upper lash line, t=0 inner .. 1 outer
        x = ix + (ox - ix) * t
        y = iy + (oy - iy) * t - 12 * np.sin(np.pi * t)
        return (x, y)

    line = [lash(t) for t in np.linspace(0, 1, 24)]

    # Lid (rose gold / copper base)
    lid_top = [(x, y - 34 - 10 * np.sin(np.pi * t))
               for t, (x, y) in zip(np.linspace(0, 1, 24), line)]
    lid_poly = line + lid_top[::-1]
    lid = mask(lambda g: g.polygon(lid_poly, fill=255), 7)
    out = tint(out, rgb(196, 122, 98), lid * 0.70)
    # copper towards outer half
    outer_half = mask(lambda g: g.ellipse(
        [ox - d * 0 - 60, oy - 55, ox + 60, oy + 5], fill=255), 18)
    out = tint(out, rgb(176, 96, 64), outer_half * lid * 0.45)
    out = screen(out, rgb(120, 80, 60), lid * 0.18)  # metallic sheen
    out = shimmer(out, lid, 0.012, 0.7, 0.55, rgb(255, 214, 180))

    # Crease (terracotta / soft brown), slightly above lid, winged outward
    crease_pts = [(x + d * 6 * t, y - 40 - 12 * np.sin(np.pi * t))
                  for t, (x, y) in zip(np.linspace(0, 1, 24), line)]
    crease = mask(lambda g: g.line(crease_pts, fill=255, width=16), 9)
    out = tint(out, rgb(150, 82, 58), crease * 0.70)
    out = tint(out, rgb(110, 66, 52), crease * 0.35)

    # Inner-corner champagne highlight (frosty, heavy)
    hi = mask(lambda g: g.ellipse([ix - 22, iy - 20, ix + 22, iy + 14],
                                  fill=255), 6)
    out = screen(out, rgb(250, 226, 190), hi * 0.80)
    out = shimmer(out, hi, 0.03, 0.6, 0.9, rgb(255, 245, 225))

    # Eyeliner: thin black line on the upper lash line + subtle wing
    wing_tip = (ox + d * 22, oy - 14)
    liner_pts = [lash(t) for t in np.linspace(0.08, 1, 20)] + [wing_tip]
    widths = mask(lambda g: g.line(liner_pts, fill=255, width=3), 0.7)
    wing = mask(lambda g: g.polygon(
        [lash(0.85), (ox, oy - 3), wing_tip], fill=255), 0.8)
    liner = np.clip(widths + wing, 0, 1)
    out = out * (1 - liner[..., None] * 0.92) + rgb(12, 8, 8) * liner[..., None] * 0.92

    # Light-coloured lower waterline
    water = [(ix + (ox - ix) * t + 0, iy + 6 + (oy - iy) * t + 2 * np.sin(np.pi * t))
             for t in np.linspace(0.12, 0.9, 16)]
    wl = mask(lambda g: g.line(water, fill=255, width=2), 0.8)
    out = screen(out, rgb(230, 200, 180), wl * 0.55)

# ---------------------------------------------------------------- CHEEKS ---
for cx, cy, d in [(600, 1300, -1), (1340, 1300, 1)]:
    # Blush on the apples, swept up toward the temples
    m = Image.new("L", (W, H), 0)
    g = ImageDraw.Draw(m)
    g.ellipse([cx - 120, cy - 70, cx + 120, cy + 70], fill=255)
    m = m.rotate(d * 18, center=(cx, cy))
    blush = gaussian_filter(np.asarray(m).astype(np.float32) / 255.0, 40)
    out = tint(out, rgb(214, 104, 124), blush * 0.40, keep_shading=0.9)

    # Luminous highlight on the cheekbone high points
    hx, hy = cx + d * 25, cy - 85
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).ellipse([hx - 95, hy - 24, hx + 95, hy + 24], fill=255)
    m = m.rotate(d * 20, center=(hx, hy))
    hl = gaussian_filter(np.asarray(m).astype(np.float32) / 255.0, 18)
    out = screen(out, rgb(245, 215, 185), hl * 0.55)
    out = shimmer(out, hl, 0.004, 0.7, 0.35, rgb(255, 240, 220))

# Nose bridge highlight
bridge = mask(lambda g: g.line([(965, 1030), (965, 1240)], fill=255,
                               width=18), 10)
bridge *= mask(lambda g: g.ellipse([940, 980, 990, 1290], fill=255), 10)
out = screen(out, rgb(245, 218, 190), bridge * 0.50)

# ------------------------------------------------------------------ LIPS ---
O = dict(ox=800, oy=1330, s=0.5)  # lip points drawn on a 2x crop
upper_outer = P([(112, 368), (150, 292), (200, 214), (255, 146), (320, 120),
                 (400, 134), (482, 118), (545, 140), (598, 200), (645, 280),
                 (688, 368)], **O)
upper_inner = P([(688, 368), (620, 300), (560, 274), (400, 260), (250, 274),
                 (180, 300), (112, 368)], **O)
lower_inner = P([(140, 376), (220, 348), (400, 352), (580, 342), (660, 372)],
                **O)
lower_outer = P([(660, 372), (622, 424), (548, 468), (400, 492), (262, 472),
                 (190, 428), (140, 376)], **O)
upper_poly = upper_outer + upper_inner
lower_poly = lower_inner + lower_outer

lips = mask(lambda g: (g.polygon(upper_poly, fill=255),
                       g.polygon(lower_poly, fill=255)), 2.5)
lip_core = mask(lambda g: (g.polygon(upper_poly, fill=255),
                           g.polygon(lower_poly, fill=255)), 10)
edge = np.clip(lips - lip_core * 1.15, 0, 1)

# Metallic pink / frosted rose body
out = tint(out, rgb(206, 104, 134), lips * 0.85, keep_shading=0.8)
centre = mask(lambda g: g.ellipse([905, 1405, 1100, 1575], fill=255), 30)
out = tint(out, rgb(232, 156, 172), lips * centre * 0.55, keep_shading=0.8)
out = screen(out, rgb(110, 70, 80), lips * 0.20)  # frost

# Deep plum / espresso overdrawn liner
liner_ring = mask(lambda g: (g.line(upper_outer, fill=255, width=9),
                             g.line(lower_outer, fill=255, width=9)), 2.5)
out = tint(out, rgb(58, 24, 32), np.clip(liner_ring * 0.95 + edge * 0.7, 0, 0.95),
           keep_shading=0.5)

# Gloss highlights + glitter
gloss = mask(lambda g: (g.ellipse([945, 1520, 1050, 1555], fill=255),
                        g.ellipse([960, 1410, 1040, 1428], fill=255)), 6)
out = screen(out, rgb(255, 236, 240), gloss * lips * 0.55)
out = shimmer(out, lips * lip_core, 0.02, 0.6, 0.8, rgb(255, 225, 235))

# Keep the black background (outside UV islands) untouched
bg = (img.max(axis=2) < 0.04)[..., None]
out = np.where(bg, img, out)

Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)).save(DST)
print("saved", DST)
