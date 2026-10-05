"""Apply the rose-gold glam face-chart look to the face UV texture.

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
AXIS = 997  # vertical symmetry axis of the face in the texture

lum = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
# Skin micro-detail ratio (pores, wrinkles) so makeup keeps the texture.
detail = np.clip(lum / (gaussian_filter(lum, 6) + 1e-4), 0.6, 1.5)[..., None]
shade_full = np.clip(lum / (gaussian_filter(lum, 40) + 1e-4), 0.7, 1.3)[..., None]


def mask(draw_fn, blur):
    m = Image.new("L", (W, H), 0)
    draw_fn(ImageDraw.Draw(m))
    a = np.asarray(m).astype(np.float32) / 255.0
    return gaussian_filter(a, blur) if blur else a


def rot_ellipse(cx, cy, rx, ry, angle, blur):
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    m = m.rotate(angle, center=(cx, cy))
    return gaussian_filter(np.asarray(m).astype(np.float32) / 255.0, blur)


def rgb(*c):
    return np.array(c, dtype=np.float32) / 255.0


def tint(base, color, alpha, keep_shading=0.6):
    """Blend a pigment colour while keeping texture and part of the shading."""
    shade = 1.0 + (shade_full - 1.0) * keep_shading
    layer = np.clip(color * detail * shade, 0, 1)
    a = np.clip(alpha, 0, 1)[..., None]
    return base * (1 - a) + layer * a


def screen(base, color, alpha):
    a = np.clip(alpha, 0, 1)[..., None]
    layer = 1 - (1 - base) * (1 - color)
    return base * (1 - a) + layer * a


def multiply(base, color, alpha):
    a = np.clip(alpha, 0, 1)[..., None]
    return base * (1 - a) + base * color * a


def paint(base, color, alpha):
    a = np.clip(alpha, 0, 1)[..., None]
    return base * (1 - a) + color * a


def shimmer(base, region, density, size, strength, color):
    """Sprinkle tiny metallic/glitter specks inside a region."""
    n = rng.random((H, W)).astype(np.float32)
    specks = (n > 1 - density).astype(np.float32)
    specks = gaussian_filter(specks, size) * (size * 6) ** 2 / 4
    specks = np.clip(specks, 0, 1) * region
    return screen(base, color, specks * strength)


def glint(base, x, y, size, strength=1.0):
    """Four-point star sparkle, like the highlights on the chart's lips."""
    star = mask(lambda g: (g.line([(x - size, y), (x + size, y)], fill=255, width=2),
                           g.line([(x, y - size * 0.8), (x, y + size * 0.8)], fill=255, width=2),
                           g.line([(x - size * .35, y - size * .35),
                                   (x + size * .35, y + size * .35)], fill=160, width=1),
                           g.line([(x - size * .35, y + size * .35),
                                   (x + size * .35, y - size * .35)], fill=160, width=1)), 0.8)
    core = mask(lambda g: g.ellipse([x - size * .3, y - size * .3,
                                     x + size * .3, y + size * .3], fill=255), size * 0.25)
    base = screen(base, rgb(255, 225, 245), core * strength)
    return screen(base, rgb(255, 245, 255), star * strength)


def P(pts, ox=0, oy=0, s=1.0):
    return [(ox + x * s, oy + y * s) for x, y in pts]


def bezier(p0, p1, p2, n=40):
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)
    return [tuple(p) for p in pts]


out = img.copy()

# --------------------------------------------------------------- CHEEKS ---
# Strong mauve-rose blush draped from the cheekbones toward the temples.
for d in (-1, 1):
    cx, cy = AXIS + d * 400, 1320
    blush = rot_ellipse(cx, cy, 190, 125, d * 22, 55)
    blush += rot_ellipse(AXIS + d * 520, 1200, 140, 90, d * 35, 50) * 0.6
    out = tint(out, rgb(168, 78, 102), blush * 0.55, keep_shading=0.9)
    out = tint(out, rgb(196, 104, 128), rot_ellipse(cx, cy - 20, 120, 70, d * 22, 40) * 0.25,
               keep_shading=0.9)

# Soft contour on the sides of the nose (reference shades the nose walls)
for d in (-1, 1):
    side = mask(lambda g: g.line([(AXIS + d * 48, 1060), (AXIS + d * 58, 1230)],
                                 fill=255, width=26), 14)
    out = multiply(out, rgb(150, 105, 95), side * 0.35)

# Nose bridge highlight: bright pink glow at the top, fading down; plus tip.
bridge = mask(lambda g: g.polygon([(AXIS - 22, 1040), (AXIS + 22, 1040),
                                   (AXIS + 9, 1230), (AXIS - 9, 1230)], fill=255), 9)
bridge *= np.interp(np.arange(H), [1000, 1080, 1230], [1, 1, 0.55])[:, None]
out = screen(out, rgb(226, 160, 176), bridge * 0.45)
out = screen(out, rgb(250, 190, 210), rot_ellipse(AXIS, 1080, 28, 40, 0, 12) * 0.45)
out = screen(out, rgb(245, 205, 210), rot_ellipse(AXIS, 1272, 26, 16, 0, 8) * 0.55)

# Chin and forehead-centre touches of highlight
out = screen(out, rgb(235, 190, 200), rot_ellipse(AXIS, 1655, 55, 26, 0, 16) * 0.35)
out = screen(out, rgb(235, 200, 190), rot_ellipse(AXIS, 820, 60, 110, 0, 40) * 0.20)

# ---------------------------------------------------------------- EYES ---
eyes = {
    "L": dict(outer=(660, 1117), inner=(830, 1110), d=-1),
    "R": dict(outer=(1330, 1117), inner=(1162, 1112), d=1),
}

for e in eyes.values():
    (ox, oy), (ix, iy), d = e["outer"], e["inner"], e["d"]

    def lash(t):  # point on the upper lash line, t=0 inner .. 1 outer
        return (ix + (ox - ix) * t, iy + (oy - iy) * t - 12 * np.sin(np.pi * t))

    ts = np.linspace(0, 1, 30)
    line = [lash(t) for t in ts]

    # Brows: defined, softly arched, warm dark brown, feathered fill
    head = (ix + d * 25, iy - 76)
    peak = (ix + (ox - ix) * 0.62, iy - 110)
    tail = (ox + d * 22, oy - 86)
    upper = bezier(head, (ix + (ox - ix) * 0.3, iy - 108), peak, 20) + \
        bezier(peak, (ox + d * 5, oy - 108), tail, 20)[1:]
    lower = bezier(tail, (ox - d * 10, oy - 92), peak, 20)[1:] + \
        bezier((peak[0], peak[1] + 16), (ix + (ox - ix) * 0.3, iy - 86),
               (head[0], head[1] + 14), 20)
    brow = mask(lambda g: g.polygon(upper + lower, fill=255), 2.5)
    hair = rng.random((H, W)).astype(np.float32)
    hair = gaussian_filter(hair, (0.6, 2.2))  # short strokes
    hair = np.clip((hair - hair.mean()) * 6 + 0.9, 0.65, 1.0)
    out = tint(out, rgb(84, 46, 36), brow * hair * 0.88, keep_shading=0.4)

    # Brow-bone glow under the arch (soft pink-champagne)
    bone = mask(lambda g: g.line(bezier((ix, iy - 62), (ix + (ox - ix) * .55, iy - 92),
                                        (ox, oy - 70), 30), fill=255, width=14), 9)
    out = screen(out, rgb(225, 175, 165), bone * 0.40)

    # Lid: peach / rose-gold shimmer that fills the lid
    lid_top = [(x, y - 40 - 16 * np.sin(np.pi * t)) for t, (x, y) in zip(ts, line)]
    lid = mask(lambda g: g.polygon(line + lid_top[::-1], fill=255), 8)
    out = tint(out, rgb(230, 148, 140), lid * 0.82, keep_shading=0.4)
    out = screen(out, rgb(120, 70, 60), lid * 0.25)
    out = shimmer(out, lid, 0.018, 0.7, 0.7, rgb(255, 220, 200))

    # Crease & outer V: deeper rose-brown, swept out into a sharp wing
    crease = [(x + d * 10 * t, y - 46 - 18 * np.sin(np.pi * t)) for t, (x, y) in zip(ts, line)]
    wing_tip = (ox + d * 95, oy - 52)
    outer_v = [lash(0.55), crease[17], crease[-1], wing_tip, (ox + d * 25, oy - 6)]
    vmask = mask(lambda g: (g.line(crease, fill=255, width=22),
                            g.polygon(outer_v, fill=255)), 9)
    out = tint(out, rgb(160, 80, 86), vmask * 0.72, keep_shading=0.5)
    out = tint(out, rgb(112, 56, 58), mask(lambda g: g.polygon(outer_v, fill=255), 12) * 0.45,
               keep_shading=0.5)
    out = shimmer(out, vmask * 0.6, 0.008, 0.7, 0.4, rgb(255, 200, 185))

    # Under-eye: champagne-pink shimmer along the lower lash line and a bright
    # inner-corner cut that runs down beside the nose bridge.
    lower_line = [(ix + (ox - ix) * t, iy + 8 + (oy - iy) * t + 6 * np.sin(np.pi * t))
                  for t in np.linspace(0, 1, 24)]
    under = mask(lambda g: g.line(lower_line, fill=255, width=12), 5)
    inner_cut = mask(lambda g: g.polygon([(ix - d * 6, iy - 18), (ix + d * 40, iy + 6),
                                          (ix + d * 34, iy + 26), (ix - d * 10, iy + 70),
                                          (ix - d * 30, iy + 12)], fill=255), 9)
    out = screen(out, rgb(240, 176, 176), under * 0.50)
    out = screen(out, rgb(246, 186, 186), inner_cut * 0.70)
    out = shimmer(out, np.clip(inner_cut + under * 0.7, 0, 1), 0.02, 0.6, 0.75,
                  rgb(255, 235, 235))

    # White-pink lower waterline / inner rim
    water = [(ix + (ox - ix) * t, iy + 5 + (oy - iy) * t + 2 * np.sin(np.pi * t))
             for t in np.linspace(0.03, 0.92, 18)]
    wl = mask(lambda g: g.line(water, fill=255, width=3), 0.8)
    out = screen(out, rgb(246, 214, 222), wl * 0.60)

    # Eyeliner: thin black upper line with a small flick
    flick = (ox + d * 26, oy - 16)
    liner = mask(lambda g: (g.line([lash(t) for t in np.linspace(0.06, 1, 20)] + [flick],
                                   fill=255, width=3),
                            g.polygon([lash(0.86), (ox, oy - 2), flick], fill=255)), 0.7)
    out = paint(out, rgb(14, 9, 10), liner * 0.92)

    # Wispy upper lashes, longer toward the outer corner, and fine lower lashes
    lashes = Image.new("L", (W, H), 0)
    g = ImageDraw.Draw(lashes)
    for t in np.linspace(0.1, 1.0, 34):
        x, y = lash(t)
        ln = 10 + 18 * t + rng.uniform(-3, 3)
        ang = np.deg2rad(rng.uniform(-6, 6)) + d * (0.15 + 0.9 * t)
        tip = (x + np.sin(ang) * ln, y + np.cos(ang) * ln * 0.6)
        g.line([(x, y), ((x + tip[0]) / 2 + d * 2, (y + tip[1]) / 2 + 1), tip],
               fill=235, width=2)
    for t in np.linspace(0.3, 0.95, 12):
        x, y = water[int(t * (len(water) - 1))]
        ln = 5 + 6 * t
        g.line([(x, y + 2), (x + d * ln * 0.5, y + 2 + ln)], fill=120, width=1)
    lm = gaussian_filter(np.asarray(lashes).astype(np.float32) / 255.0, 0.6)
    out = paint(out, rgb(18, 12, 14), lm * 0.85)

# ---------------------------------------------------------------- LIPS ---
O = dict(ox=800, oy=1330, s=0.5)  # lip points drawn on a 2x crop
upper_outer = P([(104, 368), (146, 290), (198, 210), (255, 140), (322, 112),
                 (400, 130), (480, 110), (548, 136), (602, 198), (650, 280),
                 (696, 368)], **O)
upper_inner = P([(696, 368), (620, 300), (560, 274), (400, 260), (250, 274),
                 (180, 300), (104, 368)], **O)
lower_inner = P([(132, 376), (220, 348), (400, 352), (580, 342), (668, 372)], **O)
lower_outer = P([(668, 372), (628, 430), (552, 476), (400, 500), (258, 480),
                 (186, 434), (132, 376)], **O)
upper_poly = upper_outer + upper_inner
lower_poly = lower_inner + lower_outer

# Pink glow on the skin above the cupid's bow (as in the chart)
bow = rot_ellipse(1000, 1382, 80, 14, 0, 8)
out = screen(out, rgb(240, 190, 215), bow * 0.55)

lips = mask(lambda g: (g.polygon(upper_poly, fill=255), g.polygon(lower_poly, fill=255)), 1.8)
# distance-like falloff from the lip edge -> ombre (dark outside, light centre)
inner = mask(lambda g: (g.polygon(upper_poly, fill=255), g.polygon(lower_poly, fill=255)), 14)
centre = np.clip((inner - 0.72) / 0.28, 0, 1) * lips

# Deep plum / espresso base everywhere on the lips
out = tint(out, rgb(78, 30, 40), lips * 0.95, keep_shading=0.7)
# Lilac-pink metallic centre
out = tint(out, rgb(214, 146, 186), centre * 0.85, keep_shading=0.7)
out = screen(out, rgb(150, 100, 140), centre * 0.35)
# Gloss bands on upper and lower lip
gloss = (rot_ellipse(1000, 1532, 70, 20, 0, 9) + rot_ellipse(950, 1425, 34, 12, -8, 7) +
         rot_ellipse(1052, 1425, 34, 12, 8, 7))
out = screen(out, rgb(255, 225, 245), gloss * lips * 0.65)
# Sharp espresso liner on the outer contour
liner_ring = mask(lambda g: (g.line(upper_outer, fill=255, width=6),
                             g.line(lower_outer, fill=255, width=6)), 1.5)
out = tint(out, rgb(50, 20, 24), liner_ring * lips.clip(0.3, 1) * 0.9, keep_shading=0.4)
# Glitter specks and star glints
out = shimmer(out, lips, 0.010, 0.6, 0.8, rgb(255, 215, 225))
out = shimmer(out, centre, 0.006, 1.0, 0.8, rgb(255, 240, 250))
for x, y, s in [(938, 1418, 16), (1062, 1420, 14), (1000, 1530, 18), (1066, 1548, 11),
                (944, 1540, 10)]:
    out = glint(out, x, y, s, 0.9)

# Keep the black background (outside UV islands) untouched
bg = (img.max(axis=2) < 0.04)[..., None]
out = np.where(bg, img, out)

Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)).save(DST)
print("saved", DST)
