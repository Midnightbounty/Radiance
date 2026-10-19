"""Draw the Radiance icon: a warm light in the corner of a dark room, with its light bouncing off the walls.
Writes radiance/radiance.ico (all Windows sizes) and build/icon_preview.png.
    python build/make_icon.py"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 1024


def render(n=N):
    y, x = np.mgrid[0:n, 0:n].astype(np.float64) / n
    # room: back wall, floor and side wall meeting in a corner (simple perspective planes)
    base = np.zeros((n, n, 3))
    wall = np.array([0.045, 0.05, 0.075])
    floor = np.array([0.025, 0.028, 0.042])
    side = np.array([0.035, 0.038, 0.058])
    horizon = 0.66 - 0.18 * x                      # floor line rising towards the right
    is_floor = y > horizon
    is_side = (x < 0.22 + 0.25 * (y - 0.15)) & ~is_floor
    base[:] = wall
    base[is_side] = side
    base[is_floor] = floor
    # light: warm source upper-middle, soft falloff (1 - (d/r)^2 like the game) plus bounce glow on the floor
    lx, ly = 0.56, 0.40
    d = np.sqrt((x - lx) ** 2 + (y - ly) ** 2)
    direct = np.clip(1 - (d / 0.48) ** 2, 0, 1) ** 2.2
    warm = np.array([1.00, 0.62, 0.22])
    img = base + direct[..., None] * warm * 0.75
    # bounce: the floor below the light picks it up and lights the walls near it a little
    fd = np.sqrt((x - lx) ** 2 + ((y - 0.80) * 1.8) ** 2)
    bounce = np.clip(1 - (fd / 0.42) ** 2, 0, 1) ** 2 * 0.28
    img += bounce[..., None] * np.array([1.0, 0.62, 0.36])
    # the source itself: a bright core with a hot halo
    core = np.clip(1 - d / 0.085, 0, 1) ** 0.8 * 2.5
    halo = np.exp(-(d / 0.13) ** 2) * 0.9
    img += (core[..., None] * np.array([1.0, 0.97, 0.88]) + halo[..., None] * np.array([1.0, 0.82, 0.5]))
    # filmic-ish tone curve keeps the core white and the room readable
    img = 1 - np.exp(-img * 1.15)
    rgb = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), 'RGB')

    # rays: thin soft spokes around the source
    rays = Image.new('L', (n, n), 0)
    dr = ImageDraw.Draw(rays)
    cx, cy = lx * n, ly * n
    for k in range(8):
        a = np.pi / 8 + k * np.pi / 4
        r0, r1 = 0.11 * n, (0.33 if k % 2 else 0.24) * n
        dr.line([(cx + np.cos(a) * r0, cy + np.sin(a) * r0), (cx + np.cos(a) * r1, cy + np.sin(a) * r1)],
                fill=235, width=int(0.034 * n))
    rays = rays.filter(ImageFilter.GaussianBlur(n * 0.008))
    glow = Image.new('RGB', (n, n), (255, 205, 120))
    rgb = Image.composite(glow, rgb, rays)

    # rounded square with a thin warm rim
    mask = Image.new('L', (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=int(0.22 * n), fill=255)
    out = Image.new('RGBA', (n, n), (0, 0, 0, 0))
    out.paste(rgb, (0, 0), mask)
    rim = Image.new('RGBA', (n, n), (0, 0, 0, 0))
    ImageDraw.Draw(rim).rounded_rectangle([int(0.006 * n)] * 2 + [n - 1 - int(0.006 * n)] * 2, radius=int(0.215 * n),
                                          outline=(255, 190, 120, 70), width=int(0.012 * n))
    return Image.alpha_composite(out, rim)


big = render()
sizes = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
ico = os.path.join(ROOT, 'radiance', 'radiance.ico')
big.resize((256, 256), Image.LANCZOS).save(ico, sizes=[(s, s) for s in sizes])
# preview: the sizes Windows actually shows, on a dark and a light background
prev = Image.new('RGB', (980, 300), (32, 32, 32))
xo = 20
for s in (256, 96, 48, 32, 24, 16):
    im = big.resize((s, s), Image.LANCZOS)
    for row, bg in ((0, (32, 32, 32)), (1, (243, 243, 243))):
        tile = Image.new('RGB', (s, s), bg)
        tile.paste(im, (0, 0), im)
        prev.paste(tile, (xo, 20 + row * (s + 12) if s < 256 else 20 if row == 0 else -1000))
    xo += s + 30
prev.save(os.path.join(ROOT, 'build', 'icon_preview.png'))
print('wrote', ico)
