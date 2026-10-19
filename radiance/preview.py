"""Comparison renders and self-validation of a baked cell.

Three ways to light the same view:
  today     - the original setup as the game draws it (all lights with their authored shadows, flat ambient)
  baked     - what the game draws with the bake: live fixture lights after the light patch + baked probes
  reference - the target: the same live lights, the fully baked lights directly with real shadows, and
              path-traced bounce at the same calibration -> differences are the bake's own error
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import trace
from .bake import NORMAL_OFFSET
from .scene import DAYLIGHT, EMITTER, light_arrays

MODES = {'today': 0, 'baked': 1, 'reference': 2, 'probes': 3, 'flat_ambient': 4}


def camera(eye, target, fov_deg=75.0):
    eye = np.asarray(eye, np.float64)
    f = np.asarray(target, np.float64) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.array([*eye, *f, *r, *u, math.tan(math.radians(fov_deg) / 2)], np.float64)


class Previewer:
    def __init__(self, res):
        self.res = res
        sc = res.sc
        lo_b, hi_b = sc.bounds
        pad = 4096

        def pack(classes, shadow, scale=None, live=None, at='game'):
            arr = light_arrays(sc.lights, classes, shadow=shadow, scale=scale, live=live, at=at)
            return arr, trace.light_grid(arr, lo_b - pad, hi_b + pad)

        self.today, self.LG_today = pack(None, 'game')
        self.live, self.LG_live = pack(None, 'patched', live=True)
        # reference: live lights exactly as the game will draw them + the fully baked lights with real shadows
        ref = np.concatenate([light_arrays(sc.lights, None, shadow='patched', live=True),
                              light_arrays(sc.lights, (EMITTER, DAYLIGHT), shadow='all', live=False, at='bake')])
        self.ref, self.LG_ref = ref, trace.light_grid(ref, lo_b - pad, hi_b + pad)
        self.bake, self.LG_bake = pack((EMITTER, DAYLIGHT), 'all', at='bake')
        cl = sc.lighting
        self.dalc = cl.ambient_matrix().astype(np.float64)
        self.dir_col = (cl.directional * cl.dir_fade).astype(np.float64)
        self.dir_vec = cl.ambient_dir().astype(np.float64)
        self.n_today = len(self.today)
        self.n_today_sh = int(self.today[:, 12].sum()) if len(self.today) else 0
        self.n_live = len(self.live)
        self.n_live_sh = int(self.live[:, 12].sum()) if len(self.live) else 0

    def render(self, cam, mode, w=960, h=540, spp=64, bounces=3, layer_w=(1.0, 1.0)):
        r = self.res
        img = np.zeros((h, w, 3))
        trace.render(cam, w, h, MODES[mode], self.today, self.LG_today, self.live, self.LG_live, self.ref,
                     self.LG_ref, self.bake, self.LG_bake, r.G.args(), self.dalc, self.dir_col, self.dir_vec,
                     np.asarray(r.origin, np.float64), float(r.spacing), np.asarray(r.dims, np.int64), r.index,
                     np.ascontiguousarray(r.coef, np.float64), np.asarray(layer_w, np.float64), NORMAL_OFFSET,
                     float(r.floor), float(r.scale), spp, bounces + 1, img)
        return img


def find_viewpoints(pv, count=3, eye=110.0):
    """Open spots at eye height inside the cell, looking along the longest free line."""
    G = pv.res.G.args()
    r = pv.res
    xyz = np.argwhere(r.index >= 0)
    xyz = xyz[r.valid[r.index[xyz[:, 0], xyz[:, 1], xyz[:, 2]]]]       # open space only
    rng = np.random.default_rng(3)
    if len(xyz) > 4000:
        xyz = xyz[rng.choice(len(xyz), 4000, replace=False)]
    cands = []
    for ix, iy, iz in xyz:
        p = r.origin + np.array([ix, iy, iz]) * r.spacing
        k, t, u, v = trace.intersect(p[0], p[1], p[2], 0.0, 0.0, -1.0, 1e5, False, G)
        if k < 0 or t > 400 or trace.surface(G, k, u, v, 0.0, 0.0, -1.0)[6]:
            continue
        cands.append((p[0], p[1], p[2] - t + eye))
    scored = []
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    for c in cands:
        dist = []
        for a in angles:
            dx, dy = math.cos(a), math.sin(a)
            k, t, u, v = trace.intersect(c[0], c[1], c[2], dx, dy, -0.08, 1e5, False, G)
            if k < 0:
                dist.append(0.0)
                continue
            back = trace.surface(G, k, u, v, dx, dy, -0.08)[6]
            dist.append(0.0 if back else min(t, 3000.0))
        dist = np.array(dist)
        if (dist == 0).mean() > 0.1:
            continue
        best = int(np.argmax(np.convolve(np.r_[dist, dist[:3]], np.ones(4) / 4, 'valid')[:32]))
        a_view = angles[(best + 1) % 32]
        score = float(np.median(dist))
        # prefer looking at the strongest light source in reach: that is where the lighting happens
        lights = pv.today
        if len(lights):
            d = lights[:, :3] - np.array(c)
            dist_l = np.linalg.norm(d, axis=1)
            power = lights[:, 3:6].sum(1) * np.clip(1 - dist_l / np.maximum(lights[:, 6], 1), 0, 1)
            j = int(np.argmax(power))
            if power[j] > 0 and dist_l[j] > 150:
                a_view = math.atan2(d[j, 1], d[j, 0])
                score += 2000.0 * float(min(power[j], 1.0))
        scored.append((score, c, a_view))
    scored.sort(key=lambda x: -x[0])
    views = []
    for _, c, a in scored:
        if any(np.linalg.norm(np.array(c) - np.array(v[0])) < 500 for v in views):
            continue
        tgt = (c[0] + math.cos(a) * 400, c[1] + math.sin(a) * 400, c[2] - 60)
        eye_p = c
        if _sees_backfaces(G, eye_p, tgt) > 0.1:      # camera would stand behind a wall
            continue
        views.append((eye_p, tgt))
        if len(views) >= count:
            break
    return views


def _sees_backfaces(G, eye, tgt, n=9):
    """Share of a coarse ray grid over the view that hits back faces or nothing."""
    cam = camera(eye, tgt, 80)
    bad = 0
    for sx in np.linspace(-0.8, 0.8, n):
        for sy in np.linspace(-0.4, 0.4, n):
            d = cam[3:6] + cam[6:9] * sx * cam[12] + cam[9:12] * sy * cam[12]
            d /= np.linalg.norm(d)
            k, t, u, v = trace.intersect(eye[0], eye[1], eye[2], d[0], d[1], d[2], 1e5, False, G)
            if k < 0 or trace.surface(G, k, u, v, d[0], d[1], d[2])[6]:
                bad += 1
    return bad / (n * n)


def to_image(img, exposure=1.0):
    a = np.clip(img * exposure, 0, 1)
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8))


def labeled(img, text):
    im = img.copy()
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('segoeui.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    x0, y0, x1, y1 = d.textbbox((10, 8), text, font=font)
    d.rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), fill=(0, 0, 0))
    d.text((10, 8), text, fill=(255, 255, 255), font=font)
    return im


def grid(images, cols=2):
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    out = Image.new('RGB', (w * cols, h * rows), (0, 0, 0))
    for i, im in enumerate(images):
        out.paste(im, ((i % cols) * w, (i // cols) * h))
    return out


def _views(pv, views):
    if isinstance(views, list):
        return views                     # fixed cameras (to compare variants of the same cell)
    vps = find_viewpoints(pv, views)
    if not vps:
        lo_b, hi_b = pv.res.sc.bounds
        c = (lo_b + hi_b) / 2
        vps = [((c[0], c[1], lo_b[2] + 120), (c[0] + 400, c[1], lo_b[2] + 60))]
    return vps


def compare_report(res, path, log=print, w=640, h=360, views=2, spp=96):
    """Side-by-side: today | baked (what the game draws with the bake) | path-traced reference.
    Each row gets one exposure (like the game's eye adaptation), the same for all three panels."""
    pv = Previewer(res)
    rows = []
    wts = np.array([0.2126, 0.7152, 0.0722])
    for i, (eye, tgt) in enumerate(_views(pv, views)):
        cam = camera(eye, tgt, 80)
        today = pv.render(cam, 'today', w, h)
        baked = pv.render(cam, 'baked', w, h)
        ref = pv.render(cam, 'reference', w, h, spp=spp)
        exp = float(np.clip(0.18 / max((today @ wts).mean(), 1e-4), 1.0, 8.0))
        tag = f'  [exposure x{exp:.1f}]' if exp > 1.05 else ''
        rows.append(labeled(to_image(today, exp),
                            f'Today: {pv.n_today} live lights ({pv.n_today_sh} shadowed) + flat ambient{tag}'))
        rows.append(labeled(to_image(baked, exp), f'Baked: {pv.n_live} live lights ({pv.n_live_sh} shadowed) + baked GI'))
        rows.append(labeled(to_image(ref, exp), 'Reference: path traced'))
        log(f'  preview view {i + 1} rendered')
    img = grid(rows, 3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return path


def _down(img, f=4):
    """Block-average: compares lighting, not path-tracing noise."""
    h, w = img.shape[0] // f * f, img.shape[1] // f * f
    return img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def validate(res, views=2, w=240, h=136, spp=160):
    """Luminance error against the path-traced reference: baked (what ships) vs today (original look)."""
    pv = Previewer(res)
    wts = np.array([0.2126, 0.7152, 0.0722])
    out = []
    for eye, tgt in _views(pv, views):
        cam = camera(eye, tgt, 80)
        ref = _down(np.clip(pv.render(cam, 'reference', w, h, spp=spp), 0, 1) @ wts)
        bak = _down(np.clip(pv.render(cam, 'baked', w, h), 0, 1) @ wts)
        tod = _down(np.clip(pv.render(cam, 'today', w, h), 0, 1) @ wts)
        if ref.mean() < 0.002:         # unlit view: nothing to compare
            continue
        m = ref.mean()
        out.append({'baked_vs_today': float(bak.mean() / max(tod.mean(), 1e-4)),
                    'baked_err': float(np.abs(bak - ref).mean() / m), 'today_err': float(np.abs(tod - ref).mean() / m),
                    'baked_brightness': float(bak.mean() / m), 'today_brightness': float(tod.mean() / m)})
    if not out:
        return None
    return {k: round(float(np.mean([o[k] for o in out])), 3) for k in out[0]}
