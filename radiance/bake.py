"""Bake one interior cell: scene -> sparse probe volume -> .rdp + report.

Light roles (decided per light by the scene builder, see scene.classify):
  emitter  - fires, candles, torches: stay live in game; their bounce is baked (layer 0)
  daylight - window / sun lights: fully baked, direct and bounce with real shadows (layer 1); off in game
  fill     - fake bounce lights: not baked, off in game; the computed bounce replaces them
Brightness is calibrated per cell: the baked ambient carries the same average light on the cell's surfaces
as the original setup (flat ambient + fill lights + live daylight), just distributed physically.
"""
import json
import os
import time

import numpy as np

from . import probefile, trace, volume
from .plugins import fk_str
from .scene import DAYLIGHT, EMITTER, FILL, LIGHT_FLAG_SHADOW, NothingToBake, build_scene, light_arrays, load_rules, quick_fingerprint

DEFAULTS = {
    'spacing': 0,            # game units between probes; 0 = auto: 64 (~0.9 m), 96 for very large cells
    'rays': 512,             # rays per probe
    'bounces': 6,            # light bounces after the first hit (paths end early once they carry little light)
    'max_grid': 6_000_000,   # dense grid points before the spacing grows (only bricks with open space are stored)
    'invalid_backface': 0.2,
    'mode': 'quality',       # 'performance': fewer shadow-casting lights, accepts small light leaks and blur
    'relocate': True,        # move probes stuck in walls to the nearest open spot and bake them there
}
MODES = {   # (leak share above which a fixture keeps its shadow, max error for baking a shadowed key light)
    'quality': (0.25, 0.30),
    'performance': (2.0, 0.60),
}
AMBIENT_FLOOR = 0.1          # minimum share of the original flat ambient kept in game (raised per cell if needed)
CALIBRATION_BIAS = 0.95       # measured: bakes came out ~5 % brighter than the original in rendered views
                             # (median of 11 cells): the calibration samples see a slightly different mix
AMBIENT_MAX = 1.0            # most of the original flat ambient a cell may keep (more lifts the dark areas)
NORMAL_OFFSET = 24.0         # Radiance.ini NormalOffset
# Bounce boost at most 2x: boosting it further to reach the lighting mod's brightness over-brightened what is near
# the light sources (fires). Measured on 10 cells against the physical result at equal brightness: distribution error
# median 28.5 % (cap 16) -> 24.1 % (cap 2); the rest of the brightness comes from the original flat ambient instead
# (up to AMBIENT_MAX), and only past that the boost goes higher. A per-cell choice on the calibration samples disagreed with the rendered views: not used.
CALIBRATION_RANGE = (0.25, 2.0)
BOOST_LIMIT = 16.0           # boost beyond 2x only when even AMBIENT_MAX of the flat ambient stays too dark
# (spacing, max bricks): measured, 64 cuts the error ~25% in rooms; big dungeons fall back to 96, whole cities
# and glacier caves built as one interior to 128 / 192 so a cell stays under ~25 MB of VRAM and ~8 MB on disk
AUTO_SPACING = ((64.0, 4000), (96.0, 8000), (128.0, 8000), (192.0, None))
LEAK_KEEP_SHADOW = 0.25   # keep a fixture's shadow when more of its light than this would leak through walls
BAKE_KEY_LIGHT_ERROR = 0.3  # a shadowed key light is baked (and turned off) when probes reproduce it this well

TOOL_VERSION = '0.5.0-beta'


def settings_sig(st, rules):
    """What makes an old bake stale besides the game data: tool version and light rules.
    Quality settings (rays, spacing, bounces) are left out on purpose - after changing them, use Bake all."""
    return json.dumps([TOOL_VERSION, st.get('mode', 'quality'), {k: rules[k] for k in sorted(rules)}],
                      sort_keys=True, default=str)


def probe_path(out_root, key):
    return os.path.join(out_root, 'Radiance', 'Probes', key[0], f'{key[1]:06X}.rdp')


def grid_for(bounds, spacing, max_points):
    lo, hi = np.asarray(bounds[0], np.float64), np.asarray(bounds[1], np.float64)
    while True:
        dims = np.maximum(2, np.ceil((hi - lo) / spacing).astype(int) + 1)
        if dims.prod() <= max_points:
            break
        spacing *= 1.15
    center = (lo + hi) / 2
    origin = center - (dims - 1) * spacing / 2
    return origin.astype(np.float64), float(spacing), dims


def robust_bounds(verts):
    """Geometry bounds without far-away strays (skyboxes, fake exteriors placed kilometres away)."""
    lo = np.percentile(verts, 0.2, axis=0)
    hi = np.percentile(verts, 99.8, axis=0)
    pad = (hi - lo) * 0.02 + 32
    return np.maximum(verts.min(0), lo - pad), np.minimum(verts.max(0), hi + pad)


def sh_to_coef(sh):
    """Radiance SH [.., layer, 4(Y00,Y1-1,Y10,Y11), 3] -> irradiance (game units) a + b.n  [.., layer, 4(a,bx,by,bz), 3]."""
    out = np.empty_like(sh)
    # E(n)/pi with cosine lobe convolution: A0 = pi, A1 = 2pi/3
    out[..., 0, :] = sh[..., 0, :] * trace.SH_Y0
    k1 = (2.0 / 3.0) * trace.SH_Y1
    out[..., 1, :] = sh[..., 3, :] * k1
    out[..., 2, :] = sh[..., 1, :] * k1
    out[..., 3, :] = sh[..., 2, :] * k1
    return out


def lum(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def relocate(points, valid, spacing, gargs, threshold):
    """Probes inside walls: find the nearest open spot within half a grid step and bake there instead
    (their value then comes from the room they face, not an average of both sides of the wall)."""
    dirs = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)],
                    float)
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    out = points.copy()
    todo = np.nonzero(~valid)[0]
    moved = 0
    for frac in (0.2, 0.35, 0.49):
        if not len(todo):
            break
        cand = (points[todo][:, None, :] + dirs[None] * spacing * frac).reshape(-1, 3)
        back = np.zeros(len(cand))
        trace.probe_validity(np.ascontiguousarray(cand), 32, gargs, back, threshold)
        back = back.reshape(len(todo), len(dirs))
        best = back.argmin(1)
        ok = back[np.arange(len(todo)), best] < threshold
        out[todo[ok]] = points[todo[ok]] + dirs[best[ok]] * spacing * frac
        moved += int(ok.sum())
        todo = todo[~ok]
    return out, moved


def relocate_lights(sc, gargs, log):
    """Lights whose centre is inside geometry (or outside the walls) send nothing in a shadowed bake, while in game
    an unshadowed light shines through the wall anyway (e.g. Lux's window lights set into thick walls and roofs).
    Move each such light to the nearest open spot so its light enters the room, as the author intended.
    Shadowed live lights are left alone: in game their own shadow map blocks them the same way."""
    cand_l = [L for L in sc.lights if not (L.live and L.flags & LIGHT_FLAG_SHADOW)]
    if not cand_l:
        return
    pts = np.array([L.pos for L in cand_l], np.float64)
    own = np.array([L.owner for L in cand_l], np.int64)
    back = np.zeros(len(cand_l))
    trace.light_validity(pts, own, 64, gargs, back)
    n = 48
    k = np.arange(n) + 0.5
    zz = 1.0 - 2.0 * k / n
    rr = np.sqrt(1.0 - zz * zz)
    ang = np.pi * (3.0 - np.sqrt(5.0)) * k
    dirs = np.stack([np.cos(ang) * rr, np.sin(ang) * rr, zz], 1)
    moved = []
    for L, b, p, o in zip(cand_l, back, pts, own):
        if b <= 0.5:
            continue
        dists = [d for d in (16.0, 32.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0) if d <= max(32.0, 0.6 * L.radius)]
        cand = (p[None, None, :] + np.asarray(dists)[:, None, None] * dirs[None]).reshape(-1, 3)
        cb = np.zeros(len(cand))
        trace.light_validity(np.ascontiguousarray(cand), np.full(len(cand), o, np.int64), 32, gargs, cb)
        cb = cb.reshape(len(dists), n)
        for di in range(len(dists)):
            j = int(cb[di].argmin())
            if cb[di, j] < 0.2:
                L.bake_pos = (p + dists[di] * dirs[j]).astype(np.float32)
                moved.append(dists[di])
                break
    if moved:
        log(f'  {len(moved)} light(s) sit inside walls: baked from the nearest open spot '
            f'(median {np.median(moved):.0f} units away)')


def surface_points(G, valid_pts, count=3000, rays=6, seed=11):
    """Random visible surface points (position, normal, albedo) seen from where a player can be: open probes
    at most eye height above a floor. Calibrating on these keeps the brightness the player actually sees."""
    rng = np.random.default_rng(seed)
    fd = np.zeros(len(valid_pts))
    volume.floor_distance(np.ascontiguousarray(valid_pts), G.args(), fd)
    near = valid_pts[fd <= 200.0]
    if len(near) >= 50:
        valid_pts = near
    pts = valid_pts if len(valid_pts) <= count else valid_pts[rng.choice(len(valid_pts), count, replace=False)]
    n = len(pts) * rays
    pos = np.zeros((n, 3))
    nrm = np.zeros((n, 3))
    alb = np.zeros((n, 3))
    ok = np.zeros(n, np.bool_)
    volume.surface_samples(np.ascontiguousarray(pts), rays, G.args(), seed, pos, nrm, alb, ok)
    return pos[ok], nrm[ok], alb[ok]


def decide_shadows(sc, G, pos, nrm, log, leak_keep=LEAK_KEEP_SHADOW):
    """Fixture lights with omni/hemisphere shadows lose them in the light patch (big VR cost) - unless that would
    let a noticeable part of their light through walls. Spotlights always keep theirs."""
    kept = removed = 0
    for L in sc.lights:
        if not L.live or not L.flags & (0x800 | 0x1000) or L.flags & 0x400:
            continue
        # a light the game switches is not a bake source, but its shadow still needs the leak test
        row = light_arrays([L], None, shadow='all', at='game' if L.switched else 'bake')[0].astype(np.float64)
        tot, vis = volume.light_leak(pos, nrm, row, G.args())
        leak = 1.0 - vis / tot if tot > 1e-6 else 0.0
        L.keep_shadow = leak > leak_keep
        kept += L.keep_shadow
        removed += not L.keep_shadow
    if kept or removed:
        log(f'  fixture shadows: {removed} removed, {kept} kept (would leak through walls)')


def probe_error(row, LGl, samples, ref, points, valid, index, origin, spacing, dims, G):
    """Relative error of one light baked into the probes against its exact light `ref` on the samples."""
    pos, nrm, alb = samples
    open_pts = np.ascontiguousarray(points[valid])
    tmp = np.zeros((len(open_pts), 2, 4, 3))
    volume.probe_direct(open_pts, row, LGl, G.args(), tmp)
    coef = np.zeros((len(points), 2, 4, 3))
    coef[valid] = tmp
    filled = volume.dilate(index, coef, valid, 16)
    coef[~filled] = 0.0
    approx = np.zeros_like(pos)
    ok = np.zeros(len(pos), np.bool_)
    volume.eval_volume(pos, nrm, origin, spacing, dims, index, coef, np.ones(2), NORMAL_OFFSET, approx, ok)
    return float(np.abs(lum(alb * approx) - ref)[ok].sum() / max(ref[ok].sum(), 1e-9))


def decide_daylight(sc, G, points, valid, index, origin, spacing, dims, samples, log):
    """Unshadowed window / sun / large lights: keep them live exactly as the lighting mod placed them, or bake
    them (real shadows, no light through walls, but soft)? Measured per light on the surfaces a player sees,
    against the exact shadowed light: live is off by the light that leaks where it can't reach, baked by the
    blur of the probes. The smaller error wins. Live ones still get their bounce baked."""
    pos, nrm, alb = samples
    if len(pos) < 50:
        return
    lo_b, hi_b = sc.bounds
    live_n = baked_n = 0
    for L in sc.lights:
        if L.cls != DAYLIGHT or L.live or L.switched:
            continue
        row = light_arrays([L], None, shadow='all', at='bake')
        row[:, 13] = 0.0
        LGl = trace.light_grid(row, lo_b - 4096, hi_b + 4096)
        exact = np.zeros_like(pos)
        volume.eval_direct(pos, nrm, row, LGl, G.args(), 1, exact)
        ref = lum(alb * exact)
        if ref.sum() < 1e-6:
            continue                      # reaches nothing a player sees: baking it costs nothing
        game = light_arrays([L], None, shadow='game')
        game[:, 13] = 0.0
        as_live = np.zeros_like(pos)
        volume.eval_direct(pos, nrm, game, trace.light_grid(game, lo_b - 4096, hi_b + 4096), G.args(), 0, as_live)
        err_live = float(np.abs(lum(alb * as_live) - ref).sum() / ref.sum())
        err_bake = probe_error(row, LGl, samples, ref, points, valid, index, origin, spacing, dims, G)
        if err_live < err_bake:
            L.live = True
            L.reason += f' -> live ({err_live * 100:.0f}% error live, {err_bake * 100:.0f}% baked)'
            live_n += 1
        else:
            L.reason += f' -> baked ({err_bake * 100:.0f}% error baked, {err_live * 100:.0f}% live)'
            baked_n += 1
    if live_n or baked_n:
        log(f'  window / large lights: {baked_n} baked (they would leak through walls), {live_n} kept live '
            '(sharper than the probes, little leak)')


def decide_key_lights(sc, G, points, valid, index, origin, spacing, dims, samples, log,
                      max_error=BAKE_KEY_LIGHT_ERROR):
    """Shadow-casting key lights (spots, suns - not flickering fixtures) cost a shadow map each in game.
    Bake one into the probes and compare with the exact shadowed light on the surfaces a player sees: if the
    probes reproduce it closely (soft pools of light), bake it and turn it off; if not (sharp beams), keep it."""
    pos, nrm, alb = samples
    if len(pos) < 50:
        return
    lo_b, hi_b = sc.bounds
    baked = kept = 0
    for L in sc.lights:
        if not (L.live and L.flags & 0x1C00) or not L.reason.startswith(('authored shadows', 'shadowed')):
            continue
        if L.switched:
            continue                      # switched by the game: never baked
        row = light_arrays([L], None, shadow='all', at='bake')
        row[:, 13] = 0.0
        LGl = trace.light_grid(row, lo_b - 4096, hi_b + 4096)
        exact = np.zeros_like(pos)
        volume.eval_direct(pos, nrm, row, LGl, G.args(), 1, exact)
        ref = lum(alb * exact)
        if ref.sum() < 1e-6:
            continue
        err = probe_error(row, LGl, samples, ref, points, valid, index, origin, spacing, dims, G)
        if err < max_error:
            L.live = False
            L.reason += f' -> baked ({err * 100:.0f}% error)'
            baked += 1
        else:
            L.reason += f' -> live ({err * 100:.0f}% error if baked)'
            kept += 1
    if baked or kept:
        log(f'  shadowed key lights: {baked} baked and turned off, {kept} kept live (too sharp to bake)')


def calibrate(sc, origin, spacing, dims, index, bounce, day_direct, samples, G, log):
    """Scale for the baked bounce so the cell keeps its original average brightness on surfaces.
    Original = flat ambient + fill lights + live daylight as the game draws them; baked = bounce * s + baked
    daylight (physical, not scaled) + the kept share of the flat ambient."""
    pos, nrm, alb = samples
    if len(pos) < 50:
        return 1.0, 1.0, {}          # cannot calibrate: keep the original look
    lo_b, hi_b = sc.bounds
    m = sc.lighting.ambient_matrix().astype(np.float64)
    amb = np.maximum(0.0, nrm @ m[:, :3].T + m[:, 3])
    # the replaced lights count only where they really reach: an unshadowed light also brightens surfaces behind
    # walls, and matching that leaked brightness made rooms too bright and too flat
    orig_l = light_arrays(sc.lights, None, shadow='all', live=False, at='bake')
    orig = np.zeros_like(pos)
    if len(orig_l):
        volume.eval_direct(pos, nrm, orig_l, trace.light_grid(orig_l, lo_b - 4096, hi_b + 4096), G.args(), 0, orig)
    B = np.zeros_like(pos)
    D = np.zeros_like(pos)
    okb = np.zeros(len(pos), np.bool_)
    okd = np.zeros(len(pos), np.bool_)
    volume.eval_volume(pos, nrm, origin, spacing, dims, index, bounce, np.ones(2), NORMAL_OFFSET, B, okb)
    volume.eval_volume(pos, nrm, origin, spacing, dims, index, day_direct, np.ones(2), NORMAL_OFFSET, D, okd)
    sel = okb
    if sel.sum() < 50:
        return 1.0, 1.0, {}          # cannot calibrate: keep the original look
    target = lum(alb[sel] * (amb[sel] + orig[sel])).mean() * CALIBRATION_BIAS
    floor = lum(alb[sel] * amb[sel]).mean() * AMBIENT_FLOOR
    direct_ = lum(alb[sel] * D[sel]).mean()
    have = lum(alb[sel] * B[sel]).mean()
    stats = {'target': round(float(target), 4), 'bounce': round(float(have), 4), 'daylight': round(float(direct_), 4),
             'floor': round(float(floor), 4), 'samples': int(sel.sum())}
    amb_l = lum(alb[sel] * amb[sel]).mean()
    s = float(np.clip((target - floor - direct_) / have, *CALIBRATION_RANGE)) if have > 1e-6 else 1.0
    # bounce alone cannot reach the original brightness (few real light sources, very bright authored ambient):
    # keep more of the original flat ambient instead (up to AMBIENT_MAX) - never darker than the author intended
    f = AMBIENT_FLOOR
    rest = target - direct_ - s * have
    if rest > floor and amb_l > 1e-6:
        f = float(np.clip(rest / amb_l, AMBIENT_FLOOR, AMBIENT_MAX))
        rest = target - direct_ - s * have - f * amb_l
        if rest > 0.01 * target and have > 1e-6:
            # still darker than intended with the most ambient we allow: only now boost the bounce further
            s = float(min(s + rest / have, BOOST_LIMIT))
    stats['scale'] = round(s, 3)
    stats['floor_share'] = round(f, 3)
    log(f'  brightness: original {target:.3f} = baked daylight {direct_:.3f} + bounce {have:.3f} x {s:.2f} '
        f'+ {f * 100:.0f}% of the original flat ambient')
    return s, f, stats


class BakeResult:
    """Everything previews and validation need, in memory."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def bake_cell(lo, key, out_root, settings=None, rules=None, log=print, report_dir=None, progress=None):
    st = dict(DEFAULTS)
    st.update(settings or {})
    t0 = time.time()
    rules = rules or load_rules(None)
    sc = build_scene(lo, key, rules, log)
    name = sc.cell['name'] or sc.cell['edid']
    log(f'[{fk_str(key)}] {name}: {len(sc.tris):,} triangles, {len(sc.lights)} lights, '
        f'{sc.textures.shape[0]} textures ({time.time() - t0:.1f}s)')
    n_live = sum(L.live for L in sc.lights)
    n_static = sum(1 for L in sc.lights if L.cls == DAYLIGHT and not L.live)
    n_fill = sum(1 for L in sc.lights if L.cls == FILL)
    n_sw = sum(L.switched for L in sc.lights)
    log(f'  lights: {n_live} live, {n_static} window / large (live or baked: measured below), '
        f'{n_fill} fill (replaced by baked bounce)' + (f', {n_sw} switched by the game (live, not baked)' if n_sw else ''))

    t1 = time.time()
    G = trace.Geometry(sc)
    gargs = G.args()
    lo_b, hi_b = robust_bounds(sc.verts)
    relocate_lights(sc, gargs, log)
    bake_l = light_arrays(sc.lights, (EMITTER, DAYLIGHT), shadow='all', at='bake')
    LG = trace.light_grid(bake_l, lo_b - 4096, hi_b + 4096)
    log(f'  BVH {G.nnodes:,} nodes ({time.time() - t1:.1f}s)')

    # 1. which grid points are in open space (auto spacing: fine grid unless the cell is huge)
    t2 = time.time()
    tries = AUTO_SPACING if not st['spacing'] else ((float(st['spacing']), None),)
    prev = None
    for ti, (want, max_bricks) in enumerate(tries):
        if prev and max_bricks is not None and ti + 1 < len(tries) and                 prev[1] * (prev[0] / want) ** 3 > 2 * max_bricks:
            continue        # brick count falls ~ spacing^3: this step can't fit either, skip its validity pass
        origin, spacing, dims = grid_for((lo_b, hi_b), want, st['max_grid'])
        back = np.zeros(int(dims.prod()))
        volume.grid_validity(origin, spacing, dims.astype(np.int64), 32, gargs, back)
        valid_grid = (back < st['invalid_backface']).reshape(tuple(dims[::-1])).transpose(2, 1, 0)
        act = volume.active_bricks(valid_grid)
        if max_bricks is None or act.sum() <= max_bricks:
            break
        prev = (spacing, int(act.sum()))
    log(f'  grid {dims[0]}x{dims[1]}x{dims[2]} @ {spacing:.0f}u')
    index, n = volume.covered_probes(act, dims)
    if n == 0:
        raise NothingToBake('no open space inside the cell (unfinished, test or sealed cell)')
    xyz = np.argwhere(index >= 0)
    rows = index[xyz[:, 0], xyz[:, 1], xyz[:, 2]]
    points = np.zeros((n, 3))
    points[rows] = origin + xyz * spacing
    valid = np.zeros(n, np.bool_)
    valid[rows] = valid_grid[xyz[:, 0], xyz[:, 1], xyz[:, 2]]
    log(f'  open space: {valid.sum():,} probes in {int(act.sum()):,} bricks ({time.time() - t2:.1f}s)')
    bake_pts = points
    if st.get('relocate'):
        # every wall probe: limiting this to probes next to open space measured clearly worse (6.5 -> 8.9 %)
        bake_pts, moved = relocate(points, valid, spacing, gargs, st['invalid_backface'])
        log(f'  relocated {moved:,} probes out of walls')
        valid = valid | (np.linalg.norm(bake_pts - points, axis=1) > 0)

    # which lights stay live: fixture shadows that can go (leak test), key lights that bake well
    samples = surface_points(G, bake_pts[valid])
    leak_keep, max_error = MODES.get(st['mode'], MODES['quality'])
    decide_shadows(sc, G, samples[0], samples[1], log, leak_keep)
    decide_key_lights(sc, G, bake_pts, valid, index, origin, spacing, dims, samples, log, max_error)
    decide_daylight(sc, G, bake_pts, valid, index, origin, spacing, dims, samples, log)
    day_l = light_arrays(sc.lights, (EMITTER, DAYLIGHT), shadow='all', live=False, at='bake')   # fully baked lights
    LGd = trace.light_grid(day_l, lo_b - 4096, hi_b + 4096)

    # 2. light the open probes: bounce (path traced) + direct from fully baked daylight
    t3 = time.time()
    nl = 2
    todo = np.nonzero(valid)[0]
    sh = np.zeros((n, nl, 4, 3))
    chunk = 2048
    for s in range(0, len(todo), chunk):
        sel = todo[s:s + chunk]
        sh_c = np.zeros((len(sel), nl, 4, 3))
        st_c = np.zeros((len(sel), 2))
        trace.bake_probes(bake_pts[sel], st['rays'], st['bounces'] + 1, nl, bake_l, LG, gargs, s + 1, sh_c, st_c)
        sh[sel] = sh_c
        if progress:
            progress(min(len(todo), s + chunk) / max(1, len(todo)))
    bounce = sh_to_coef(sh)
    day_direct = np.zeros((n, nl, 4, 3))
    if len(day_l):
        tmp = np.zeros((len(todo), nl, 4, 3))
        volume.probe_direct(np.ascontiguousarray(bake_pts[todo]), day_l, LGd, gargs, tmp)
        day_direct[todo] = tmp
    # 3. probes inside walls take their open neighbours' values (bricks never hold black)
    both = np.concatenate([bounce, day_direct], axis=1)
    filled = volume.dilate(index, both, valid, 16)
    both[~filled] = 0.0
    bounce, day_direct = both[:, :nl], both[:, nl:]
    log(f'  lit {len(todo):,} probes ({time.time() - t3:.1f}s)')

    # 4. surface samples -> which fixture shadows can go, brightness calibration; final layers:
    #    0 = fixtures bounce (+ baked key lights), 1 = static/daylight direct + bounce
    scale, floor_share, cal = calibrate(sc, origin, spacing, dims, index, bounce, day_direct, samples, G, log)
    coef = bounce * scale + day_direct           # baked lights' direct light stays physical

    # 5. pack and write
    indirection, atlas, abricks = volume.pack(act, index, coef.astype(np.float32))
    fp = quick_fingerprint(lo, key, settings_sig(st, rules))
    path = probe_path(out_root, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    probefile.write(path, key, origin, spacing, dims, fp, st['rays'], st['bounces'], volume.BRICK, indirection,
                    atlas, abricks, scale, floor_share)
    info = {
        'cell': fk_str(key), 'edid': sc.cell['edid'], 'name': sc.cell['name'], 'plugin': sc.cell['plugin'],
        'fingerprint': fp, 'file': os.path.relpath(path, out_root), 'bytes': os.path.getsize(path),
        'triangles': int(len(sc.tris)), 'grid': {'origin': [float(x) for x in origin], 'spacing': spacing,
                                                 'dims': [int(x) for x in dims]},
        'bricks': int(act.sum()), 'open_probes': int(valid.sum()), 'calibration': cal,
        'seconds': round(time.time() - t0, 1), 'settings': st,
        'lights': [L.to_dict() for L in sc.lights], 'cell_lighting': sc.lighting.to_dict(),
        'skipped_refs': sc.skipped, 'missing_textures': sc.texture_missing[:50], 'log': sc.log[:50],
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, f'{key[0]}_{key[1]:06X}.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=1)
    log(f'  wrote {info["file"]} ({info["bytes"] / 1024:.0f} KB, {info["bricks"]} bricks) in {info["seconds"]}s')
    res = BakeResult(sc=sc, G=G, origin=origin, spacing=spacing, dims=dims, index=index, coef=coef, scale=scale,
                     floor=floor_share, valid=valid)
    return info, res
