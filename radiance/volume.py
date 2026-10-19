"""Sparse probe volume: which probes exist, how they are filled and how they are packed for the GPU.

The cell is covered by a regular probe grid. Only "bricks" of BRICK x BRICK x BRICK grid cells that touch
open space are kept. A brick stores (BRICK+1)^3 probes including its shared border, so the GPU can filter
inside one brick without looking at neighbours. Big dungeons keep full detail where the player can go and
store nothing for the rock in between.
"""
import math

import numpy as np
from numba import njit, prange

from .trace import EPS, LIGHT_SKIP, direct, intersect, sample_volume, surface, _rng

BRICK = 4              # grid cells per brick edge
BP = BRICK + 1         # probes per brick edge (shared border)


# ------------------------------------------------------------------------------------------------
# validity: is a grid point in open space? (inside the walls or outside the building it sees back faces / void)
# ------------------------------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def grid_validity(origin, spacing, dims, rays, G, out):
    nx, ny = dims[0], dims[1]
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in prange(out.shape[0]):
        x = i % nx
        y = (i // nx) % ny
        z = i // (nx * ny)
        px = origin[0] + x * spacing
        py = origin[1] + y * spacing
        pz = origin[2] + z * spacing
        bad = 0
        for r in range(rays):
            dz = 1.0 - (2.0 * r + 1.0) / rays
            rr = math.sqrt(max(0.0, 1.0 - dz * dz))
            a = golden * r + 0.37 * (i % 97)
            dx = math.cos(a) * rr
            dy = math.sin(a) * rr
            k, t, u, v = intersect(px, py, pz, dx, dy, dz, 1e7, False, G)
            if k < 0 or surface(G, k, u, v, dx, dy, dz)[6]:
                bad += 1
                if bad * 5 > rays:      # already invalid (> 20 %), stop early
                    break
        out[i] = bad / rays


def active_bricks(valid):
    """bool[bx,by,bz]: bricks whose (BRICK+1)^3 probes include at least one valid probe."""
    dims = np.array(valid.shape)
    bd = np.maximum(1, -(-(dims - 1) // BRICK))
    pad = np.zeros(bd * BRICK + 1, bool)
    pad[:dims[0], :dims[1], :dims[2]] = valid
    act = np.zeros(bd, bool)
    for ox in range(BP):
        for oy in range(BP):
            for oz in range(BP):
                act |= pad[ox:ox + bd[0] * BRICK:BRICK, oy:oy + bd[1] * BRICK:BRICK, oz:oz + bd[2] * BRICK:BRICK]
    return act


def covered_probes(act, dims):
    """Probes that belong to at least one active brick -> int32 index grid (-1 elsewhere) and their count."""
    cov = np.zeros(np.array(act.shape) * BRICK + 1, bool)
    for bx, by, bz in np.argwhere(act):
        cov[bx * BRICK:bx * BRICK + BP, by * BRICK:by * BRICK + BP, bz * BRICK:bz * BRICK + BP] = True
    cov = cov[:dims[0], :dims[1], :dims[2]]
    index = np.full(tuple(dims), -1, np.int32)
    n = int(cov.sum())
    index[cov] = np.arange(n, dtype=np.int32)
    return index, n


# ------------------------------------------------------------------------------------------------
# direct light from fully baked lights (daylight): irradiance a + b.n per probe
# ------------------------------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def probe_direct(points, lights, LG, G, out):
    """out[p, layer, 4, 3] += L1 irradiance of each light at p, shadowed. max(0, n.l) ~= 1/4 + 1/2 n.l"""
    glo, gcell, gdims, goff, gidx = LG
    for p in prange(points.shape[0]):
        px, py, pz = points[p, 0], points[p, 1], points[p, 2]
        cx = int((px - glo[0]) / gcell)
        cy = int((py - glo[1]) / gcell)
        cz = int((pz - glo[2]) / gcell)
        if cx < 0 or cy < 0 or cz < 0 or cx >= gdims[0] or cy >= gdims[1] or cz >= gdims[2]:
            continue
        c = (cz * gdims[1] + cy) * gdims[0] + cx
        for q in range(goff[c], goff[c + 1]):
            i = gidx[q]
            lx = lights[i, 0] - px
            ly = lights[i, 1] - py
            lz = lights[i, 2] - pz
            d2 = lx * lx + ly * ly + lz * lz
            r = lights[i, 6]
            if d2 >= r * r or d2 < 1e-6:
                continue
            d = math.sqrt(d2)
            lx /= d
            ly /= d
            lz /= d
            f = d / r
            att = 1.0 - f * f
            if lights[i, 7] > -1.5:
                ca = -(lx * lights[i, 9] + ly * lights[i, 10] + lz * lights[i, 11])
                if ca <= lights[i, 7]:
                    continue
                if ca < lights[i, 8]:
                    s = (ca - lights[i, 7]) / (lights[i, 8] - lights[i, 7])
                    att *= s * s * (3.0 - 2.0 * s)
            tmax = d - LIGHT_SKIP
            if tmax > EPS:
                hit = intersect(px, py, pz, lx, ly, lz, tmax, True, G, int(lights[i, 14]))
                if hit[0] >= 0:
                    continue
            layer = int(lights[i, 13])
            if layer < 0:
                continue
            for ch in range(3):
                e = lights[i, 3 + ch] * att
                out[p, layer, 0, ch] += 0.25 * e
                out[p, layer, 1, ch] += 0.5 * e * lx
                out[p, layer, 2, ch] += 0.5 * e * ly
                out[p, layer, 3, ch] += 0.5 * e * lz


# ------------------------------------------------------------------------------------------------
# fill probes inside geometry from their open neighbours (so filtering never pulls in black)
# ------------------------------------------------------------------------------------------------
@njit(cache=True)
def dilate(index, coef, valid, iterations):
    n = coef.shape[0]
    have = valid.copy()
    dims = index.shape
    pos = np.empty((n, 3), np.int64)
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                r = index[x, y, z]
                if r >= 0:
                    pos[r, 0] = x
                    pos[r, 1] = y
                    pos[r, 2] = z
    acc = np.zeros(coef.shape[1:])
    for it in range(iterations):
        newly = np.zeros(n, np.bool_)
        for r in range(n):
            if have[r]:
                continue
            acc[:] = 0.0
            cnt = 0
            for a in range(3):
                for s in (-1, 1):
                    q0 = pos[r, 0] + (s if a == 0 else 0)
                    q1 = pos[r, 1] + (s if a == 1 else 0)
                    q2 = pos[r, 2] + (s if a == 2 else 0)
                    if q0 < 0 or q1 < 0 or q2 < 0 or q0 >= dims[0] or q1 >= dims[1] or q2 >= dims[2]:
                        continue
                    o = index[q0, q1, q2]
                    if o >= 0 and have[o]:
                        acc += coef[o]
                        cnt += 1
            if cnt:
                coef[r] = acc / cnt
                newly[r] = True
        if not newly.any():
            break
        have |= newly
    return have


# ------------------------------------------------------------------------------------------------
# surface samples for brightness calibration: first hits of random rays from open probes
# ------------------------------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def surface_samples(points, rays, G, seed, pos, nrm, alb, ok):
    for p in prange(points.shape[0]):
        state = np.empty(1, np.uint64)
        state[0] = np.uint64(seed) * np.uint64(6364136223846793005) + np.uint64(p) * np.uint64(1442695040888963407) + np.uint64(1)
        for _ in range(4):
            _rng(state)
        for r in range(rays):
            j = p * rays + r
            z = 2.0 * _rng(state) - 1.0
            a = 2.0 * math.pi * _rng(state)
            rr = math.sqrt(max(0.0, 1.0 - z * z))
            dx = math.cos(a) * rr
            dy = math.sin(a) * rr
            k, t, u, v = intersect(points[p, 0], points[p, 1], points[p, 2], dx, dy, z, 1e7, False, G)
            if k < 0:
                continue
            ar, ag, ab, nx, ny, nz, back = surface(G, k, u, v, dx, dy, z)
            if back:
                continue
            pos[j, 0] = points[p, 0] + dx * t
            pos[j, 1] = points[p, 1] + dy * t
            pos[j, 2] = points[p, 2] + z * t
            nrm[j, 0] = nx
            nrm[j, 1] = ny
            nrm[j, 2] = nz
            alb[j, 0] = ar
            alb[j, 1] = ag
            alb[j, 2] = ab
            ok[j] = True


@njit(cache=True, parallel=True)
def eval_direct(pos, nrm, lights, LG, G, shadow_mode, out):
    for j in prange(pos.shape[0]):
        tmp = np.zeros((3, 3))
        direct(pos[j, 0], pos[j, 1], pos[j, 2], nrm[j, 0], nrm[j, 1], nrm[j, 2], lights, LG, G, shadow_mode, tmp)
        for ch in range(3):
            out[j, ch] = tmp[0, ch] + tmp[1, ch] + tmp[2, ch]


@njit(cache=True, parallel=True)
def eval_volume(pos, nrm, grid_lo, spacing, dims, index, coef, layer_w, offset, out, ok):
    for j in prange(pos.shape[0]):
        e = np.zeros(3)
        ok[j] = sample_volume(pos[j, 0], pos[j, 1], pos[j, 2], nrm[j, 0], nrm[j, 1], nrm[j, 2], grid_lo, spacing,
                              dims, index, coef, layer_w, offset, e)
        out[j, 0] = e[0]
        out[j, 1] = e[1]
        out[j, 2] = e[2]


@njit(cache=True, parallel=True)
def light_leak(pos, nrm, L, G):
    """How much of a light's energy on nearby surfaces is blocked by geometry -> (total, visible).
    If a fixture loses its shadows in game, the blocked part becomes light leaking through walls."""
    n = pos.shape[0]
    tot = np.zeros(n)
    vis = np.zeros(n)
    for j in prange(n):
        lx = L[0] - pos[j, 0]
        ly = L[1] - pos[j, 1]
        lz = L[2] - pos[j, 2]
        d2 = lx * lx + ly * ly + lz * lz
        if d2 >= L[6] * L[6] or d2 < 1e-6:
            continue
        d = math.sqrt(d2)
        lx /= d
        ly /= d
        lz /= d
        ndl = nrm[j, 0] * lx + nrm[j, 1] * ly + nrm[j, 2] * lz
        if ndl <= 0.0:
            continue
        f = d / L[6]
        w = (1.0 - f * f) * ndl
        tot[j] = w
        tmax = d - LIGHT_SKIP
        if tmax <= EPS:
            vis[j] = w
            continue
        hit = intersect(pos[j, 0] + nrm[j, 0] * EPS, pos[j, 1] + nrm[j, 1] * EPS, pos[j, 2] + nrm[j, 2] * EPS,
                        lx, ly, lz, tmax, True, G, int(L[14]))
        if hit[0] < 0:
            vis[j] = w
    return tot.sum(), vis.sum()


@njit(cache=True, parallel=True)
def floor_distance(points, G, out):
    """Distance straight down to a front-facing surface (inf if none): where a player could stand."""
    for p in prange(points.shape[0]):
        k, t, u, v = intersect(points[p, 0], points[p, 1], points[p, 2], 0.0, 0.0, -1.0, 1e5, False, G)
        if k < 0 or surface(G, k, u, v, 0.0, 0.0, -1.0)[6]:
            out[p] = np.inf
        else:
            out[p] = t


# ------------------------------------------------------------------------------------------------
# GPU packing
# ------------------------------------------------------------------------------------------------
def pack(act, index, coef):
    """-> (indirection uint32[bz,by,bx] with brick id + 1 or 0, atlas float16[layer, rgb, AZ, AY, AX, 4], atlas bricks)."""
    bricks = np.argwhere(act)
    nb = len(bricks)
    ax = max(1, int(math.ceil(nb ** (1 / 3))))
    ay = max(1, int(math.ceil(math.sqrt(nb / ax))))
    az = max(1, int(math.ceil(nb / (ax * ay))))
    nl = coef.shape[1]
    atlas = np.zeros((nl, 3, az * BP, ay * BP, ax * BP, 4), np.float32)
    indirection = np.zeros(act.shape[::-1], np.uint32)          # z, y, x
    dims = index.shape
    for bi, (bx, by, bz) in enumerate(bricks):
        indirection[bz, by, bx] = bi + 1
        cx, cy, cz = bi % ax, (bi // ax) % ay, bi // (ax * ay)
        x0, y0, z0 = bx * BRICK, by * BRICK, bz * BRICK
        sub = index[x0:x0 + BP, y0:y0 + BP, z0:z0 + BP]
        # bricks on the far edge of the grid can be cut off: repeat the last probe
        if sub.shape != (BP, BP, BP):
            sx = np.minimum(np.arange(BP) + x0, dims[0] - 1)
            sy = np.minimum(np.arange(BP) + y0, dims[1] - 1)
            sz = np.minimum(np.arange(BP) + z0, dims[2] - 1)
            sub = index[np.ix_(sx, sy, sz)]
        vals = coef[np.maximum(sub, 0)]                          # BP,BP,BP, layer, 4, rgb
        vals[sub < 0] = 0.0
        # -> layer, rgb, z, y, x, 4
        atlas[:, :, cz * BP:(cz + 1) * BP, cy * BP:(cy + 1) * BP, cx * BP:(cx + 1) * BP, :] = \
            vals.transpose(3, 5, 2, 1, 0, 4)
    return indirection, atlas.astype(np.float16), (ax, ay, az)
