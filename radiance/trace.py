"""CPU ray tracer (numba): BVH, Skyrim light model, probe gather and camera renders.

Light units follow the game shader, not physics: a surface's lit color is albedo * (sum of lights + ambient),
each light contributing color * fade * (1 - (d / radius)^2) * max(N.L, 0). Indirect light is gathered in the
same units, so the baked ambient can replace the cell ambient term one to one.
"""
import math

import numpy as np
from numba import njit, prange

LEAF = 4
STACK = 64
EPS = 0.05
LIGHT_SKIP = 8.0      # shadow rays ignore the last few units before a light (lights sit inside fixtures)
SH_Y0 = 0.282095
SH_Y1 = 0.488603


# ------------------------------------------------------------------------------------------------
# BVH (binned SAH)
# ------------------------------------------------------------------------------------------------
@njit(cache=True)
def _build(bmin, bmax, cent, idx, nmin, nmax, left, start, count):
    n = idx.shape[0]
    stack = np.empty((128, 3), np.int64)   # node, start, end
    nodes = 1
    sp = 0
    stack[0, 0] = 0
    stack[0, 1] = 0
    stack[0, 2] = n
    sp = 1
    NB = 16
    bin_cnt = np.zeros(NB, np.int64)
    bin_min = np.zeros((NB, 3))
    bin_max = np.zeros((NB, 3))
    rc_s = np.zeros(NB + 1, np.int64)
    rlo_s = np.zeros((NB + 1, 3))
    rhi_s = np.zeros((NB + 1, 3))
    llo = np.zeros(3)
    lhi = np.zeros(3)
    lo0 = np.zeros(3)
    hi0 = np.zeros(3)
    clo = np.zeros(3)
    chi = np.zeros(3)
    while sp > 0:
        sp -= 1
        node = stack[sp, 0]
        s = stack[sp, 1]
        e = stack[sp, 2]
        for a in range(3):
            lo0[a] = np.inf
            hi0[a] = -np.inf
            clo[a] = np.inf
            chi[a] = -np.inf
        for k in range(s, e):
            t = idx[k]
            for a in range(3):
                lo0[a] = min(lo0[a], bmin[t, a])
                hi0[a] = max(hi0[a], bmax[t, a])
                clo[a] = min(clo[a], cent[t, a])
                chi[a] = max(chi[a], cent[t, a])
        nmin[node, :] = lo0
        nmax[node, :] = hi0
        cnt = e - s
        if cnt <= LEAF:
            left[node] = -1
            start[node] = s
            count[node] = cnt
            continue
        best_cost = np.inf
        best_axis = -1
        best_split = 0
        for a in range(3):
            ext = chi[a] - clo[a]
            if ext <= 1e-6:
                continue
            bin_cnt[:] = 0
            bin_min[:] = np.inf
            bin_max[:] = -np.inf
            for k in range(s, e):
                t = idx[k]
                b = int(NB * (cent[t, a] - clo[a]) / ext)
                if b >= NB:
                    b = NB - 1
                bin_cnt[b] += 1
                for q in range(3):
                    bin_min[b, q] = min(bin_min[b, q], bmin[t, q])
                    bin_max[b, q] = max(bin_max[b, q], bmax[t, q])
            # suffix sweep (bins >= i), then prefix sweep while scanning the split planes
            rc_s[NB] = 0
            for q in range(3):
                rlo_s[NB, q] = np.inf
                rhi_s[NB, q] = -np.inf
            for bb in range(NB - 1, -1, -1):
                rc_s[bb] = rc_s[bb + 1] + bin_cnt[bb]
                for q in range(3):
                    rlo_s[bb, q] = min(rlo_s[bb + 1, q], bin_min[bb, q])
                    rhi_s[bb, q] = max(rhi_s[bb + 1, q], bin_max[bb, q])
            lc = 0
            for q in range(3):
                llo[q] = np.inf
                lhi[q] = -np.inf
            for sp_i in range(1, NB):
                bb = sp_i - 1
                lc += bin_cnt[bb]
                for q in range(3):
                    llo[q] = min(llo[q], bin_min[bb, q])
                    lhi[q] = max(lhi[q], bin_max[bb, q])
                rc = rc_s[sp_i]
                if lc == 0 or rc == 0:
                    continue
                dl0 = lhi[0] - llo[0]
                dl1 = lhi[1] - llo[1]
                dl2 = lhi[2] - llo[2]
                dr0 = rhi_s[sp_i, 0] - rlo_s[sp_i, 0]
                dr1 = rhi_s[sp_i, 1] - rlo_s[sp_i, 1]
                dr2 = rhi_s[sp_i, 2] - rlo_s[sp_i, 2]
                cost = lc * (dl0 * dl1 + dl1 * dl2 + dl2 * dl0) + rc * (dr0 * dr1 + dr1 * dr2 + dr2 * dr0)
                if cost < best_cost:
                    best_cost = cost
                    best_axis = a
                    best_split = sp_i
        if best_axis < 0:
            # all centroids coincide: split in the middle of the index range
            mid = (s + e) // 2
        else:
            a = best_axis
            ext = chi[a] - clo[a]
            i = s
            j = e - 1
            while i <= j:
                b = int(NB * (cent[idx[i], a] - clo[a]) / ext)
                if b >= NB:
                    b = NB - 1
                if b < best_split:
                    i += 1
                else:
                    tmp = idx[i]
                    idx[i] = idx[j]
                    idx[j] = tmp
                    j -= 1
            mid = i
            if mid == s or mid == e:
                mid = (s + e) // 2
        l = nodes
        nodes += 2
        left[node] = l
        count[node] = 0
        stack[sp, 0] = l
        stack[sp, 1] = s
        stack[sp, 2] = mid
        sp += 1
        stack[sp, 0] = l + 1
        stack[sp, 1] = mid
        stack[sp, 2] = e
        sp += 1
    return nodes


class Geometry:
    """Triangle data reordered for the BVH, plus everything a hit needs for shading."""

    def __init__(self, sc):
        V = sc.verts.astype(np.float64)
        T = sc.tris
        a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        n = len(T)
        bmin = np.minimum(np.minimum(a, b), c)
        bmax = np.maximum(np.maximum(a, b), c)
        cent = (bmin + bmax) * 0.5
        idx = np.arange(n, dtype=np.int64)
        nmin = np.zeros((2 * n, 3))
        nmax = np.zeros((2 * n, 3))
        left = np.full(2 * n, -1, np.int64)
        start = np.zeros(2 * n, np.int64)
        count = np.zeros(2 * n, np.int64)
        nodes = _build(bmin, bmax, cent, idx, nmin, nmax, left, start, count)
        self.nmin = nmin[:nodes].astype(np.float32)
        self.nmax = nmax[:nodes].astype(np.float32)
        self.left = left[:nodes].astype(np.int32)
        self.start = start[:nodes].astype(np.int32)
        self.count = count[:nodes].astype(np.int32)
        a, b, c = a[idx], b[idx], c[idx]
        self.v0 = a.astype(np.float32)
        self.e1 = (b - a).astype(np.float32)
        self.e2 = (c - a).astype(np.float32)
        ng = np.cross(b - a, c - a)
        ln = np.linalg.norm(ng, axis=1)
        ng /= np.maximum(ln, 1e-12)[:, None]
        self.ng = ng.astype(np.float32)
        Ti = T[idx]
        self.uv = np.ascontiguousarray(sc.uv[Ti]).astype(np.float32)          # (n,3,2)
        self.col = np.ascontiguousarray(sc.colors[Ti]).astype(np.float32)     # (n,3,3)
        self.mat = sc.tri_mat[idx].astype(np.int32)
        self.owner = sc.tri_owner[idx].astype(np.int32)
        self.materials = sc.materials.astype(np.float32)
        self.tex = sc.textures.astype(np.float32)
        self.ntris = n
        self.nnodes = nodes

    def args(self):
        return (self.nmin, self.nmax, self.left, self.start, self.count, self.v0, self.e1, self.e2, self.ng,
                self.uv, self.col, self.mat, self.materials, self.tex, self.owner)


# ------------------------------------------------------------------------------------------------
# intersection
# ------------------------------------------------------------------------------------------------
@njit(cache=True, inline='always')
def _tex_sample(tex, tid, u, v, ch):
    n = tex.shape[1]
    x = int(math.floor((u - math.floor(u)) * n)) % n
    y = int(math.floor((v - math.floor(v)) * n)) % n
    return tex[tid, y, x, ch]


@njit(cache=True)
def intersect(ox, oy, oz, dx, dy, dz, tmax, any_hit, G, ignore=-1):
    """-> (tri, t, u, v); tri = -1 on miss. Triangles of placed reference `ignore` are skipped
    (a light's own fixture mesh: the game's lights shine through their lanterns, crystals and sconces)."""
    nmin, nmax, left, start, count, v0, e1, e2, ng, uv, col, mat, materials, tex, owner = G
    ix = 1.0 / dx if abs(dx) > 1e-12 else 1e12
    iy = 1.0 / dy if abs(dy) > 1e-12 else 1e12
    iz = 1.0 / dz if abs(dz) > 1e-12 else 1e12
    stack = np.empty(STACK, np.int32)
    sp = 0
    stack[0] = 0
    sp = 1
    best = tmax
    bt = -1
    bu = 0.0
    bv = 0.0
    while sp > 0:
        sp -= 1
        node = stack[sp]
        t0x = (nmin[node, 0] - ox) * ix
        t1x = (nmax[node, 0] - ox) * ix
        t0y = (nmin[node, 1] - oy) * iy
        t1y = (nmax[node, 1] - oy) * iy
        t0z = (nmin[node, 2] - oz) * iz
        t1z = (nmax[node, 2] - oz) * iz
        tmin_ = max(max(min(t0x, t1x), min(t0y, t1y)), max(min(t0z, t1z), 0.0))
        tmax_ = min(min(max(t0x, t1x), max(t0y, t1y)), min(max(t0z, t1z), best))
        if tmin_ > tmax_:
            continue
        l = left[node]
        if l >= 0:
            if sp + 2 > STACK:
                continue
            stack[sp] = l
            stack[sp + 1] = l + 1
            sp += 2
            continue
        s = start[node]
        for k in range(s, s + count[node]):
            ax, ay, az = e1[k, 0], e1[k, 1], e1[k, 2]
            bx, by, bz = e2[k, 0], e2[k, 1], e2[k, 2]
            px = dy * bz - dz * by
            py = dz * bx - dx * bz
            pz = dx * by - dy * bx
            det = ax * px + ay * py + az * pz
            if abs(det) < 1e-10:
                continue
            inv = 1.0 / det
            tx = ox - v0[k, 0]
            ty = oy - v0[k, 1]
            tz = oz - v0[k, 2]
            u = (tx * px + ty * py + tz * pz) * inv
            if u < 0.0 or u > 1.0:
                continue
            qx = ty * az - tz * ay
            qy = tz * ax - tx * az
            qz = tx * ay - ty * ax
            v = (dx * qx + dy * qy + dz * qz) * inv
            if v < 0.0 or u + v > 1.0:
                continue
            t = (bx * qx + by * qy + bz * qz) * inv
            if t <= 1e-4 or t >= best:
                continue
            if ignore >= 0 and owner[k] == ignore:
                continue
            m = mat[k]
            if materials[m, 2] > 0.5:      # alpha test
                w = 1.0 - u - v
                tu = w * uv[k, 0, 0] + u * uv[k, 1, 0] + v * uv[k, 2, 0]
                tv = w * uv[k, 0, 1] + u * uv[k, 1, 1] + v * uv[k, 2, 1]
                if _tex_sample(tex, int(materials[m, 0]), tu, tv, 3) < 0.5:
                    continue
            best = t
            bt = k
            bu = u
            bv = v
            if any_hit:
                return bt, best, bu, bv
    return bt, best, bu, bv


@njit(cache=True, inline='always')
def surface(G, k, u, v, dx, dy, dz):
    """-> albedo rgb, facing normal, is_backface."""
    nmin, nmax, left, start, count, v0, e1, e2, ng, uv, col, mat, materials, tex, owner = G
    w = 1.0 - u - v
    m = mat[k]
    tid = int(materials[m, 0])
    tu = w * uv[k, 0, 0] + u * uv[k, 1, 0] + v * uv[k, 2, 0]
    tv = w * uv[k, 0, 1] + u * uv[k, 1, 1] + v * uv[k, 2, 1]
    r = _tex_sample(tex, tid, tu, tv, 0) * (w * col[k, 0, 0] + u * col[k, 1, 0] + v * col[k, 2, 0])
    g = _tex_sample(tex, tid, tu, tv, 1) * (w * col[k, 0, 1] + u * col[k, 1, 1] + v * col[k, 2, 1])
    b = _tex_sample(tex, tid, tu, tv, 2) * (w * col[k, 0, 2] + u * col[k, 1, 2] + v * col[k, 2, 2])
    nx, ny, nz = ng[k, 0], ng[k, 1], ng[k, 2]
    back = False
    if nx * dx + ny * dy + nz * dz > 0.0:
        nx, ny, nz = -nx, -ny, -nz
        back = materials[m, 1] < 0.5
    return r, g, b, nx, ny, nz, back


@njit(cache=True, inline='always')
def emission(G, k, u, v):
    """Light a surface gives off (emissive color x glow map), in the same units as lit surfaces."""
    nmin, nmax, left, start, count, v0, e1, e2, ng, uv, col, mat, materials, tex, owner = G
    m = mat[k]
    er = materials[m, 3]
    eg = materials[m, 4]
    eb = materials[m, 5]
    if er + eg + eb <= 0.0:
        return 0.0, 0.0, 0.0
    gid = int(materials[m, 6])
    if gid >= 0:
        w = 1.0 - u - v
        tu = w * uv[k, 0, 0] + u * uv[k, 1, 0] + v * uv[k, 2, 0]
        tv = w * uv[k, 0, 1] + u * uv[k, 1, 1] + v * uv[k, 2, 1]
        er *= _tex_sample(tex, gid, tu, tv, 0)
        eg *= _tex_sample(tex, gid, tu, tv, 1)
        eb *= _tex_sample(tex, gid, tu, tv, 2)
    return er, eg, eb


# ------------------------------------------------------------------------------------------------
# lights
# ------------------------------------------------------------------------------------------------
def light_grid(lights, lo, hi, cell=192.0):
    """CSR lists of lights touching each grid cell, to keep per-hit light loops short."""
    lo = np.asarray(lo, np.float64) - 1.0
    hi = np.asarray(hi, np.float64) + 1.0
    dims = np.maximum(1, np.ceil((hi - lo) / cell).astype(np.int64))
    L = np.ascontiguousarray(lights, np.float64)
    if L.ndim != 2:
        L = L.reshape(0, 15)
    counts = np.zeros(int(np.prod(dims)), np.int32)
    _light_cells(L, lo, cell, dims, counts, counts, np.zeros(0, np.int32), False)
    off = np.zeros(len(counts) + 1, np.int32)
    np.cumsum(counts, out=off[1:])
    flat = np.zeros(int(off[-1]), np.int32)
    _light_cells(L, lo, cell, dims, counts, off, flat, True)
    return (lo.astype(np.float32), np.float32(cell), dims.astype(np.int32), off, flat)


@njit(cache=True)
def _light_cells(lights, lo, cell, dims, counts, off, flat, fill):
    """Pass 1 (fill=False): count lights per grid cell. Pass 2: write light ids in light order."""
    if fill:
        counts[:] = 0
    for i in range(lights.shape[0]):
        r = lights[i, 6]
        a = np.empty(3, np.int64)
        b = np.empty(3, np.int64)
        for d in range(3):
            a[d] = min(max(int(math.floor((lights[i, d] - r - lo[d]) / cell)), 0), dims[d] - 1)
            b[d] = min(max(int(math.floor((lights[i, d] + r - lo[d]) / cell)), 0), dims[d] - 1)
        for x in range(a[0], b[0] + 1):
            for y in range(a[1], b[1] + 1):
                for z in range(a[2], b[2] + 1):
                    c = (z * dims[1] + y) * dims[0] + x
                    if fill:
                        flat[off[c] + counts[c]] = i
                    counts[c] += 1


@njit(cache=True)
def direct(px, py, pz, nx, ny, nz, lights, LG, G, shadow_mode, out):
    """Accumulate direct light per layer into out[layer, rgb] (layer -1 lights go to the last row).
    shadow_mode: 0 = only lights flagged shadowed are occluded, 1 = every light is occluded, 2 = none."""
    glo, gcell, gdims, goff, gidx = LG
    cx = int((px - glo[0]) / gcell)
    cy = int((py - glo[1]) / gcell)
    cz = int((pz - glo[2]) / gcell)
    if cx < 0 or cy < 0 or cz < 0 or cx >= gdims[0] or cy >= gdims[1] or cz >= gdims[2]:
        return
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
        ndl = nx * lx + ny * ly + nz * lz
        if ndl <= 0.0:
            continue
        f = d / r
        att = (1.0 - f * f) * ndl
        if lights[i, 7] > -1.5:          # spot cone
            ca = -(lx * lights[i, 9] + ly * lights[i, 10] + lz * lights[i, 11])
            co = lights[i, 7]
            ci = lights[i, 8]
            if ca <= co:
                continue
            if ca < ci:
                s = (ca - co) / (ci - co)
                att *= s * s * (3.0 - 2.0 * s)
        if shadow_mode == 1 or (shadow_mode == 0 and lights[i, 12] > 0.5):
            tmax = d - LIGHT_SKIP
            if tmax > EPS:
                hit = intersect(px + nx * EPS, py + ny * EPS, pz + nz * EPS, lx, ly, lz, tmax, True, G,
                                int(lights[i, 14]))
                if hit[0] >= 0:
                    continue
        layer = int(lights[i, 13])
        if layer < 0:
            layer = out.shape[0] - 1
        out[layer, 0] += lights[i, 3] * att
        out[layer, 1] += lights[i, 4] * att
        out[layer, 2] += lights[i, 5] * att


# ------------------------------------------------------------------------------------------------
# random numbers / sampling
# ------------------------------------------------------------------------------------------------
@njit(cache=True, inline='always')
def _rng(state):
    state[0] ^= (state[0] << np.uint64(13))
    state[0] ^= (state[0] >> np.uint64(7))
    state[0] ^= (state[0] << np.uint64(17))
    return (state[0] >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, inline='always')
def _cosine_dir(nx, ny, nz, r1, r2):
    phi = 2.0 * math.pi * r1
    sr = math.sqrt(r2)
    lx = math.cos(phi) * sr
    ly = math.sin(phi) * sr
    lz = math.sqrt(max(0.0, 1.0 - r2))
    if abs(nz) < 0.999:
        tx, ty, tz = -ny, nx, 0.0
    else:
        tx, ty, tz = 0.0, -nz, ny
    tl = math.sqrt(tx * tx + ty * ty + tz * tz)
    tx /= tl
    ty /= tl
    tz /= tl
    bx = ny * tz - nz * ty
    by = nz * tx - nx * tz
    bz = nx * ty - ny * tx
    return (tx * lx + bx * ly + nx * lz, ty * lx + by * ly + ny * lz, tz * lx + bz * ly + nz * lz)


@njit(cache=True)
def trace_path(ox, oy, oz, dx, dy, dz, lights, LG, G, max_bounce, state, acc):
    """Radiance arriving at o from direction d, per layer, accumulated into acc[layer, rgb].
    Returns 0 = hit front, 1 = backface, 2 = miss (first hit only)."""
    tr = 1.0
    tg = 1.0
    tb = 1.0
    status = 0
    nl = acc.shape[0]
    tmp = np.zeros((nl, 3))
    for bounce in range(max_bounce):
        k, t, u, v = intersect(ox, oy, oz, dx, dy, dz, 1e7, False, G)
        if k < 0:
            if bounce == 0:
                status = 2
            break
        r, g, b, nx, ny, nz, back = surface(G, k, u, v, dx, dy, dz)
        if back:
            if bounce == 0:
                status = 1
            break
        px = ox + dx * t
        py = oy + dy * t
        pz = oz + dz * t
        # glowing surfaces (mushrooms, crystals, coals, lit windows) light their surroundings (fixture layer)
        emr, emg, emb = emission(G, k, u, v)
        acc[0, 0] += tr * emr
        acc[0, 1] += tg * emg
        acc[0, 2] += tb * emb
        tmp[:, :] = 0.0
        direct(px, py, pz, nx, ny, nz, lights, LG, G, 1, tmp)
        for li in range(nl):
            acc[li, 0] += tr * r * tmp[li, 0]
            acc[li, 1] += tg * g * tmp[li, 1]
            acc[li, 2] += tb * b * tmp[li, 2]
        tr *= r
        tg *= g
        tb *= b
        if bounce >= 1:
            pmax = max(tr, max(tg, tb))
            if pmax < 1e-3:
                break
            q = min(0.95, pmax)
            if _rng(state) > q:
                break
            tr /= q
            tg /= q
            tb /= q
        dx, dy, dz = _cosine_dir(nx, ny, nz, _rng(state), _rng(state))
        ox = px + nx * EPS
        oy = py + ny * EPS
        oz = pz + nz * EPS
    return status


# ------------------------------------------------------------------------------------------------
# probe bake
# ------------------------------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def bake_probes(points, rays, max_bounce, nlayers, lights, LG, G, seed, out_sh, out_stats):
    """out_sh[p, layer, coeff(4: Y00, Y1-1(y), Y10(z), Y11(x)), rgb] radiance SH.
    out_stats[p] = (backface fraction, miss fraction, mean hit distance)."""
    n = points.shape[0]
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for p in prange(n):
        state = np.empty(1, np.uint64)
        state[0] = np.uint64(seed) * np.uint64(2654435761) + np.uint64(p) * np.uint64(97531) + np.uint64(12345)
        for _ in range(4):
            _rng(state)
        # random rotation of a Fibonacci sphere (stratified directions)
        a1 = _rng(state) * 2.0 * math.pi
        a2 = math.acos(2.0 * _rng(state) - 1.0)
        a3 = _rng(state) * 2.0 * math.pi
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        c3, s3 = math.cos(a3), math.sin(a3)
        acc = np.zeros((nlayers + 1, 3))
        back = 0
        miss = 0
        for i in range(rays):
            z = 1.0 - (2.0 * i + 1.0) / rays
            rr = math.sqrt(max(0.0, 1.0 - z * z))
            ph = golden * i
            x = math.cos(ph) * rr
            y = math.sin(ph) * rr
            # rotate: Rz(a1) Rx(a2) Rz(a3)
            x1 = c3 * x - s3 * y
            y1 = s3 * x + c3 * y
            z1 = z
            y2 = c2 * y1 - s2 * z1
            z2 = s2 * y1 + c2 * z1
            dx = c1 * x1 - s1 * y2
            dy = s1 * x1 + c1 * y2
            dz = z2
            acc[:, :] = 0.0
            st = trace_path(points[p, 0], points[p, 1], points[p, 2], dx, dy, dz, lights, LG, G, max_bounce,
                            state, acc)
            if st == 1:
                back += 1
            elif st == 2:
                miss += 1
            for li in range(nlayers):
                for ch in range(3):
                    L = acc[li, ch]
                    out_sh[p, li, 0, ch] += L * SH_Y0
                    out_sh[p, li, 1, ch] += L * SH_Y1 * dy
                    out_sh[p, li, 2, ch] += L * SH_Y1 * dz
                    out_sh[p, li, 3, ch] += L * SH_Y1 * dx
        w = 4.0 * math.pi / rays
        for li in range(nlayers):
            for c in range(4):
                for ch in range(3):
                    out_sh[p, li, c, ch] *= w
        out_stats[p, 0] = back / rays
        out_stats[p, 1] = miss / rays


@njit(cache=True, parallel=True)
def probe_validity(points, rays, G, out, stop=1.0):
    """Cheap pre-pass: fraction of first hits that are back faces (probe inside geometry or outside the walls).
    Counting stops once the fraction exceeds `stop` (the point is rejected anyway); values up to it are exact."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for p in prange(points.shape[0]):
        back = 0
        for i in range(rays):
            z = 1.0 - (2.0 * i + 1.0) / rays
            rr = math.sqrt(max(0.0, 1.0 - z * z))
            dx = math.cos(golden * i + 0.37 * p) * rr
            dy = math.sin(golden * i + 0.37 * p) * rr
            k, t, u, v = intersect(points[p, 0], points[p, 1], points[p, 2], dx, dy, z, 1e7, False, G)
            if k < 0 or surface(G, k, u, v, dx, dy, z)[6]:
                back += 1
                if back > stop * rays:
                    break
        out[p] = back / rays


@njit(cache=True, parallel=True)
def light_validity(points, ignore, rays, G, out):
    """Share of rays from a light position that hit the back of a surface (= the light is inside solid geometry).
    Unlike probe validity, rays escaping to the sky do not count: a light in a window opening is not inside a wall.
    The light's own fixture mesh (ignore[p]) does not count either."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for p in prange(points.shape[0]):
        back = 0
        for i in range(rays):
            z = 1.0 - (2.0 * i + 1.0) / rays
            rr = math.sqrt(max(0.0, 1.0 - z * z))
            dx = math.cos(golden * i + 0.37 * p) * rr
            dy = math.sin(golden * i + 0.37 * p) * rr
            k, t, u, v = intersect(points[p, 0], points[p, 1], points[p, 2], dx, dy, z, 1e7, False, G, ignore[p])
            if k >= 0 and surface(G, k, u, v, dx, dy, z)[6]:
                back += 1
        out[p] = back / rays




# ------------------------------------------------------------------------------------------------
# probe sampling - identical to the in-game shader (Radiance.hlsli): trilinear over the probe grid,
# normal offset, per-layer clamp. index[x,y,z] = row in coef or -1 where no brick covers the probe.
# ------------------------------------------------------------------------------------------------
@njit(cache=True, inline='always')
def sample_volume(px, py, pz, nx, ny, nz, grid_lo, spacing, dims, index, coef, layer_w, offset, out):
    """Baked irradiance (game units) at p with normal n into out[3]. Returns False where the volume has no data."""
    fx = (px + nx * offset - grid_lo[0]) / spacing
    fy = (py + ny * offset - grid_lo[1]) / spacing
    fz = (pz + nz * offset - grid_lo[2]) / spacing
    fx = min(max(fx, 0.0), dims[0] - 1.0)
    fy = min(max(fy, 0.0), dims[1] - 1.0)
    fz = min(max(fz, 0.0), dims[2] - 1.0)
    ix = min(int(fx), dims[0] - 2)
    iy = min(int(fy), dims[1] - 2)
    iz = min(int(fz), dims[2] - 2)
    tx = fx - ix
    ty = fy - iy
    tz = fz - iz
    nl = coef.shape[1]
    acc = np.zeros((nl, 3))
    wsum = 0.0
    for c in range(8):
        ox = c & 1
        oy = (c >> 1) & 1
        oz = (c >> 2) & 1
        row = index[ix + ox, iy + oy, iz + oz]
        if row < 0:
            continue
        w = (tx if ox else 1.0 - tx) * (ty if oy else 1.0 - ty) * (tz if oz else 1.0 - tz)
        wsum += w
        for li in range(nl):
            for ch in range(3):
                acc[li, ch] += w * (coef[row, li, 0, ch] + coef[row, li, 1, ch] * nx + coef[row, li, 2, ch] * ny
                                    + coef[row, li, 3, ch] * nz)
    out[0] = 0.0
    out[1] = 0.0
    out[2] = 0.0
    if wsum < 0.999:          # the shader only samples inside a brick, where all 8 corners exist
        return False
    for li in range(nl):
        for ch in range(3):
            out[ch] += max(acc[li, ch] / wsum, 0.0) * layer_w[li]
    return True


# ------------------------------------------------------------------------------------------------
# camera renders
# ------------------------------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def render(cam, w, h, mode, lights_today, LG_today, lights_live, LG_live, lights_ref, LG_ref, lights_bake, LG_bake,
           G, dalc, dir_col, dir_vec, grid_lo, spacing, dims, index, coef, layer_w, offset, ambient_floor,
           bounce_scale, spp, max_bounce, img):
    """mode 0 = today: every light as the game draws it (shadow flags as authored) + flat directional ambient
       mode 1 = baked: live fixture lights (after the light patch) + baked probes + ambient floor
       mode 2 = reference: the same live lights + baked lights directly with real shadows + path-traced bounce
       mode 3 = baked probes only
       mode 4 = flat directional ambient only"""
    ex, ey, ez = cam[0], cam[1], cam[2]
    fx, fy, fz = cam[3], cam[4], cam[5]
    rx, ry, rz = cam[6], cam[7], cam[8]
    ux, uy, uz = cam[9], cam[10], cam[11]
    tanh = cam[12]
    for pix in prange(w * h):
        py_ = pix // w
        px_ = pix % w
        state = np.empty(1, np.uint64)
        state[0] = np.uint64(pix) * np.uint64(2654435761) + np.uint64(777)
        for _ in range(3):
            _rng(state)
        sx = (2.0 * (px_ + 0.5) / w - 1.0) * tanh
        sy = (1.0 - 2.0 * (py_ + 0.5) / h) * tanh * h / w
        dx = fx + rx * sx + ux * sy
        dy = fy + ry * sx + uy * sy
        dz = fz + rz * sx + uz * sy
        dl = math.sqrt(dx * dx + dy * dy + dz * dz)
        dx /= dl
        dy /= dl
        dz /= dl
        k, t, u, v = intersect(ex, ey, ez, dx, dy, dz, 1e7, False, G)
        if k < 0:
            img[py_, px_, :] = 0.0
            continue
        ar, ag, ab, nx, ny, nz, back = surface(G, k, u, v, dx, dy, dz)
        px = ex + dx * t
        py = ey + dy * t
        pz = ez + dz * t
        if back:
            # single-sided face seen from behind (the game culls it): dark purple so it is easy to spot
            img[py_, px_, 0] = 0.05
            img[py_, px_, 1] = 0.0
            img[py_, px_, 2] = 0.05
            continue
        emr, emg, emb = emission(G, k, u, v)
        Er = 0.0
        Eg = 0.0
        Eb = 0.0
        tmp = np.zeros((3, 3))
        amb = np.zeros(3)
        # the game's directional ambient: a 3x4 matrix applied to (n, 1)
        cr = max(0.0, dalc[0, 0] * nx + dalc[0, 1] * ny + dalc[0, 2] * nz + dalc[0, 3])
        cg = max(0.0, dalc[1, 0] * nx + dalc[1, 1] * ny + dalc[1, 2] * nz + dalc[1, 3])
        cb = max(0.0, dalc[2, 0] * nx + dalc[2, 1] * ny + dalc[2, 2] * nz + dalc[2, 3])
        ndl = -(nx * dir_vec[0] + ny * dir_vec[1] + nz * dir_vec[2])
        if ndl > 0.0:
            Er += dir_col[0] * ndl
            Eg += dir_col[1] * ndl
            Eb += dir_col[2] * ndl
        if mode == 0:
            direct(px, py, pz, nx, ny, nz, lights_today, LG_today, G, 0, tmp)
            Er += cr
            Eg += cg
            Eb += cb
        elif mode == 1:
            direct(px, py, pz, nx, ny, nz, lights_live, LG_live, G, 0, tmp)
            if sample_volume(px, py, pz, nx, ny, nz, grid_lo, spacing, dims, index, coef, layer_w, offset, amb):
                Er += amb[0] + ambient_floor * cr
                Eg += amb[1] + ambient_floor * cg
                Eb += amb[2] + ambient_floor * cb
            else:
                Er += cr
                Eg += cg
                Eb += cb
        elif mode == 2:
            direct(px, py, pz, nx, ny, nz, lights_ref, LG_ref, G, 0, tmp)
            acc = np.zeros((3, 3))
            for s in range(spp):
                ddx, ddy, ddz = _cosine_dir(nx, ny, nz, _rng(state), _rng(state))
                trace_path(px + nx * EPS, py + ny * EPS, pz + nz * EPS, ddx, ddy, ddz, lights_bake, LG_bake, G,
                           max_bounce, state, acc)
            for li in range(2):
                Er += acc[li, 0] / spp * layer_w[li] * bounce_scale
                Eg += acc[li, 1] / spp * layer_w[li] * bounce_scale
                Eb += acc[li, 2] / spp * layer_w[li] * bounce_scale
            Er += ambient_floor * cr
            Eg += ambient_floor * cg
            Eb += ambient_floor * cb
        elif mode == 3:
            sample_volume(px, py, pz, nx, ny, nz, grid_lo, spacing, dims, index, coef, layer_w, offset, amb)
            Er = amb[0]
            Eg = amb[1]
            Eb = amb[2]
        else:
            Er = cr
            Eg = cg
            Eb = cb
        for li in range(3):
            Er += tmp[li, 0]
            Eg += tmp[li, 1]
            Eb += tmp[li, 2]
        img[py_, px_, 0] = ar * Er + emr
        img[py_, px_, 1] = ag * Eg + emg
        img[py_, px_, 2] = ab * Eb + emb
