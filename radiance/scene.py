"""Build a ray-traceable scene for one interior cell from the winning records of the load order."""
import hashlib
import json
import math
import os
import re
import struct

import numpy as np

from . import nif
from .plugins import first, fk_str, zstr
from .textures import TextureBank


class NothingToBake(Exception):
    """The cell has nothing a probe volume could light (empty, test or unfinished cell). Not an error."""


# Light classes
EMITTER, DAYLIGHT, FILL = 'emitter', 'daylight', 'fill'
LAYERS = {EMITTER: 0, DAYLIGHT: 1}       # probe layers; FILL lights are not baked

LIGHT_FLAG_NEGATIVE = 0x4
LIGHT_FLAG_OFF = 0x20
LIGHT_FLAG_SPOT = 0x200 | 0x400
LIGHT_FLAG_SHADOW = 0x400 | 0x800 | 0x1000

SKIP_MODEL = re.compile(r'(^|\\)(marker|markers|editor)|marker_|xmarker|fxlightrays|fxfog|fxmist|fxdust|'
                        r'fxambbeam|lightbeam|lightray|cobweb', re.I)
FIXTURE_MODEL = re.compile(r'candle|torch|lantern|chandel|brazier|fire|hearth|lamp|sconce|ember|lightbulb|'
                           r'window|glow|crystal|mushroom|coals|forge|smelter|brazer|lightorb|lightcone', re.I)

DEFAULT_RULES = {
    # Regexes on the light's EditorID, checked in this order. First match wins.
    'fill': [r'Bounce', r'Fill', r'Amb(ient)?([A-Z0-9_]|$)', r'Desat(Green)?Light', r'Spot\d*(NoPort|Clip)?\d*$',
             r'InteriorLight\d'],
    'daylight': [r'Window', r'Sun(light)?', r'Daylight', r'Skylight', r'Moon'],
    'emitter': [r'Fire', r'Candle', r'Torch', r'Flicker', r'Ember', r'Lantern', r'Chandelier', r'Brazier',
                r'Bazier', r'Lamp', r'Sconce', r'Hearth', r'Forge', r'Crystal', r'Mushroom'],
    # Lights with no rule match: emitter if a visible fixture mesh is within this distance, else fill.
    'fixture_distance': 72.0,
    # Lights at least this large are fully baked (static key lights), whatever their name.
    'static_radius': 1500.0,
}


def euler_matrix(rx, ry, rz):
    """Bethesda reference rotation -> 3x3 for row vectors (p' = p @ m)."""
    def rot(axis, a):
        c, s = math.cos(a), math.sin(a)
        if axis == 0:
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == 1:
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    m = rot(2, -rz) @ rot(1, -ry) @ rot(0, -rx)
    return m.T


def rgba(b, o):
    return np.frombuffer(bytes(b[o:o + 3]), np.uint8).astype(np.float32) / 255.0


class Light:
    __slots__ = ('ref', 'edid', 'base_plugin', 'pos', 'color', 'radius', 'fade', 'flags', 'fov', 'dir', 'cls',
                 'reason', 'has_model', 'fixture_dist', 'owner', 'keep_shadow', 'live', 'bake_pos', 'switched')

    def __init__(self):
        self.owner = -1          # placed reference whose mesh holds the light (ignored for its shadows)
        self.keep_shadow = False  # decided by the bake: removing this light's shadow would leak light
        self.live = True          # stays a real light in game (False: fully baked or replaced, hidden in game)
        self.bake_pos = None      # where the bake places it when its own position is inside a wall (else pos)
        self.switched = False     # the game turns it on/off at runtime (enable parent): never baked

    def to_dict(self):
        return {'ref': self.ref, 'edid': self.edid, 'plugin': self.base_plugin,
                'pos': [round(float(x), 1) for x in self.pos], 'color': [round(float(x), 3) for x in self.color],
                'radius': round(self.radius, 1), 'fade': round(self.fade, 3), 'flags': hex(self.flags),
                'shadow': bool(self.flags & LIGHT_FLAG_SHADOW), 'spot': bool(self.flags & LIGHT_FLAG_SPOT),
                'fov': round(self.fov, 1), 'class': self.cls, 'reason': self.reason,
                'fixture_dist': None if self.fixture_dist is None else round(self.fixture_dist, 1),
                'keep_shadow': bool(self.keep_shadow), 'live': bool(self.live)}


def load_rules(path):
    rules = dict(DEFAULT_RULES)
    if path and os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            user = json.load(f)
        for k in ('fill', 'daylight', 'emitter'):
            if k in user:
                rules[k] = list(user[k]) + ([] if user.get('replace_defaults') else rules[k])
        for k in ('fixture_distance', 'static_radius'):
            if k in user:
                rules[k] = float(user[k])
        rules['overrides'] = user.get('overrides', {})   # "EditorID" or "000ABC:Plugin.esp" -> class
    rules.setdefault('overrides', {})
    return rules


def classify(light, rules):
    ov = rules['overrides'].get(light.ref) or rules['overrides'].get(light.edid)
    if ov:
        return ov, 'override'
    if light.flags & LIGHT_FLAG_SHADOW:
        # the author paid for a shadow map: a deliberate key light, never a fake fill. It stays live with its
        # shadows (a probe grid cannot hold sharp shadows); its bounce is baked.
        for pat in rules['daylight']:
            if re.search(pat, light.edid):
                return DAYLIGHT, f'shadowed, name~{pat}'
        return EMITTER, 'authored shadows (key light)'
    for pat in rules['fill']:
        if re.search(pat, light.edid):
            return FILL, f'name~{pat}'
    if light.radius >= rules['static_radius']:
        # nothing that lights a whole hall is a fixture: bake it fully, with real shadows
        return DAYLIGHT, f'large light (radius {light.radius:.0f})'
    for cls in ('daylight', 'emitter'):
        for pat in rules[cls]:
            if re.search(pat, light.edid):
                return cls, f'name~{pat}'
    if light.has_model:
        return EMITTER, 'visible light model'
    if light.fixture_dist is not None and light.fixture_dist <= rules['fixture_distance']:
        return EMITTER, f'fixture {light.fixture_dist:.0f}u away'
    return FILL, 'no fixture nearby'


class CellLighting:
    """Vanilla interior lighting values (XCLL, with lighting-template inheritance)."""

    def __init__(self):
        self.ambient = np.zeros(3, np.float32)
        self.directional = np.zeros(3, np.float32)
        self.dir_rot = (0, 0)
        self.dir_fade = 1.0
        self.dalc = np.zeros((6, 3), np.float32)   # +X -X +Y -Y +Z -Z
        self.fog_near = np.zeros(3, np.float32)

    def to_dict(self):
        return {'ambient': self.ambient.round(3).tolist(), 'directional': self.directional.round(3).tolist(),
                'dir_rot': list(self.dir_rot), 'dir_fade': round(self.dir_fade, 3),
                'directional_ambient': self.dalc.round(3).tolist()}

    def ambient_matrix(self):
        """3x4 matrix M with ambient(n) = M.(nx, ny, nz, 1), like the game's DirectionalAmbient.
        Approximation of the engine's conversion: slopes from opposite faces, constant = mean of the six."""
        d = self.dalc if self.dalc.any() else np.tile(self.ambient, (6, 1))
        m = np.zeros((3, 4), np.float32)
        for a in range(3):
            m[:, a] = (d[2 * a] - d[2 * a + 1]) * 0.5
        m[:, 3] = d.mean(0)
        return m

    def ambient_dir(self):
        """Direction the interior directional light travels (unit, z-up)."""
        rxy, rz = math.radians(self.dir_rot[0]), math.radians(self.dir_rot[1])
        d = np.array([math.sin(rz) * math.cos(rxy), math.cos(rz) * math.cos(rxy), math.sin(rxy)])
        return -(d / (np.linalg.norm(d) + 1e-9)).astype(np.float32)


def read_cell_lighting(lo, cell_plugin, cell_subs):
    cl = CellLighting()
    xcll = first(cell_subs, b'XCLL')
    tmpl = None
    ltmp = first(cell_subs, b'LTMP')
    if ltmp:
        rec = lo.base_record(cell_plugin.key(struct.unpack('<I', ltmp)[0]))
        if rec:
            tmpl = rec[3]
    inherit = struct.unpack_from('<I', xcll, 88)[0] if xcll and len(xcll) >= 92 else 0xFFFFFFFF

    def from_tmpl(bit):
        return tmpl is not None and (xcll is None or inherit & bit)

    t_data = first(tmpl, b'DATA') if tmpl else None
    t_dalc = first(tmpl, b'DALC') if tmpl else None
    if from_tmpl(0x1) and t_data:
        cl.ambient = rgba(t_data, 0)
        if t_dalc:
            cl.dalc = np.array([rgba(t_dalc, i * 4) for i in range(6)], np.float32)
    elif xcll:
        cl.ambient = rgba(xcll, 0)
        cl.dalc = np.array([rgba(xcll, 40 + i * 4) for i in range(6)], np.float32)
        if inherit & 0x1 and tmpl is None and (cl.dalc >= 0.999).all():
            # inherited from a template that is missing: the cell's own values are white placeholders
            cl.dalc = np.tile(cl.ambient, (6, 1))
    src = t_data if (from_tmpl(0x2) and t_data) else xcll
    if src:
        cl.directional = rgba(src, 4)
    src = t_data if (from_tmpl(0x20) and t_data) else xcll
    if src:
        cl.dir_rot = struct.unpack_from('<ii', src, 20)
    src = t_data if (from_tmpl(0x40) and t_data) else xcll
    if src:
        cl.dir_fade = struct.unpack_from('<f', src, 28)[0]
    if xcll:
        cl.fog_near = rgba(xcll, 8)
    return cl


class Scene:
    """Flat triangle soup + materials + lights for one cell."""

    def __init__(self):
        self.cell = None
        self.lights = []
        self.lighting = None
        self.log = []
        self.mesh_sources = {}


def build_scene(lo, key, rules=None, log=print):
    vfs = lo.vfs
    rules = rules or load_rules(None)
    sc = Scene()
    cell_plugin, cell_subs = lo.cell_record(key)
    sc.cell = lo.cell_summary(key)
    sc.lighting = read_cell_lighting(lo, cell_plugin, cell_subs)
    refs = lo.cell_refs(key)
    texbank = TextureBank(vfs)
    mesh_cache = {}

    verts, tris_v, uvs, cols, mats, owners = [], [], [], [], [], []
    ref_boxes = []          # bounds of each placed reference with geometry (index = triangle owner id)
    materials = []          # (tex id, two sided, alpha blend, emissive r, g, b)
    mat_ids = {}
    nverts = 0
    fixtures = []           # (bbox min, bbox max)
    light_refs = []
    skipped = {'disabled': 0, 'no_base': 0, 'no_model': 0, 'marker': 0, 'missing_mesh': 0, 'bad_mesh': 0}

    def get_mesh(path):
        k = path.lower()
        if k in mesh_cache:
            return mesh_cache[k]
        data = vfs.read('meshes\\' + k)
        res = None
        if data is None:
            skipped['missing_mesh'] += 1
        else:
            try:
                res = nif.load(data)
            except Exception as e:       # noqa: BLE001 - one broken mesh must not stop a bake
                skipped['bad_mesh'] += 1
                sc.log.append(f'bad mesh {path}: {e!r}')
            sc.mesh_sources[k] = vfs.source('meshes\\' + k)
        mesh_cache[k] = res
        return res

    for rk, (p, rflags, subs, raw) in refs.items():
        if rflags & 0x820:                       # deleted / initially disabled
            skipped['disabled'] += 1
            continue
        if first(subs, b'XESP'):
            esp = first(subs, b'XESP')
            if len(esp) >= 5 and esp[4] & 1:     # "opposite of parent": usually an alternate state
                skipped['disabled'] += 1
                continue
        name = first(subs, b'NAME')
        data = first(subs, b'DATA')
        if name is None or data is None or len(data) < 24:
            continue
        bkey = p.key(struct.unpack('<I', name)[0])
        br = lo.base_record(bkey)
        if br is None:
            skipped['no_base'] += 1
            continue
        bp, btype, bflags, bsubs = br
        x, y, z, rx, ry, rz = struct.unpack_from('<6f', data)
        if z < -20000:                            # "disabled and moved below the floor" convention
            skipped['disabled'] += 1
            continue
        xscl = first(subs, b'XSCL')
        scale = struct.unpack('<f', xscl)[0] if xscl else 1.0
        rot = euler_matrix(rx, ry, rz)
        pos = np.array([x, y, z])
        model = zstr(first(bsubs, b'MOD2' if btype == b'ARMO' else b'MODL'))
        if model and not model.isprintable():
            model = ''
        shapes, points = None, {}
        if model and SKIP_MODEL.search(model):
            skipped['marker'] += 1
            model = ''
        if model:
            mesh = get_mesh(model)
            if mesh:
                shapes, points = mesh
        if btype == b'LIGH':
            light_refs.append((rk, p, subs, bp, bsubs, pos, rot, scale, points, bool(shapes)))
        if not shapes:
            if not model and btype != b'LIGH':
                skipped['no_model'] += 1
            continue
        lo_b, hi_b = np.full(3, np.inf), np.full(3, -np.inf)
        emissive_ref = False
        for s in shapes:
            if s.alpha_blend and not s.alpha_test:
                continue                          # glass, cobwebs, soot cards: no occlusion, no bounce
            if s.decal:
                continue
            w = (s.verts.astype(np.float64) * scale) @ rot + pos
            tex = texbank.get(s.diffuse)
            em = s.emissive if s.emissive is not None else (0.0, 0.0, 0.0)
            glow = -1
            if s.emissive is not None:
                emissive_ref = True
                if s.glow:
                    g = texbank.get(s.glow)
                    glow = g if g > 0 else -1
            mk = (tex, s.two_sided, s.alpha_test, em, glow)
            mid = mat_ids.get(mk)
            if mid is None:
                mid = mat_ids[mk] = len(materials)
                materials.append((tex, float(s.two_sided), float(s.alpha_test), em[0], em[1], em[2], float(glow)))
            verts.append(w.astype(np.float32))
            uvs.append(s.uv if s.uv is not None else np.zeros((len(w), 2), np.float32))
            cols.append(s.colors[:, :3] if s.colors is not None else np.ones((len(w), 3), np.float32))
            tris_v.append(s.tris + nverts)
            mats.append(np.full(len(s.tris), mid, np.int32))
            owners.append(np.full(len(s.tris), len(ref_boxes), np.int32))
            nverts += len(w)
            lo_b = np.minimum(lo_b, w.min(0))
            hi_b = np.maximum(hi_b, w.max(0))
        if np.isfinite(lo_b).all():
            ref_boxes.append((lo_b, hi_b))
            if FIXTURE_MODEL.search(model) or emissive_ref:
                fixtures.append((lo_b, hi_b))

    if not verts:
        raise NothingToBake('the cell has no geometry')
    V = np.concatenate(verts)
    T = np.concatenate(tris_v)
    sc.verts = V
    sc.tris = T
    sc.uv = np.concatenate(uvs).astype(np.float32)
    sc.colors = np.clip(np.concatenate(cols), 0, 1).astype(np.float32)
    sc.tri_mat = np.concatenate(mats)
    sc.tri_owner = np.concatenate(owners)
    sc.materials = np.array(materials, np.float32).reshape(-1, 7)   # tex, two sided, alpha test, emissive rgb, glow tex
    sc.textures = texbank.array()
    sc.texture_missing = sorted(texbank.missing)
    sc.texture_ids = dict(texbank.ids)
    sc.skipped = skipped
    sc.bounds = (V.min(0), V.max(0))

    # ---- lights -----------------------------------------------------------------------------------
    fx_lo = np.array([f[0] for f in fixtures]) if fixtures else np.zeros((0, 3))
    fx_hi = np.array([f[1] for f in fixtures]) if fixtures else np.zeros((0, 3))
    rb_lo = np.array([b[0] for b in ref_boxes])
    rb_hi = np.array([b[1] for b in ref_boxes])
    rb_ext = (rb_hi - rb_lo).max(1)
    for rk, p, subs, bp, bsubs, pos, rot, scale, points, has_mesh in light_refs:
        d = first(bsubs, b'DATA')
        if d is None or len(d) < 24:
            continue
        _, radius, r, g, b, _, flags = struct.unpack_from('<iI4BI', d, 0)
        fov = struct.unpack_from('<f', d, 20)[0]
        if flags & LIGHT_FLAG_OFF:
            continue
        fnam = first(bsubs, b'FNAM')
        fade = struct.unpack('<f', fnam)[0] if fnam else 1.0
        xrds = first(subs, b'XRDS')
        if xrds:
            radius += struct.unpack('<f', xrds)[0]
        xlig = first(subs, b'XLIG')
        if xlig and len(xlig) >= 8:
            dfov, dfade = struct.unpack_from('<2f', xlig)
            fov += dfov
            fade += dfade
        L = Light()
        L.ref = fk_str(rk)
        L.edid = zstr(first(bsubs, b'EDID'))
        L.base_plugin = bp.name
        off = points.get('attachlight')
        L.pos = (pos + (off * scale) @ rot if off is not None else pos).astype(np.float32)
        L.color = np.array([r, g, b], np.float32) / 255.0 * max(fade, 0.0)
        L.radius = float(max(radius, 1.0))
        L.fade = float(fade)
        L.flags = flags
        L.fov = float(fov)
        L.dir = (np.array([1.0, 0, 0]) @ rot).astype(np.float32)
        L.has_model = has_mesh
        if len(fx_lo):
            dd = np.maximum(np.maximum(fx_lo - L.pos, 0), np.maximum(L.pos - fx_hi, 0))
            L.fixture_dist = float(np.sqrt((dd ** 2).sum(1)).min())
        else:
            L.fixture_dist = None
        # the fixture mesh around the light (lantern, sconce, crystal): smallest small object containing it
        inside = np.nonzero(((rb_lo - 8 <= L.pos) & (L.pos <= rb_hi + 8)).all(1) & (rb_ext <= 300))[0]
        if len(inside):
            L.owner = int(inside[np.argmin(rb_ext[inside])])
        L.cls, L.reason = classify(L, rules)
        # lights the author gave shadows stay live with them (a probe grid cannot hold sharp shadows);
        # unshadowed daylight / large lights already shine through walls in game, so baking them can only help
        L.live = L.cls == EMITTER or (L.cls == DAYLIGHT and bool(flags & LIGHT_FLAG_SHADOW))
        esp = first(subs, b'XESP')
        if esp and len(esp) >= 4 and p.key(struct.unpack_from('<I', esp, 0)[0]) != ('skyrim.esm', 0x14):
            # enabled by a parent (day/night, quest state): its light comes and goes, so the bake must not hold it
            L.switched = True
            L.live = True
            L.reason += ' (switched by the game: stays live, not baked)'
        if fade <= 0 or radius <= 0:
            continue
        sc.lights.append(L)
    return sc


def quick_fingerprint(lo, key, extra=''):
    """sha1 over everything a bake of this cell depends on, without loading meshes:
    every record version of the cell and its references, the winning base records and the identity
    (provider, size, date) of each winning mesh file, plus the lighting template."""
    h = lo.cell_fingerprint_parts(key)
    h.update(extra.encode())
    seen = set()
    for rk, (p, rflags, subs, raw) in sorted(lo.cell_refs(key).items()):
        name = first(subs, b'NAME')
        if name is None:
            continue
        bkey = p.key(struct.unpack('<I', name)[0])
        if bkey in seen:
            continue
        seen.add(bkey)
        h.update(lo.base_raw(bkey))
        br = lo.base_record(bkey)
        if br:
            model = zstr(first(br[3], b'MODL'))
            if model and model.isprintable():
                h.update(lo.vfs.source('meshes/' + model).encode())
    cp, subs = lo.cell_record(key)
    ltmp = first(subs, b'LTMP')
    if ltmp:
        h.update(lo.base_raw(cp.key(struct.unpack('<I', ltmp)[0])))
    return h.hexdigest()


def light_arrays(lights, classes=None, shadow='all', scale=None, live=None, at='game'):
    """Pack lights into a float32 array for the tracer. at='bake': light sources for the bake (positions moved
    out of walls, lights the game switches at runtime left out); at='game': as the game draws them.
    Columns: pos(3) color(3) radius, spot cos outer, spot cos inner, dir(3), shadowed, layer, own fixture.
    shadow: 'all' (every light occludes, for baking), 'game' (as authored), 'patched' (as after the light patch:
    only shadow spotlights keep shadows). scale: optional {class: color multiplier}."""
    rows = []
    for L in lights:
        if classes is not None and L.cls not in classes:
            continue
        if live is not None and L.live != live:
            continue
        if at == 'bake' and L.switched:
            continue
        pos = L.bake_pos if at == 'bake' and L.bake_pos is not None else L.pos
        spot = bool(L.flags & LIGHT_FLAG_SPOT)
        half = math.radians(max(1.0, min(L.fov, 179.0)) * 0.5)
        co = math.cos(half) if spot else -2.0
        ci = math.cos(half * 0.8) if spot else -1.0
        sign = -1.0 if L.flags & LIGHT_FLAG_NEGATIVE else 1.0
        if shadow == 'all':
            shadowed = 1.0
        elif shadow == 'patched':
            shadowed = 1.0 if (L.flags & 0x400 or (L.flags & LIGHT_FLAG_SHADOW and L.keep_shadow)) else 0.0
        else:
            shadowed = 1.0 if L.flags & LIGHT_FLAG_SHADOW else 0.0
        mult = (scale or {}).get(L.cls, 1.0)
        rows.append([*pos, *(L.color * sign * mult), L.radius, co, ci, *L.dir, shadowed,
                     float(LAYERS.get(L.cls, -1)), float(L.owner)])
    return np.array(rows, np.float32).reshape(-1, 15)
