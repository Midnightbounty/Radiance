"""SSE/VR NIF (20.2.0.7, BS stream 100) geometry reader.

Extracts render geometry only: triangle shapes with their world-of-mesh transform, UVs, vertex colors,
diffuse texture and the material flags that matter for lighting. Collision, skinning, particles and
effect-shader geometry (flames, light rays, fog) are ignored.
"""
import struct

import numpy as np

NODE_TYPES = {'NiNode', 'BSFadeNode', 'BSLeafAnimNode', 'BSMultiBoundNode', 'BSOrderedNode', 'BSValueNode',
              'BSTreeNode', 'BSMasterParticleSystem', 'NiBillboardNode', 'BSDamageStage', 'BSBlastNode',
              'BSDebrisNode', 'BSRangeNode', 'NiSwitchNode', 'NiLODNode', 'BSLODTriShape'}
TRISHAPES = {'BSTriShape', 'BSSubIndexTriShape', 'BSMeshLODTriShape', 'BSDynamicTriShape'}
SKIP_NAMES = ('editormarker', 'markerx', 'collision')


class Shape:
    __slots__ = ('name', 'verts', 'tris', 'uv', 'colors', 'normals', 'diffuse', 'glow', 'alpha_blend',
                 'alpha_test', 'two_sided', 'emissive', 'decal', 'use_vc')

    def __init__(self):
        self.uv = None
        self.colors = None
        self.normals = None
        self.diffuse = ''
        self.glow = ''
        self.alpha_blend = False
        self.alpha_test = False
        self.two_sided = False
        self.emissive = None
        self.decal = False
        self.use_vc = False


class R:
    """Little-endian cursor."""

    def __init__(self, b):
        self.b = b
        self.o = 0

    def u8(self):
        v = self.b[self.o]
        self.o += 1
        return v

    def u16(self):
        v = struct.unpack_from('<H', self.b, self.o)[0]
        self.o += 2
        return v

    def u32(self):
        v = struct.unpack_from('<I', self.b, self.o)[0]
        self.o += 4
        return v

    def i32(self):
        v = struct.unpack_from('<i', self.b, self.o)[0]
        self.o += 4
        return v

    def f32(self, n=1):
        v = struct.unpack_from('<%df' % n, self.b, self.o)
        self.o += 4 * n
        return v if n > 1 else v[0]

    def refs(self):
        n = self.u32()
        v = struct.unpack_from('<%di' % n, self.b, self.o)
        self.o += 4 * n
        return v


class Nif:
    def __init__(self, data):
        o = data.index(b'\n') + 1
        version, endian, user_ver, nblocks = struct.unpack_from('<IBII', data, o)
        o += 13
        if version != 0x14020007:
            raise ValueError('unsupported NIF version %08X' % version)
        self.bs_ver = struct.unpack_from('<I', data, o)[0]
        o += 4
        for _ in range(4 if self.bs_ver >= 103 else 3):
            o += 1 + data[o]
        nt = struct.unpack_from('<H', data, o)[0]
        o += 2
        types = []
        for _ in range(nt):
            n = struct.unpack_from('<I', data, o)[0]
            o += 4
            types.append(data[o:o + n].decode('latin1'))
            o += n
        tidx = struct.unpack_from('<%dH' % nblocks, data, o)
        o += 2 * nblocks
        sizes = struct.unpack_from('<%dI' % nblocks, data, o)
        o += 4 * nblocks
        ns, _ = struct.unpack_from('<II', data, o)
        o += 8
        self.strings = []
        for _ in range(ns):
            n = struct.unpack_from('<I', data, o)[0]
            o += 4
            self.strings.append(data[o:o + n].decode('latin1'))
            o += n
        ng = struct.unpack_from('<I', data, o)[0]
        o += 4 + 4 * ng
        self.types = [types[t & 0x7FFF] for t in tidx]
        self.blocks = []
        for s in sizes:
            self.blocks.append(data[o:o + s])
            o += s
        rest = data[o:]
        nroots = struct.unpack_from('<I', rest, 0)[0] if len(rest) >= 4 else 0
        self.roots = list(struct.unpack_from('<%di' % nroots, rest, 4)) if nroots else [0]

    def string(self, idx):
        return self.strings[idx] if 0 <= idx < len(self.strings) else ''

    # -- common prefixes -------------------------------------------------------------------------
    def _net(self, r):
        name = self.string(r.i32())
        r.refs()       # extra data
        r.i32()        # controller
        return name

    def _av(self, r):
        name = self._net(r)
        flags = r.u32()
        t = np.array(r.f32(3), dtype=np.float64)
        rot = np.array(r.f32(9), dtype=np.float64).reshape(3, 3)
        s = r.f32()
        r.i32()        # collision object
        # Gamebryo stores the matrix for row vectors: p' = p @ (rot * s) + t
        m = np.eye(4)
        m[:3, :3] = rot * s
        m[3, :3] = t
        return name, flags, m

    # -- traversal -------------------------------------------------------------------------------
    def walk(self):
        """-> (shapes, attach_light_points). Shape vertices are in mesh space (all node transforms applied)."""
        shapes = []
        points = {}
        seen = set()

        def visit(i, parent):
            if i < 0 or i >= len(self.blocks) or i in seen:
                return
            seen.add(i)
            t = self.types[i]
            if t in NODE_TYPES or t.endswith('Node'):
                r = R(self.blocks[i])
                name, flags, m = self._av(r)
                if flags & 1 or name.lower().startswith(SKIP_NAMES):
                    return
                world = m @ parent
                points[name.lower()] = world[3, :3].copy()
                children = r.refs()
                if t == 'NiSwitchNode':
                    r.refs()
                    r.u16()
                    idx = r.u32()
                    children = children[idx:idx + 1]
                for c in children:
                    visit(c, world)
            elif t in TRISHAPES:
                s = self._bstrishape(i, parent)
                if s is not None:
                    shapes.append(s)
            elif t in ('NiTriShape', 'NiTriStrips'):
                try:
                    s = self._nitrishape(i, parent)
                except (struct.error, IndexError, ValueError):
                    s = None
                if s is not None:
                    shapes.append(s)

        for root in self.roots:
            visit(root, np.eye(4))
        return shapes, points

    # -- geometry --------------------------------------------------------------------------------
    def _bstrishape(self, i, parent):
        t = self.types[i]
        r = R(self.blocks[i])
        name, flags, m = self._av(r)
        if flags & 1:
            return None
        r.f32(4)               # bounding sphere
        skin = r.i32()
        shader = r.i32()
        alpha = r.i32()
        desc = struct.unpack_from('<Q', r.b, r.o)[0]
        r.o += 8
        ntri = r.u16() if self.bs_ver < 130 else r.u32()
        nv = r.u16()
        dsize = r.u32()
        attrs = desc >> 44
        vsize = (desc & 0xF) * 4
        s = Shape()
        s.name = name
        if not self._material(s, shader, alpha):
            return None
        if skin >= 0:
            return None        # skinned geometry (NPC bodies, banners on skeletons) is dynamic
        vdata = None
        if dsize and nv and vsize:
            vdata = np.frombuffer(r.b, dtype=np.uint8, count=nv * vsize, offset=r.o).reshape(nv, vsize)
            r.o += nv * vsize
        tris = np.frombuffer(r.b, dtype='<u2', count=ntri * 3, offset=r.o).reshape(ntri, 3).astype(np.int32)
        r.o += ntri * 6
        if t == 'BSDynamicTriShape':
            dn = r.u32() // 16
            pos = np.frombuffer(r.b, dtype='<f4', count=dn * 4, offset=r.o).reshape(dn, 4)[:, :3]
        elif vdata is not None and attrs & 0x1:
            if attrs & 0x400 or self.bs_ver < 130:  # SSE stores full-precision positions
                pos = vdata[:, 0:12].copy().view('<f4').reshape(nv, 3)
            else:
                pos = vdata[:, 0:6].copy().view('<f2').reshape(nv, 3).astype(np.float32)
        else:
            return None
        if vdata is not None:
            if attrs & 0x2:
                uo = ((desc >> 8) & 0xF) * 4
                s.uv = vdata[:, uo:uo + 4].copy().view('<f2').reshape(nv, 2).astype(np.float32)
            if attrs & 0x8:
                no = ((desc >> 16) & 0xF) * 4
                s.normals = vdata[:, no:no + 3].astype(np.float32) / 127.5 - 1.0
            if attrs & 0x20:
                co = ((desc >> 24) & 0xF) * 4
                s.colors = vdata[:, co:co + 4].astype(np.float32) / 255.0
        if not s.use_vc:
            s.colors = None
        world = m @ parent
        s.verts = (np.hstack([pos.astype(np.float64), np.ones((len(pos), 1))]) @ world)[:, :3].astype(np.float32)
        if s.normals is not None:
            s.normals = (s.normals.astype(np.float64) @ world[:3, :3]).astype(np.float32)
        s.tris = tris
        if len(tris) == 0 or tris.max() >= len(s.verts):
            return None
        return s

    def _nitrishape(self, i, parent):
        r = R(self.blocks[i])
        name, flags, m = self._av(r)
        if flags & 1:
            return None
        data = r.i32()
        skin = r.i32()
        nmat = r.u32()
        r.o += 8 * nmat
        r.i32()
        r.u8()
        shader = r.i32()
        alpha = r.i32()
        s = Shape()
        s.name = name
        if skin >= 0 or not self._material(s, shader, alpha) or data < 0:
            return None
        d = R(self.blocks[data])
        d.i32()                # group id
        nv = d.u16()
        d.u8()
        d.u8()
        if not d.u8():
            return None
        pos = np.frombuffer(d.b, dtype='<f4', count=nv * 3, offset=d.o).reshape(nv, 3)
        d.o += nv * 12
        bsflags = d.u16()
        if self.bs_ver > 34:
            d.u32()            # material CRC
        if d.u8():
            s.normals = np.frombuffer(d.b, dtype='<f4', count=nv * 3, offset=d.o).reshape(nv, 3).copy()
            d.o += nv * 12
            if bsflags & 4096:
                d.o += nv * 24
        d.f32(4)
        if d.u8():
            s.colors = np.frombuffer(d.b, dtype='<f4', count=nv * 4, offset=d.o).reshape(nv, 4).copy()
            d.o += nv * 16
        if bsflags & 1:
            s.uv = np.frombuffer(d.b, dtype='<f4', count=nv * 2, offset=d.o).reshape(nv, 2).copy()
            d.o += nv * 8 * (bsflags & 63)
        d.u16()                # consistency
        d.i32()                # additional data
        if self.types[i] == 'NiTriShape':
            ntri = d.u16()
            d.u32()
            if not d.u8():
                return None
            tris = np.frombuffer(d.b, dtype='<u2', count=ntri * 3, offset=d.o).reshape(ntri, 3).astype(np.int32)
        else:
            d.u16()
            nstrips = d.u16()
            lens = [d.u16() for _ in range(nstrips)]
            if not d.u8():
                return None
            tl = []
            for ln in lens:
                st = np.frombuffer(d.b, dtype='<u2', count=ln, offset=d.o).astype(np.int32)
                d.o += ln * 2
                for k in range(ln - 2):
                    a, b, c = st[k], st[k + 1], st[k + 2]
                    if a != b and b != c and a != c:
                        tl.append((a, b, c) if k % 2 == 0 else (a, c, b))
            tris = np.array(tl, dtype=np.int32).reshape(-1, 3)
        if not s.use_vc:
            s.colors = None
        world = m @ parent
        s.verts = (np.hstack([pos.astype(np.float64), np.ones((nv, 1))]) @ world)[:, :3].astype(np.float32)
        if s.normals is not None:
            s.normals = (s.normals.astype(np.float64) @ world[:3, :3]).astype(np.float32)
        s.tris = tris
        if len(tris) == 0 or tris.max() >= nv:
            return None
        return s

    # -- materials -------------------------------------------------------------------------------
    def _material(self, s, shader, alpha):
        """Fill material fields; False if the shape should not take part in lighting."""
        if shader < 0:
            return False
        st = self.types[shader]
        if st != 'BSLightingShaderProperty':
            return False       # BSEffectShaderProperty, sky, water, ... are not lit surfaces
        r = R(self.blocks[shader])
        if self.bs_ver < 130:
            r.u32()            # shader type
        self._net(r)
        f1 = r.u32()
        f2 = r.u32()
        r.f32(4)               # uv offset / scale
        tset = r.i32()
        emit = r.f32(3)
        emult = r.f32()
        s.two_sided = bool(f2 & (1 << 4))
        s.decal = bool(f1 & ((1 << 26) | (1 << 27)))
        if f1 & ((1 << 15) | (1 << 16)):
            return False       # refraction / heat haze cards
        e = (emit[0] * emult, emit[1] * emult, emit[2] * emult)
        if (f1 & (1 << 22) or f2 & (1 << 6)) and max(e) > 0.02:
            s.emissive = e
        use_vc = bool(f2 & (1 << 5))
        if tset >= 0 and self.types[tset] == 'BSShaderTextureSet':
            t = R(self.blocks[tset])
            n = t.i32()
            texs = []
            for _ in range(n):
                ln = t.u32()
                texs.append(t.b[t.o:t.o + ln].decode('latin1').rstrip('\0'))
                t.o += ln
            if texs:
                s.diffuse = texs[0]
                if 'effects\\' in s.diffuse.lower() or s.diffuse.lower().endswith('_n.dds'):
                    return False   # fog / vapor cards that misuse the lit shader
            if len(texs) > 2 and f2 & (1 << 6):
                s.glow = texs[2]
        s.use_vc = use_vc
        if alpha >= 0 and self.types[alpha] == 'NiAlphaProperty':
            a = R(self.blocks[alpha])
            self._net(a)
            af = a.u16()
            s.alpha_blend = bool(af & 1)
            s.alpha_test = bool(af & 0x200)
        return True


def load(data):
    nif = Nif(data)
    return nif.walk()
