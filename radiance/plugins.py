"""Skyrim SE/VR plugin reader: interior cell index, winning records, placed references.

Form keys are load-order independent: (origin plugin name lower-case, 24-bit local id).
"""
import hashlib
import mmap
import os
import pickle
import struct
import zlib

# Base record types whose placed references can occlude or emit light.
BASE_TYPES = {b'STAT', b'MSTT', b'FURN', b'CONT', b'DOOR', b'LIGH', b'ACTI', b'FLOR', b'TREE', b'SCOL',
              b'MOVS', b'MISC', b'TACT', b'IDLM', b'BOOK', b'ALCH', b'INGR', b'WEAP', b'ARMO', b'APPA',
              b'SLGM', b'KEYM', b'AMMO', b'LGTM'}       # LGTM = lighting template (cells point to it with LTMP)
INDEX_VERSION = 4
OWN_PLUGIN = 'radiance.esp'   # the baker's output plugin never counts as input


def fk_str(key):
    return f'{key[1]:06X}:{key[0]}'


class Plugin:
    def __init__(self, name, path):
        self.name = name
        self.lname = name.lower()
        self.path = path
        self._f = open(path, 'rb')
        self.mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        typ, size, flags = struct.unpack_from('<4sII', self.mm, 0)
        assert typ == b'TES4', f'{name}: not a plugin'
        self.flags = flags
        self.localized = bool(flags & 0x80)
        self.masters = []
        for t, d in subrecords(bytes(self.mm[24:24 + size])):
            if t == b'MAST':
                self.masters.append(d[:-1].decode('cp1252').lower())
        self.header_end = 24 + size

    def key(self, formid):
        mi = formid >> 24
        origin = self.masters[mi] if mi < len(self.masters) else self.lname
        return (origin, formid & 0xFFFFFF)

    def record(self, off):
        """-> (type, flags, formid, data) at file offset."""
        typ, size, flags, fid = struct.unpack_from('<4sIII', self.mm, off)
        data = self.mm[off + 24:off + 24 + size]
        if flags & 0x40000:
            data = zlib.decompress(data[4:])
        return typ, flags, fid, data

    def raw_record(self, off):
        size = struct.unpack_from('<I', self.mm, off + 4)[0]
        return self.mm[off:off + 24 + size]

    def close(self):
        self.mm.close()
        self._f.close()


def subrecords(data):
    out = []
    o = 0
    n = len(data)
    big = None
    while o + 6 <= n:
        t = data[o:o + 4]
        sz = data[o + 4] | (data[o + 5] << 8)
        o += 6
        if t == b'XXXX':
            big = struct.unpack_from('<I', data, o)[0]
            o += sz
            continue
        if big is not None:
            sz, big = big, None
        out.append((t, data[o:o + sz]))
        o += sz
    return out


def first(subs, typ):
    for t, d in subs:
        if t == typ:
            return d
    return None


def zstr(b):
    if b is None:
        return ''
    return b.split(b'\0', 1)[0].decode('cp1252', 'replace')


class StringTable:
    """Localized names (FULL) from strings\\<plugin>_english.strings."""

    def __init__(self, vfs):
        self.vfs = vfs
        self.cache = {}

    def get(self, plugin, sid, lang='english'):
        tab = self.cache.get(plugin)
        if tab is None:
            tab = {}
            raw = self.vfs.read(f'strings\\{os.path.splitext(plugin)[0]}_{lang}.strings')
            if raw:
                cnt, _ = struct.unpack_from('<II', raw, 0)
                base = 8 + cnt * 8
                for i in range(cnt):
                    s_id, off = struct.unpack_from('<II', raw, 8 + i * 8)
                    end = raw.index(b'\0', base + off)
                    tab[s_id] = raw[base + off:end].decode('utf-8', 'replace')
            self.cache[plugin] = tab
        return tab.get(sid, '')


class CellInfo:
    __slots__ = ('key', 'versions', 'children')

    def __init__(self, key):
        self.key = key
        self.versions = []   # (plugin idx, record offset)
        self.children = []   # (plugin idx, group offset, group size)


class LoadOrder:
    def __init__(self, vfs, cache_dir=None, log=print):
        self.vfs = vfs
        self.log = log
        self.plugins = []
        for name in vfs.plugins:
            if name.lower() == OWN_PLUGIN:
                continue
            path = vfs.plugin_path(name)
            if path is None:
                log(f'  missing plugin file: {name}')
                continue
            self.plugins.append(Plugin(name, path))
        self.pidx = {p.lname: i for i, p in enumerate(self.plugins)}
        self.strings = StringTable(vfs)
        sig = hashlib.sha1(repr([INDEX_VERSION] + [(p.lname, os.path.getsize(p.path), int(os.path.getmtime(p.path)))
                                                   for p in self.plugins]).encode()).hexdigest()
        cache = os.path.join(cache_dir, f'index-{sig[:16]}.pkl') if cache_dir else None
        loaded = False
        if cache and os.path.isfile(cache):
            try:
                with open(cache, 'rb') as f:
                    self.base, self.cells = pickle.load(f)
                loaded = True
            except Exception:          # noqa: BLE001 - stale or damaged cache: rebuild it
                pass
        if not loaded:
            self._build()
            if cache:
                os.makedirs(cache_dir, exist_ok=True)
                for old in os.listdir(cache_dir):
                    if old.startswith('index-'):
                        os.remove(os.path.join(cache_dir, old))
                with open(cache, 'wb') as f:
                    pickle.dump((self.base, self.cells), f, protocol=pickle.HIGHEST_PROTOCOL)

    # -- index -----------------------------------------------------------------------------------
    def _build(self):
        self.base = {}   # key -> (plugin idx, offset)   (winner = last plugin)
        cells = {}
        for pi, p in enumerate(self.plugins):
            mm = p.mm
            o = p.header_end
            end = len(mm)
            while o + 24 <= end:
                typ, gsize, label = struct.unpack_from('<4sI4s', mm, o)
                if typ != b'GRUP':
                    break
                if label in BASE_TYPES:
                    r = o + 24
                    ge = o + gsize
                    while r < ge:
                        rt, rs, _, fid = struct.unpack_from('<4sIII', mm, r)
                        if rt == b'GRUP':
                            r += rs
                            continue
                        self.base[p.key(fid)] = (pi, r)
                        r += 24 + rs
                elif label == b'CELL':
                    self._index_cells(pi, p, o + 24, o + gsize, cells)
                o += gsize
        self.cells = cells

    def _index_cells(self, pi, p, o, end, cells):
        mm = p.mm
        while o < end:
            typ, size, a, b = struct.unpack_from('<4sIII', mm, o)
            if typ == b'GRUP':
                gtype = struct.unpack_from('<i', mm, o + 12)[0]
                if gtype in (2, 3):          # interior block / sub-block: descend
                    o += 24
                    continue
                if gtype == 6:               # cell children
                    key = p.key(a)
                    c = cells.get(key)
                    if c is None:
                        c = cells[key] = CellInfo(key)
                    c.children.append((pi, o, size))
                o += size
                continue
            if typ == b'CELL':
                key = p.key(b)
                c = cells.get(key)
                if c is None:
                    c = cells[key] = CellInfo(key)
                c.versions.append((pi, o))
            o += 24 + size

    # -- records ---------------------------------------------------------------------------------
    def base_record(self, key):
        hit = self.base.get(key)
        if hit is None:
            return None
        pi, off = hit
        p = self.plugins[pi]
        typ, flags, fid, data = p.record(off)
        return p, typ, flags, subrecords(data)

    def base_raw(self, key):
        hit = self.base.get(key)
        if hit is None:
            return b''
        return bytes(self.plugins[hit[0]].raw_record(hit[1]))

    def full_name(self, p, subs):
        d = first(subs, b'FULL')
        if d is None:
            return ''
        if p.localized and len(d) == 4:
            return self.strings.get(p.name, struct.unpack('<I', d)[0])
        return zstr(d)

    def cell_summary(self, key):
        c = self.cells[key]
        pi, off = c.versions[-1] if c.versions else (None, None)
        if pi is None:
            return None
        p = self.plugins[pi]
        typ, flags, fid, data = p.record(off)
        subs = subrecords(data)
        dflags = first(subs, b'DATA')
        interior = bool(dflags and dflags[0] & 1)
        return {
            'key': key, 'edid': zstr(first(subs, b'EDID')), 'name': self.full_name(p, subs),
            'plugin': p.name, 'origin': key[0], 'interior': interior, 'deleted': bool(flags & 0x20),
            'overrides': [self.plugins[v[0]].name for v in c.versions],
        }

    def list_interiors(self):
        out = []
        for key in self.cells:
            s = self.cell_summary(key)
            if s and s['interior'] and not s['deleted']:
                out.append(s)
        out.sort(key=lambda s: (s['name'] or s['edid']).lower())
        return out

    def cell_record(self, key):
        pi, off = self.cells[key].versions[-1]
        p = self.plugins[pi]
        typ, flags, fid, data = p.record(off)
        return p, subrecords(data)

    def cell_refs(self, key, groups=None):
        """Winning version of every reference placed in the cell: key -> (plugin, flags, subrecords, raw).
        groups (optional dict) receives key -> child group type (8 persistent, 9 temporary)."""
        refs = {}
        for pi, goff, gsize in self.cells[key].children:
            p = self.plugins[pi]
            mm = p.mm
            o = goff + 24
            end = goff + gsize
            gtype = 9
            while o < end:
                typ, size = struct.unpack_from('<4sI', mm, o)
                if typ == b'GRUP':
                    gtype = struct.unpack_from('<i', mm, o + 12)[0]
                    o += 24
                    continue
                if typ == b'REFR':
                    rt, flags, fid, data = p.record(o)
                    k = p.key(fid)
                    refs[k] = (p, flags, subrecords(data), bytes(p.raw_record(o)))
                    if groups is not None:
                        groups[k] = gtype
                o += 24 + size
        return refs

    def cell_fingerprint_parts(self, key):
        """Raw bytes of every record version that defines the cell and its contents (in load order)."""
        h = hashlib.sha1()
        c = self.cells[key]
        for pi, off in c.versions:
            h.update(self.plugins[pi].lname.encode())
            h.update(self.plugins[pi].raw_record(off))
        for pi, goff, gsize in c.children:
            h.update(self.plugins[pi].lname.encode())
            h.update(self.plugins[pi].mm[goff:goff + gsize])
        return h
