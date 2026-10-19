"""Radiance.esp: the light changes that go with the bake.

For every baked cell:
  * fill lights (fake bounce, replaced by the baked bounce) and daylight lights (fully baked) are listed in
    Radiance\lights.txt: the SKSE plugin hides them while baked lighting is on and shows them again when
    it is toggled off (F7), so the player can compare with the original setup; nothing goes into save games.
    Without the SKSE plugin they simply stay on (plain original lighting). Replaced lights with omni shadows get
    the no-shadow copy below (no shadow-map cost while hidden); replaced shadow spotlights (rare) are set
    "initially disabled".
  * fixture lights that cast omni/hemisphere shadows are switched to a no-shadow copy of their light record
    (shadow spotlights keep their shadows: without the flag the engine would turn them into omni lights)

Records are copied from the winning plugin with every form id remapped to this plugin's master list.
Only fields known to this writer are remapped; a record with any unknown field is left alone and logged,
so a patch can be incomplete but never points at the wrong form.
"""
import os
import struct

from .plugins import first, zstr

ESP_NAME = 'Radiance.esp'
FORM_VERSION = 44
SHADOW_OMNI = 0x800 | 0x1000      # hemisphere, omnidirectional
SHADOW_SPOT = 0x400

# subrecord -> how form ids are stored: list of byte offsets, 'array' (every 4 bytes), 'pairs8' (formid, float),
# 'lstring' (localized string id), None (no form ids)
COMMON = {'EDID': None, 'FULL': 'lstring', 'DATA': None}
REFR_FIELDS = dict(COMMON, **{
    'NAME': [0], 'XSCL': None, 'XRDS': None, 'XLIG': None, 'XEMI': [0], 'XESP': [0], 'XOWN': [0], 'XRNK': None,
    'XLCN': [0], 'XEZN': [0], 'XLRT': 'array', 'XLRL': [0], 'XLKR': [0, 4], 'XAPD': None, 'XAPR': 'pairs8',
    'XMBR': [0], 'XLIB': [0], 'XPWR': [0], 'XFVC': None, 'XPRM': None, 'XMBO': None, 'XHTW': None,
    'XLCM': None, 'XACT': None, 'XCNT': None, 'XTEL': [0], 'XNDP': [0], 'XLOC': [4], 'XRGD': None, 'XRGB': None,
    'XALP': None, 'XLTW': [0], 'XIS2': None, 'XTRI': None, 'ONAM': None, 'XMRK': None, 'FNAM': None,
    'TNAM': None, 'XLRM': [0], 'XRMR': None, 'LNAM': [0], 'INAM': [0], 'XPOD': [0, 4], 'XCVR': None,
    'XCVL': None, 'XCZR': [0], 'XCZA': None, 'XCZC': [0], 'XSPC': [0], 'XCHG': None,
})
CELL_FIELDS = dict(COMMON, **{
    'XCLC': None, 'XCLL': None, 'TVDT': None, 'MHDT': None, 'LTMP': [0], 'LNAM': [0], 'XCLW': None,
    'XNAM': None, 'XCLR': 'array', 'XLCN': [0], 'XWCN': None, 'XWCS': None, 'XWCU': None, 'XCWT': [0],
    'XOWN': [0], 'XRNK': None, 'XILL': [0], 'XWEM': None, 'XCCM': [0], 'XCAS': [0], 'XEZN': [0], 'XCMO': [0],
    'XCIM': [0],
})
LIGH_FIELDS = dict(COMMON, **{
    'OBND': None, 'MODL': None, 'MODT': None, 'ICON': None, 'MICO': None, 'FNAM': None, 'SNAM': [0],
    'YNAM': [0], 'ZNAM': [0], 'NAM0': None, 'LNAM': [0],
})


class Unsupported(Exception):
    pass


class Writer:
    def __init__(self, lo, log=print):
        self.lo = lo
        self.log = log
        self.order = {p.lname: i for i, p in enumerate(lo.plugins)}
        self.masters = []          # lower-case names, load order
        self.names = {p.lname: p.name for p in lo.plugins}
        self.collect = True

    # -- form ids ----------------------------------------------------------------------------------
    def origin_of(self, plugin, fid):
        if fid == 0:
            return None
        mi = fid >> 24
        return plugin.masters[mi] if mi < len(plugin.masters) else plugin.lname

    def remap(self, plugin, fid):
        origin = self.origin_of(plugin, fid)
        if origin is None:
            return 0
        if self.collect:
            if origin not in self.masters:
                self.masters.append(origin)
            return fid
        return (self.masters.index(origin) << 24) | (fid & 0xFFFFFF)

    def finalize_masters(self):
        self.masters.sort(key=lambda m: self.order.get(m, -1))
        self.collect = False

    # -- subrecords --------------------------------------------------------------------------------
    def convert(self, plugin, rtype, subs, table):
        out = bytearray()
        for t, d in subs:
            key = t.decode('latin1')
            if key not in table:
                raise Unsupported(f'{rtype} field {key}')
            how = table[key]
            d = bytearray(d)
            if how == 'lstring':
                if plugin.localized and len(d) == 4:
                    d = bytearray(self.lo.strings.get(plugin.name, struct.unpack('<I', d)[0]).encode('cp1252', 'replace')
                                  + b'\0')
            elif how == 'array':
                for o in range(0, len(d) - 3, 4):
                    struct.pack_into('<I', d, o, self.remap(plugin, struct.unpack_from('<I', d, o)[0]))
            elif how == 'pairs8':
                for o in range(0, len(d) - 7, 8):
                    struct.pack_into('<I', d, o, self.remap(plugin, struct.unpack_from('<I', d, o)[0]))
            elif how:
                for o in how:
                    if o + 4 <= len(d):
                        struct.pack_into('<I', d, o, self.remap(plugin, struct.unpack_from('<I', d, o)[0]))
            if len(d) > 0xFFFF:
                out += b'XXXX' + struct.pack('<HI', 4, len(d)) + t + struct.pack('<H', 0)
            else:
                out += t + struct.pack('<H', len(d))
            out += d
        return bytes(out)


def record(rtype, flags, fid, data):
    return rtype + struct.pack('<IIIIHH', len(data), flags & ~0x40000, fid, 0, FORM_VERSION, 0) + data


def group(label, gtype, content):
    lab = label if isinstance(label, bytes) else struct.pack('<I', label)
    return b'GRUP' + struct.pack('<I', 24 + len(content)) + lab + struct.pack('<iHHHH', gtype, 0, 0, 0, 0) + content


def build(sess, log=print):
    """Write <output>/Radiance.esp from the manifest's baked cells. Returns a summary dict."""
    lo = sess.lo
    man = sess.manifest()['cells']
    keep_shadows = int(sess.settings.get('keep_fixture_shadows', 0))
    plan = []        # (cell key, [(ref key, action)])
    hidden = []      # (cell id, ref id): replaced by the bake, hidden in game by the SKSE plugin
    for cid, ent in sorted(man.items()):
        lights = ent.get('lights')
        if not lights:
            continue
        local, plugin = cid.split(':', 1)
        key = (plugin, int(local, 16))
        if key not in lo.cells:
            continue
        if len(lights[0]) < 6:
            continue                              # baked by an older version: rebake
        acts = []
        shadowed = [L for L in lights if L[5] and L[2] & SHADOW_OMNI and not L[2] & SHADOW_SPOT and not L[4]]
        shadowed.sort(key=lambda L: -L[3])        # strongest first: keep those if asked to
        keep = {L[0] for L in shadowed[:keep_shadows]}
        for ref, cls, flags, power, keep_shadow, live in lights:
            if not live:                          # fill (replaced by baked bounce) or fully baked light
                if flags & SHADOW_SPOT:
                    acts.append((ref, 'disable'))
                    continue
                hidden.append((cid, ref))
                if flags & SHADOW_OMNI:
                    acts.append((ref, 'noshadow'))
            elif flags & SHADOW_OMNI and not flags & SHADOW_SPOT and not keep_shadow and ref not in keep:
                acts.append((ref, 'noshadow'))    # the bake measured that removing the shadow does not leak
        if acts:
            plan.append((key, acts))

    write_hidden(sess.out, hidden)
    w = Writer(lo, log)
    stats = {'cells': 0, 'disabled': 0, 'unshadowed': 0, 'skipped': 0, 'hidden': len(hidden)}
    for pass_ in (0, 1):
        if pass_ == 1:
            w.finalize_masters()
        new_ligh = {}           # base key -> (new local id, record bytes)
        cells_out = []
        for key, acts in plan:
            groups = {}
            refs = lo.cell_refs(key, groups)
            cp, csubs = lo.cell_record(key)
            pi, coff = lo.cells[key].versions[-1]
            _, cflags, cfid, _ = cp.record(coff)
            try:
                cell_data = w.convert(cp, 'CELL', csubs, CELL_FIELDS)
            except Unsupported as e:
                if pass_:
                    log(f'  patch: skipped cell {local_str(key)} ({e})')
                    stats['skipped'] += 1
                continue
            ref_recs = {8: [], 9: []}
            for refstr, action in acts:
                local, plugin = refstr.split(':', 1)
                rk = (plugin, int(local, 16))
                hit = refs.get(rk)
                if hit is None:
                    continue
                p, rflags, subs, raw = hit
                fid = struct.unpack_from('<I', raw, 12)[0]
                try:
                    if action == 'noshadow':
                        name = first(subs, b'NAME')
                        bkey = p.key(struct.unpack('<I', name)[0])
                        if bkey not in new_ligh:
                            new_ligh[bkey] = make_noshadow(w, lo, bkey, 0x800 + len(new_ligh))
                        if new_ligh[bkey] is None:
                            continue
                        data = convert_ref_newbase(w, p, subs, new_ligh[bkey][0])
                        if pass_:
                            stats['unshadowed'] += 1
                    else:
                        data = w.convert(p, 'REFR', subs, REFR_FIELDS)
                        rflags |= 0x800
                        if pass_:
                            stats['disabled'] += 1
                except Unsupported as e:
                    if pass_:
                        log(f'  patch: skipped ref {refstr} ({e})')
                        stats['skipped'] += 1
                    continue
                ref_recs[groups.get(rk, 9) if groups.get(rk) in (8, 9) else 9].append(
                    record(b'REFR', rflags, w.remap(p, fid), data))
            if not ref_recs[8] and not ref_recs[9]:
                continue
            cells_out.append((w.remap(cp, cfid), cflags, cell_data, ref_recs))
        if pass_ == 1:
            break

    # ---- assemble ----------------------------------------------------------------------------------
    self_idx = len(w.masters)
    body = b''
    nrec = 0
    if new_ligh:
        recs = b''
        for bkey, val in new_ligh.items():
            if val is None:
                continue
            rec = bytearray(val[1])
            struct.pack_into('<I', rec, 12, (self_idx << 24) | val[0])
            recs += bytes(rec)
            nrec += 1
        body += group(b'LIGH', 0, recs)
        nrec += 1
    blocks = {}
    for cfid, cflags, cdata, ref_recs in cells_out:
        obj = cfid & 0xFFFFFF
        blocks.setdefault(obj % 10, {}).setdefault((obj // 10) % 10, []).append((cfid, cflags, cdata, ref_recs))
    cell_top = b''
    for b in sorted(blocks):
        sub_content = b''
        for sb in sorted(blocks[b]):
            content = b''
            for cfid, cflags, cdata, ref_recs in blocks[b][sb]:
                content += record(b'CELL', cflags, cfid, cdata)
                nrec += 1
                children = b''
                for gtype in (8, 9):
                    if ref_recs[gtype]:
                        children += group(cfid, gtype, b''.join(ref_recs[gtype]))
                        nrec += 1 + len(ref_recs[gtype])
                content += group(cfid, 6, children)
                nrec += 1
                stats['cells'] += 1
            sub_content += group(sb, 3, content)
            nrec += 1
        cell_top += group(b, 2, sub_content)
        nrec += 1
    if cell_top:
        body += group(b'CELL', 0, cell_top)
        nrec += 1

    hdr = b'HEDR' + struct.pack('<HfII', 12, 1.71, nrec, 0x800 + len(new_ligh))
    hdr += sub(b'CNAM', b'Radiance\0')
    hdr += sub(b'SNAM', b'Generated by the Radiance baker. Regenerate after load order changes.\0')
    for m in w.masters:
        hdr += sub(b'MAST', w.names.get(m, m).encode('cp1252') + b'\0') + sub(b'DATA', b'\0' * 8)
    tes4 = b'TES4' + struct.pack('<IIIIHH', len(hdr), 0x200, 0, 0, FORM_VERSION, 0) + hdr
    path = os.path.join(sess.out, ESP_NAME)
    if not cells_out and not new_ligh:
        if os.path.isfile(path):
            os.remove(path)
        log(f"Light patch: no plugin needed; {stats['hidden']} lights replaced by the bake (hidden in game).")
        return stats
    with open(path, 'wb') as f:
        f.write(tes4 + body)
    stats['masters'] = len(w.masters)
    stats['new_lights'] = sum(1 for v in new_ligh.values() if v)
    log(f"Light patch: {ESP_NAME} - {stats['cells']} cell(s), {stats['hidden']} lights replaced by the bake (hidden in "
        f"game, back on with the toggle key), {stats['disabled']} shadow spotlights turned off, "
        f"{stats['unshadowed']} lights without shadows, {stats['skipped']} skipped, {stats['masters']} masters")
    return stats


def write_hidden(out, hidden):
    """Radiance\\lights.txt for the SKSE plugin: 'cell <plugin>/<id>' lines, each followed by its lights."""
    path = os.path.join(out, 'Radiance', 'lights.txt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ['# lights replaced by the bake: hidden while baked lighting is on (generated, do not edit)']
    last = None
    for cid, ref in sorted(hidden):
        if cid != last:
            lines.append('cell ' + key_str(cid))
            last = cid
        lines.append(key_str(ref))
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    os.replace(tmp, path)


def key_str(fk):
    """'01605E:Skyrim.esm' -> 'skyrim.esm/01605E' (the SKSE plugin's form key format)."""
    local, plugin = fk.split(':', 1)
    return f'{plugin.lower()}/{int(local, 16) & 0xFFFFFF:06X}'


def sub(t, d):
    return t + struct.pack('<H', len(d)) + d


def local_str(key):
    return f'{key[1]:06X}:{key[0]}'


def make_noshadow(w, lo, bkey, local_id):
    """Copy of a LIGH base without shadow flags. Form id is patched in at assembly (placeholder 0xFFFFFFFF)."""
    br = lo.base_record(bkey)
    if br is None:
        return None
    p, typ, flags, subs = br
    if typ != b'LIGH':
        return None
    out = []
    for t, d in subs:
        if t == b'EDID':
            d = zstr(d).encode('cp1252', 'replace')[:120] + b'_BLNS\0'
        elif t == b'DATA':
            d = bytearray(d)
            struct.pack_into('<I', d, 12, struct.unpack_from('<I', d, 12)[0] & ~(SHADOW_OMNI | SHADOW_SPOT))
            d = bytes(d)
        out.append((t, d))
    try:
        data = w.convert(p, 'LIGH', out, LIGH_FIELDS)
    except Unsupported as e:
        w.log(f'  patch: cannot copy light {local_str(bkey)} ({e})')
        return None
    return local_id, record(b'LIGH', flags, 0xFFFFFFFF, data)


def convert_ref_newbase(w, p, subs, new_local):
    """REFR whose NAME points at one of our own new records: convert the rest normally."""
    rest = [(t, d) for t, d in subs if t != b'NAME']
    data = w.convert(p, 'REFR', rest, REFR_FIELDS)
    self_fid = 0 if w.collect else (len(w.masters) << 24) | new_local
    name = sub(b'NAME', struct.pack('<I', self_fid))
    return _insert_after_edid(data, name) if data.startswith(b'EDID') else name + data


def _insert_after_edid(data, name_sub):
    ln = struct.unpack_from('<H', data, 4)[0]
    return data[:6 + ln] + name_sub + data[6 + ln:]
