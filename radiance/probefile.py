"""Probe volume file (.rdp), one per cell.

Little-endian. Version 2 (sparse bricks):
  char[4]  'RDNC'
  u32      version (2)
  u16+utf8 origin plugin of the cell form (lower case)
  u32      cell local form id (24 bits; the load-order index is resolved at runtime)
  f32[3]   grid origin = world position of probe (0,0,0) (interior cell space = world space)
  f32      spacing (units between probes)
  u32[3]   grid dims (probes)
  u32      layers (0 = fixtures bounce, 1 = daylight direct + bounce)
  u32      flags (reserved)
  u8[20]   sha1 fingerprint of everything the bake depended on
  u32[2]   rays per probe, bounces
  u32      brick size in grid cells (4; a brick stores (size+1)^3 probes including its shared border)
  u32[3]   brick grid dims (bx, by, bz)
  u32[3]   atlas size in bricks (ax, ay, az)
  u32      brick count
  f32      brightness calibration applied to the data (informational)
  f32      share of the original flat ambient to keep on top of the baked light (per cell)
  u64[2]   payload size uncompressed, compressed (flags bit 1: payload is zlib-compressed)
  payload (zlib):
    u32[bz][by][bx]        indirection: brick id + 1, 0 = no data (game uses its own ambient there)
    for each layer, for each color channel (r, g, b):
       f16[az*P][ay*P][ax*P][4] atlas, P = size + 1, texel = (a, bx, by, bz):
                                irradiance(n) = max(0, a + bx*nx + by*ny + bz*nz) in game light units
Most of a volume is empty (no daylight layer, atlas padding, probes inside walls): zlib shrinks it ~6x.
Brick i sits at atlas brick (i % ax, (i / ax) % ay, i / (ax*ay)).
"""
import struct
import zlib

import numpy as np

MAGIC = b'RDNC'
VERSION = 2
FLAG_ZLIB = 2


def write(path, cell_key, origin, spacing, dims, fingerprint, rays, bounces, brick, indirection, atlas, atlas_bricks,
          calibration, floor):
    nl = atlas.shape[0]
    plug = cell_key[0].encode('utf-8')
    bz, by, bx = indirection.shape
    with open(path, 'wb') as f:
        f.write(MAGIC + struct.pack('<I', VERSION))
        f.write(struct.pack('<H', len(plug)) + plug + struct.pack('<I', cell_key[1]))
        f.write(struct.pack('<3f', *map(float, origin)) + struct.pack('<f', float(spacing)))
        f.write(struct.pack('<3I', *map(int, dims)) + struct.pack('<II', nl, FLAG_ZLIB))
        f.write(bytes.fromhex(fingerprint)[:20].ljust(20, b'\0'))
        f.write(struct.pack('<II', rays, bounces))
        f.write(struct.pack('<I', brick) + struct.pack('<3I', bx, by, bz) + struct.pack('<3I', *atlas_bricks))
        f.write(struct.pack('<I', int(indirection.max(initial=0))) + struct.pack('<ff', float(calibration), float(floor)))
        payload = np.ascontiguousarray(indirection, '<u4').tobytes() + np.ascontiguousarray(atlas, '<f2').tobytes()
        packed = zlib.compress(payload, 6)
        f.write(struct.pack('<QQ', len(payload), len(packed)))
        f.write(packed)


def read(path):
    with open(path, 'rb') as f:
        d = f.read()
    if d[:4] != MAGIC:
        raise ValueError('not a probe file')
    version = struct.unpack_from('<I', d, 4)[0]
    if version != VERSION:
        raise ValueError(f'probe file version {version}, expected {VERSION} - rebake')
    o = 8
    n = struct.unpack_from('<H', d, o)[0]
    o += 2
    plugin = d[o:o + n].decode()
    o += n
    local = struct.unpack_from('<I', d, o)[0]
    o += 4
    origin = np.array(struct.unpack_from('<3f', d, o))
    o += 12
    spacing = struct.unpack_from('<f', d, o)[0]
    o += 4
    dims = struct.unpack_from('<3I', d, o)
    o += 12
    nl, flags = struct.unpack_from('<II', d, o)
    o += 8
    fp = d[o:o + 20].hex()
    o += 20
    rays, bounces = struct.unpack_from('<II', d, o)
    o += 8
    brick = struct.unpack_from('<I', d, o)[0]
    o += 4
    bx, by, bz = struct.unpack_from('<3I', d, o)
    o += 12
    ax, ay, az = struct.unpack_from('<3I', d, o)
    o += 12
    nbricks, calib, floor = struct.unpack_from('<Iff', d, o)
    o += 12
    raw_size, packed_size = struct.unpack_from('<QQ', d, o)
    o += 16
    if flags & FLAG_ZLIB:
        d = zlib.decompress(d[o:o + packed_size])
        o = 0
    ind = np.frombuffer(d, '<u4', bx * by * bz, o).reshape(bz, by, bx)
    o += bx * by * bz * 4
    p = brick + 1
    shape = (nl, 3, az * p, ay * p, ax * p, 4)
    atlas = np.frombuffer(d, '<f2', int(np.prod(shape)), o).reshape(shape).astype(np.float32)
    return {'cell': (plugin, local), 'origin': origin, 'spacing': spacing, 'dims': dims, 'layers': nl,
            'fingerprint': fp, 'rays': rays, 'bounces': bounces, 'brick': brick, 'indirection': ind,
            'atlas': atlas, 'atlas_bricks': (ax, ay, az), 'bricks': nbricks, 'calibration': calib, 'floor': floor}


def sample(vol, p, n, layer_w=(1.0, 1.0), offset=24.0):
    """Reference implementation of the shader lookup (for tests)."""
    q = (np.asarray(p, float) + np.asarray(n, float) * offset - vol['origin']) / vol['spacing']
    dims = np.array(vol['dims'])
    q = np.clip(q, 0, dims - 1)
    bs = vol['brick']
    bdims = np.array(vol['indirection'].shape[::-1])
    b = np.minimum((q // bs).astype(int), bdims - 1)
    bid = int(vol['indirection'][b[2], b[1], b[0]])
    if bid == 0:
        return None
    bid -= 1
    ax, ay, az = vol['atlas_bricks']
    cell = np.array([bid % ax, (bid // ax) % ay, bid // (ax * ay)])
    local = q - b * bs
    t = cell * (bs + 1) + local
    i0 = np.minimum(np.floor(t).astype(int), cell * (bs + 1) + bs - 1)
    f = t - i0
    out = np.zeros(3)
    basis = np.array([1.0, *n])
    for li in range(vol['layers']):
        acc = np.zeros(3)
        for c in range(8):
            o = np.array([c & 1, (c >> 1) & 1, (c >> 2) & 1])
            w = np.prod(np.where(o, f, 1 - f))
            x, y, z = i0 + o
            for ch in range(3):
                acc[ch] += w * vol['atlas'][li, ch, z, y, x] @ basis
        out += np.maximum(acc, 0) * layer_w[li]
    return out
