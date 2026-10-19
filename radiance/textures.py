"""Diffuse textures at low resolution: decode only one small mip level of a DDS."""
import io
import struct

import numpy as np
from PIL import Image

TEX = 128  # every texture is resampled to TEX x TEX RGBA for the tracer

# bytes per 4x4 block for block-compressed formats, or bytes per pixel (negative) for plain ones
FOURCC = {b'DXT1': 8, b'DXT3': 16, b'DXT5': 16, b'ATI1': 8, b'BC4U': 8, b'ATI2': 16, b'BC5U': 16}
DXGI = {71: 8, 72: 8, 74: 16, 75: 16, 77: 16, 78: 16, 80: 8, 81: 8, 83: 16, 84: 16, 98: 16, 99: 16,
        28: -4, 29: -4, 87: -4, 88: -4, 91: -4, 61: -1}


def _level_size(w, h, bpb):
    if bpb < 0:
        return w * h * -bpb
    return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * bpb


def decode_small(data, target=TEX):
    """-> float32 (TEX, TEX, 4) in 0..1, or None."""
    if data is None or data[:4] != b'DDS ':
        return None
    h, w, _, _, mips = struct.unpack_from('<IIIII', data, 12)
    pf_flags, fourcc, bits = struct.unpack_from('<I4sI', data, 80)
    hdr = 128
    dxgi = None
    if pf_flags & 0x4 and fourcc == b'DX10':
        dxgi = struct.unpack_from('<I', data, 128)[0]
        hdr = 148
        bpb = DXGI.get(dxgi)
    elif pf_flags & 0x4:
        bpb = FOURCC.get(fourcc)
    else:
        bpb = -(bits // 8) if bits in (24, 32) else None
    if bpb is None:
        return None
    mips = max(1, mips)
    off = hdr
    lw, lh = w, h
    level = 0
    while level + 1 < mips and min(lw, lh) // 2 >= target:
        off += _level_size(lw, lh, bpb)
        lw, lh = max(1, lw // 2), max(1, lh // 2)
        level += 1
    size = _level_size(lw, lh, bpb)
    body = data[off:off + size]
    if len(body) < size:
        return None
    # rebuild a single-level DDS so Pillow only decodes the small mip
    head = bytearray(data[:hdr])
    struct.pack_into('<II', head, 12, lh, lw)
    struct.pack_into('<I', head, 28, 1)
    struct.pack_into('<I', head, 20, size if bpb > 0 else lw * -bpb)
    struct.pack_into('<I', head, 8, struct.unpack_from('<I', head, 8)[0] & ~0x20000)
    if hdr == 148:
        struct.pack_into('<I', head, 140, 1)  # array size
    try:
        img = Image.open(io.BytesIO(bytes(head) + body))
        img = img.convert('RGBA')
    except Exception:
        return None
    if img.size != (target, target):
        img = img.resize((target, target), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


class TextureBank:
    """Collects textures for one scene into a single array the tracer can index."""

    def __init__(self, vfs):
        self.vfs = vfs
        self.ids = {}
        self.images = [np.ones((TEX, TEX, 4), np.float32) * np.float32(0.5)]  # 0 = missing (mid grey)
        self.missing = set()

    def get(self, path):
        if not path:
            return 0
        k = path.lower().replace('/', '\\')
        if not k.startswith('textures\\'):
            k = 'textures\\' + k
        tid = self.ids.get(k)
        if tid is not None:
            return tid
        img = decode_small(self.vfs.read(k))
        if img is None:
            self.missing.add(k)
            tid = 0
        else:
            tid = len(self.images)
            self.images.append(img)
        self.ids[k] = tid
        return tid

    def array(self):
        return np.ascontiguousarray(np.stack(self.images))
