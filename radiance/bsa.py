"""Read-only Bethesda archive reader: BSA v104 (Oldrim, zlib) and v105 (SSE/VR, LZ4 frame)."""
import os
import struct
import zlib

import numpy as np
from numba import njit


@njit(cache=True)
def _lz4_blocks(src, dst):
    """Decode an LZ4 frame body (blocks after the frame header) into dst. Returns bytes written."""
    sp = 0
    dp = 0
    n = src.shape[0]
    while sp + 4 <= n:
        bsize = src[sp] | (src[sp + 1] << 8) | (src[sp + 2] << 16) | (np.uint32(src[sp + 3]) << 24)
        sp += 4
        if bsize == 0:
            break
        raw = (bsize & 0x80000000) != 0
        bsize &= 0x7FFFFFFF
        end = sp + bsize
        if raw:
            for i in range(bsize):
                dst[dp + i] = src[sp + i]
            dp += bsize
            sp = end
            continue
        while sp < end:
            token = src[sp]
            sp += 1
            lit = token >> 4
            if lit == 15:
                while True:
                    b = src[sp]
                    sp += 1
                    lit += b
                    if b != 255:
                        break
            for i in range(lit):
                dst[dp + i] = src[sp + i]
            dp += lit
            sp += lit
            if sp >= end:
                break
            off = src[sp] | (src[sp + 1] << 8)
            sp += 2
            ml = token & 15
            if ml == 15:
                while True:
                    b = src[sp]
                    sp += 1
                    ml += b
                    if b != 255:
                        break
            ml += 4
            st = dp - off
            for i in range(ml):
                dst[dp + i] = dst[st + i]
            dp += ml
    return dp


def lz4_frame_decompress(data, out_size):
    if struct.unpack_from('<I', data, 0)[0] != 0x184D2204:
        raise ValueError('not an LZ4 frame')
    flg = data[4]
    o = 6 + (8 if flg & 0x08 else 0) + 1  # FLG, BD, [content size], header checksum
    if flg & 0x10:
        raise ValueError('LZ4 block checksums not supported')
    src = np.frombuffer(data, dtype=np.uint8, offset=o)
    dst = np.empty(out_size + 64, dtype=np.uint8)
    n = _lz4_blocks(src, dst)
    return dst[:n].tobytes()


class BSA:
    def __init__(self, path):
        self.path = path
        self.files = {}  # 'folder\\name' (lower) -> (offset, size, compressed)
        with open(path, 'rb') as f:
            hdr = f.read(36)
            magic, ver, _, aflags, nfold, nfile, _, fnlen, _ = struct.unpack('<4sIIIIIIIH2x', hdr)
            if magic != b'BSA\0' or ver not in (103, 104, 105):
                raise ValueError(f'unsupported archive {path}')
            self.version = ver
            self.compressed = bool(aflags & 0x4)
            self.embed_names = bool(aflags & 0x100) and ver != 103
            rec = 24 if ver == 105 else 16
            folders = []
            raw = f.read(rec * nfold)
            for i in range(nfold):
                if ver == 105:
                    _, cnt, _, _ = struct.unpack_from('<QIIQ', raw, i * rec)
                else:
                    _, cnt, _ = struct.unpack_from('<QII', raw, i * rec)
                folders.append(cnt)
            entries = []
            for cnt in folders:
                ln = f.read(1)[0]
                fname = f.read(ln)[:-1].decode('cp1252').lower()
                fr = f.read(16 * cnt)
                for j in range(cnt):
                    _, size, off = struct.unpack_from('<QII', fr, j * 16)
                    entries.append((fname, size, off))
            names = f.read(fnlen).split(b'\0')[:nfile]
        for (folder, size, off), nm in zip(entries, names):
            comp = self.compressed != bool(size & 0x40000000)
            self.files[folder + '\\' + nm.decode('cp1252').lower()] = (off, size & 0x3FFFFFFF, comp)

    def read(self, key):
        off, size, comp = self.files[key]
        with open(self.path, 'rb') as f:
            f.seek(off)
            data = f.read(size)
        p = 0
        if self.embed_names:
            p = 1 + data[0]
        if not comp:
            return data[p:]
        orig = struct.unpack_from('<I', data, p)[0]
        body = data[p + 4:]
        if self.version == 105:
            return lz4_frame_decompress(body, orig)
        return zlib.decompress(body)


def norm(path):
    return path.replace('/', '\\').lower().lstrip('\\')
