"""Patch the load order's Community Shaders Lighting.hlsl (any fork/version) to read the baked probes.

The patch is two text edits, so it follows whatever CS build the user has:
  1. include our Radiance.hlsli after the shader's own includes
  2. wrap the ambient term  mul(DirectionalAmbient, float4(ambientNormal, 1.0))
"""
import os
import re

ANCHOR = 'mul(DirectionalAmbient, float4(ambientNormal, 1.0))'
MARKER = '// [Radiance patch]'
PATCHED_MARKERS = (MARKER, '// [Baked Lighting patch]', 'Radiance::Ambient(', 'BakedLighting::Ambient(')
INCLUDE_ANCHOR = '#include "Common/SharedData.hlsli"'
HERE = os.path.dirname(os.path.abspath(__file__))
# Radiance.hlsli resources, in the order Radiance.ini ShaderSlots lists them for the SKSE plugin
SLOT_NAMES = ('R', 'G', 'B', 'PARAMS', 'INDIRECTION')
MAX_SLOT = 127                    # D3D11: 128 shader resource slots per stage
REGISTER = re.compile(r'register\s*\(\s*t(\d+)\s*\)')


def _shader_roots(sess):
    """Folders whose Shaders\ the game sees (load order, low to high), without our output mod."""
    from .vfs import VFS
    out = os.path.normcase(os.path.abspath(sess.out))
    inst = getattr(sess, 'instance', None)
    roots = VFS.from_mo2(inst, sess.profile).roots if inst else sess.vfs.roots
    return [(n, f) for n, f in roots if os.path.normcase(os.path.abspath(f)) != out]


def used_slots(sess):
    """Every t-register any installed shader source declares (all Community Shaders variants and their features)."""
    used = set()
    for name, folder in _shader_roots(sess):
        base = os.path.join(folder, 'Shaders')
        if not os.path.isdir(base):
            continue
        for dp, dn, fn in os.walk(base):
            for f in fn:
                if f.lower().endswith(('.hlsl', '.hlsli', '.fx', '.h')):
                    try:
                        with open(os.path.join(dp, f), encoding='utf-8', errors='replace') as fh:
                            used.update(int(m.group(1)) for m in REGISTER.finditer(fh.read()))
                    except OSError:
                        pass
    return used


def pick_slots(used):
    """The highest free texture slots, one per resource. Forks differ (Open Shaders uses most slots above t70),
    so they are chosen per load order instead of fixed."""
    free = [t for t in range(MAX_SLOT, -1, -1) if t not in used]
    if len(free) < len(SLOT_NAMES):
        raise RuntimeError('no free shader texture slots left in this Community Shaders setup')
    return sorted(free[:len(SLOT_NAMES)])


def include_text(slots):
    with open(os.path.join(HERE, 'shader', 'Radiance.hlsli'), encoding='utf-8') as f:
        text = f.read()
    for name, slot in zip(SLOT_NAMES, slots):
        text = text.replace(f'@SLOT_{name}@', str(slot))
    return text


def find_source(sess):
    """-> (text, provider) of the Lighting.hlsl that the game would use without our output mod."""
    from .vfs import VFS
    out = os.path.normcase(os.path.abspath(sess.out))
    candidates = []
    inst = getattr(sess, 'instance', None)
    if inst:
        # resolve MO2 priority ourselves so our own patched copy is never used as the source
        v = VFS.from_mo2(inst, sess.profile)
        for name, folder in v.roots:
            p = os.path.join(folder, 'Shaders', 'Lighting.hlsl')
            if os.path.isfile(p) and os.path.normcase(os.path.abspath(folder)) != out:
                candidates.append((name, p))
    else:
        for name, folder in sess.vfs.roots:
            p = os.path.join(folder, 'Shaders', 'Lighting.hlsl')
            if os.path.isfile(p):
                candidates.append((name, p))
    if not candidates:
        return None, None
    # highest priority first; skip copies that are already patched (an older output mod still enabled, or
    # our own copy in VFS mode): the source must be the shader mod's original
    for name, path in reversed(candidates):
        with open(path, encoding='utf-8', errors='replace') as f:
            text = f.read()
        if not any(m in text for m in PATCHED_MARKERS):
            return text, name
    return None, candidates[-1][0]


WORLD_POS = re.compile(r'input\.WorldPosition\.xyz\s*\+\s*FrameBuffer::CameraPosAdjust(?:\[\s*\w+\s*\])?\.xyz')


def world_position(text):
    """The variant's own expression for the absolute world position, as written closest before the ambient line:
    VR forks index the camera per eye (CameraPosAdjust[eyeIndex]), Community Shaders for SE/AE has one camera."""
    i = text.index(ANCHOR)
    before = list(WORLD_POS.finditer(text, 0, i))
    if before:
        return re.sub(r'\s+', ' ', before[-1].group(0))
    anywhere = WORLD_POS.search(text)
    if anywhere:
        return re.sub(r'\s+', ' ', anywhere.group(0))
    return 'input.WorldPosition.xyz + FrameBuffer::CameraPosAdjust.xyz'


def patch(text, provider, slots=(110, 111, 112, 116, 117)):
    if text.count(ANCHOR) != 1:
        raise RuntimeError(f'ambient line not found exactly once in {provider} Lighting.hlsl '
                           f'({text.count(ANCHOR)} matches) - this Community Shaders version is not supported yet')
    pos = world_position(text)
    text = text.replace(ANCHOR, f'Radiance::Ambient({ANCHOR}, {pos}, ambientNormal, SampColorSampler)')
    if INCLUDE_ANCHOR in text:
        text = text.replace(INCLUDE_ANCHOR, INCLUDE_ANCHOR + '\n#include "Radiance/Radiance.hlsli"', 1)
    else:
        m = list(re.finditer(r'^#include "[^"]+"\s*$', text, re.M))
        if not m:
            raise RuntimeError('no #include block found in Lighting.hlsl')
        i = m[-1].end()
        text = text[:i] + '\n#include "Radiance/Radiance.hlsli"' + text[i:]
    # include fingerprint: changing Radiance.hlsli also changes this file, so shader caches see the change
    import hashlib
    inc = hashlib.sha1(include_text(slots).encode()).hexdigest()[:12]
    return (f'{MARKER} generated from "{provider}" (include {inc}) - rerun the baker after updating Community '
            f'Shaders\n' + text)


def install(sess, log=print):
    """Write the patched shader into the output mod. Returns True when the patch is in place."""
    text, provider = find_source(sess)
    if text is None:
        if provider:
            log('Shader patch: kept the existing patched Lighting.hlsl (original not visible in this mode).')
            inc = os.path.join(sess.out, 'Shaders', 'Radiance', 'Radiance.hlsli')
            if os.path.isfile(inc):
                sess.shader_slots = [int(m.group(1)) for m in REGISTER.finditer(open(inc, encoding='utf-8').read())]
            return True
        log('Shader patch: no Community Shaders Lighting.hlsl found - install Community Shaders (or a fork) '
            'for in-game baked lighting.')
        return False
    slots = pick_slots(used_slots(sess))
    sess.shader_slots = slots         # Radiance.ini ShaderSlots (runtime.install) tells the plugin the same
    patched = patch(text, provider, slots)
    dst = os.path.join(sess.out, 'Shaders')
    os.makedirs(os.path.join(dst, 'Radiance'), exist_ok=True)
    inc_path = os.path.join(dst, 'Radiance', 'Radiance.hlsli')
    inc = include_text(slots)
    if not os.path.isfile(inc_path) or open(inc_path, encoding='utf-8').read() != inc:
        with open(inc_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(inc)
    target = os.path.join(dst, 'Lighting.hlsl')
    old = open(target, encoding='utf-8').read() if os.path.isfile(target) else None
    if old != patched:
        with open(target, 'w', encoding='utf-8', newline='\n') as f:
            f.write(patched)
        log(f'Shader patch: Lighting.hlsl from "{provider}" patched (texture slots '
            f'{", ".join("t%d" % t for t in slots)}, free in this setup). Clear the Community Shaders shader cache once '
            '(CS menu > Clear Shader Cache) if baked lighting does not show up.')
    else:
        log(f'Shader patch: up to date (source "{provider}").')
    return True
