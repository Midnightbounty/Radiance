"""In-game runtime: copy the SKSE plugin into the output mod and check the MO2 setup around it."""
import os
import re
import shutil

from .app import TOOL_DIR
from .patchplugin import ESP_NAME

RUNTIME_DIR = os.path.join(TOOL_DIR, 'runtime')

INI_VERSION = 4
DEFAULT_INI = """; Radiance - in-game settings (read when the game starts)
; Every cell is calibrated when it is baked, so the defaults show the bake as intended.
[Settings]
IniVersion=4
; 0 turns baked lighting off everywhere
Enabled=1
; multiplier on the baked light (1.0 = as calibrated by the baker)
BounceGain=1.0
; multiplier on each cell's kept share of its original flat ambient (1.0 = as calibrated, 0 = none)
AmbientFloorMultiplier=1.0
; daylight / large static lights by day and at night (1.0 = as the lighting mod placed them)
DaylightDay=1.0
DaylightNight=1.0
; sample the baked light this many units in front of surfaces (the baker assumes 24)
NormalOffset=24
; key that switches between baked lighting and the original lights (Windows virtual-key code; easiest to change in the
; baker window under "In-game keys")
ToggleKey=118
; key that cycles the debug view: off / magenta where baked lighting is active / baked light only
DebugKey=119
; texture slots the baker chose for the shader patch (free in your Community Shaders variant) - written by the
; baker on every bake, must match Shaders\Radiance\Radiance.hlsli; do not edit
ShaderSlots=110,111,112,116,117
"""

# key name -> Windows virtual-key code (what the SKSE plugin reads with GetAsyncKeyState)
KEYS = {f'F{i}': 0x6F + i for i in range(1, 13)}
KEYS.update({'Insert': 0x2D, 'Delete': 0x2E, 'Home': 0x24, 'End': 0x23, 'Page Up': 0x21, 'Page Down': 0x22,
             'Pause': 0x13, 'Scroll Lock': 0x91})
KEYS.update({f'NumPad {i}': 0x60 + i for i in range(10)})
KEYS.update({'NumPad *': 0x6A, 'NumPad +': 0x6B, 'NumPad -': 0x6D, 'NumPad .': 0x6E, 'NumPad /': 0x6F})
KEYS.update({chr(c): c for c in range(0x41, 0x5B)})
KEYS.update({str(i): 0x30 + i for i in range(10)})
DEFAULT_KEYS = (0x76, 0x77)        # F7 toggle, F8 debug view


def key_name(vk):
    return next((n for n, v in KEYS.items() if v == vk), f'Key code {vk}')


def ini_path(out):
    return os.path.join(out, 'SKSE', 'Plugins', 'Radiance.ini')


def read_keys(out):
    """-> (toggle, debug) virtual-key codes from the output mod's Radiance.ini (defaults if missing)."""
    keys = list(DEFAULT_KEYS)
    try:
        txt = open(ini_path(out), encoding='utf-8', errors='replace').read()
    except OSError:
        return tuple(keys)
    for i, name in enumerate(('ToggleKey', 'DebugKey')):
        m = re.search(r'^\s*%s\s*=\s*(\d+)' % name, txt, re.M)     # decimal, as GetPrivateProfileInt reads it
        if m:
            v = int(m.group(1))
            if 0 < v < 256:
                keys[i] = v
    return tuple(keys)


def set_ini_values(out, values):
    """Set keys in Radiance.ini (written with the defaults first if it doesn't exist)."""
    path = ini_path(out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        txt = open(path, encoding='utf-8', errors='replace').read()
    except OSError:
        txt = DEFAULT_INI
    for name, v in values:
        line = f'{name}={v}'
        txt, n = re.subn(r'^\s*%s\s*=.*$' % name, line, txt, flags=re.M)
        if not n:
            txt = txt.rstrip('\n') + '\n' + line + '\n'
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(txt)
    os.replace(tmp, path)


def write_keys(out, toggle, debug):
    """Set the two in-game keys in Radiance.ini."""
    set_ini_values(out, (('ToggleKey', int(toggle)), ('DebugKey', int(debug))))


def install(sess, log=print):
    dst = os.path.join(sess.out, 'SKSE', 'Plugins')
    os.makedirs(dst, exist_ok=True)
    dll = os.path.join(RUNTIME_DIR, 'Radiance.dll')
    if os.path.isfile(dll):
        target = os.path.join(dst, 'Radiance.dll')
        if not os.path.isfile(target) or open(dll, 'rb').read() != open(target, 'rb').read():
            shutil.copy2(dll, target)
            log('Runtime: installed SKSE plugin Radiance.dll')
    else:
        log('Runtime: Radiance.dll missing from the tool folder (runtime\\) - baked lighting will not show in game')
    ini = os.path.join(dst, 'Radiance.ini')
    old = open(ini, encoding='utf-8', errors='replace').read() if os.path.isfile(ini) else ''
    if f'IniVersion={INI_VERSION}' not in old:       # missing or from an older version: defaults changed
        keys = read_keys(sess.out) if old else DEFAULT_KEYS      # the player's chosen keys survive
        with open(ini, 'w', encoding='utf-8') as f:
            f.write(DEFAULT_INI)
        if keys != DEFAULT_KEYS:
            write_keys(sess.out, *keys)
        if old:
            log('Runtime: Radiance.ini replaced with the new defaults (settings changed in this version)')
    slots = getattr(sess, 'shader_slots', None)
    if slots:
        set_ini_values(sess.out, (('ShaderSlots', ','.join(str(t) for t in slots)),))
    check_mo2(sess, log)


def check_mo2(sess, log):
    """Warn about MO2 settings the bake depends on (read-only)."""
    inst = getattr(sess, 'instance', None)
    if not inst:
        return
    from .vfs import VFS
    try:
        v = VFS.from_mo2(inst, sess.profile)
    except OSError:
        return
    out = os.path.normcase(os.path.abspath(sess.out))
    names = [n for n, folder in v.roots]
    folders = [os.path.normcase(os.path.abspath(f)) for n, f in v.roots]
    if out not in folders:
        log(f'MO2: the output mod "{os.path.basename(sess.out)}" is not enabled in the current profile - enable it.')
        return
    mine = folders.index(out)
    for i, (n, folder) in enumerate(v.roots):
        if i > mine and os.path.isfile(os.path.join(folder, 'Shaders', 'Lighting.hlsl')):
            log(f'MO2: "{n}" overrides the patched Lighting.hlsl. Move "{os.path.basename(sess.out)}" below "{n}" '
                '(higher priority) in the left pane.')
    if ESP_NAME.lower() not in [p.lower() for p in v.plugins] and os.path.isfile(os.path.join(sess.out, ESP_NAME)):
        log(f'MO2: tick {ESP_NAME} in the right pane (Plugins) so the light changes load.')
