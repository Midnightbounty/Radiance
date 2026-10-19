"""Session: load order + output mod + manifest. Shared by the CLI and the GUI."""
import json
import os
import re
import sys
import time
import traceback

from .bake import DEFAULTS, TOOL_VERSION, bake_cell, probe_path, settings_sig
from .plugins import LoadOrder, fk_str
from .scene import NothingToBake, load_rules, quick_fingerprint
from .vfs import GAME_FOLDERS, VFS

# the folder Radiance was unpacked into (next to Radiance.exe, or the source checkout)
TOOL_DIR = (os.path.dirname(sys.executable) if getattr(sys, 'frozen', False)
            else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(TOOL_DIR, 'config.json')
OUTPUT_MOD = 'Radiance Output'


def parse_key(s):
    """'01605E:Skyrim.esm' or 'Skyrim.esm:01605E' -> ('skyrim.esm', 0x1605E)."""
    a, b = s.split(':', 1)
    if re.fullmatch(r'[0-9A-Fa-f]{1,8}', a):
        return (b.lower(), int(a, 16) & 0xFFFFFF)
    return (a.lower(), int(b, 16) & 0xFFFFFF)


def load_config():
    cfg = {}
    if os.path.isfile(CONFIG):
        with open(CONFIG, encoding='utf-8') as f:
            cfg = json.load(f)
    return cfg


def save_config(**values):
    cfg = load_config()
    cfg.update(values)
    with open(CONFIG, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=1)


def _launching_mo2():
    """Folder of the ModOrganizer.exe that started this process (Radiance run as an MO2 executable), or None."""
    try:
        import ctypes
        from ctypes import wintypes

        class Entry(ctypes.Structure):
            _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD), ('pid', wintypes.DWORD),
                        ('heap', ctypes.c_size_t), ('module', wintypes.DWORD), ('threads', wintypes.DWORD),
                        ('ppid', wintypes.DWORD), ('prio', ctypes.c_long), ('flags', wintypes.DWORD),
                        ('exe', ctypes.c_wchar * 260)]
        k32 = ctypes.windll.kernel32
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = (wintypes.HANDLE, ctypes.POINTER(Entry))
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.QueryFullProcessImageNameW.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                   ctypes.POINTER(wintypes.DWORD))
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        snap = k32.CreateToolhelp32Snapshot(2, 0)          # TH32CS_SNAPPROCESS
        procs = {}
        e = Entry()
        e.dwSize = ctypes.sizeof(Entry)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            procs[e.pid] = (e.ppid, e.exe)
            ok = k32.Process32NextW(snap, ctypes.byref(e))
        k32.CloseHandle(snap)
        pid = os.getpid()
        for _ in range(10):                                # parent, grandparent, ...
            if pid not in procs:
                return None
            ppid, exe = procs[pid]
            if exe.lower() == 'modorganizer.exe':
                h = k32.OpenProcess(0x1000, False, pid)    # PROCESS_QUERY_LIMITED_INFORMATION
                buf = ctypes.create_unicode_buffer(32768)
                n = wintypes.DWORD(32768)
                path = buf.value if h and k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)) else None
                if h:
                    k32.CloseHandle(h)
                return os.path.dirname(path) if path else None
            pid = ppid
    except Exception:          # noqa: BLE001 - detection is best effort
        return None
    return None


def detect_instance():
    """The MO2 instance Radiance was started from: a portable instance (ModOrganizer.ini next to MO2's exe) or MO2's
    current global instance. Else a Radiance folder placed inside the instance (<instance>/tools/Radiance)."""
    mo2 = _launching_mo2()
    if mo2:
        if os.path.isfile(os.path.join(mo2, 'ModOrganizer.ini')):
            return mo2
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Software\\Mod Organizer Team\\Mod Organizer') as k:
                name = winreg.QueryValueEx(k, 'CurrentInstance')[0]
            inst = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'ModOrganizer', name)
            if os.path.isfile(os.path.join(inst, 'ModOrganizer.ini')):
                return inst
        except OSError:
            pass
    inst = os.path.dirname(os.path.dirname(TOOL_DIR))
    return inst if os.path.isfile(os.path.join(inst, 'ModOrganizer.ini')) else None


guess_instance = detect_instance


class NoInstance(SystemExit):
    pass


class Session:
    def __init__(self, mo2=None, profile=None, data=None, out=None, log=print):
        cfg = load_config()
        self.log = log
        mo2 = mo2 or detect_instance() or cfg.get('mo2')
        if mo2 and not os.path.isfile(os.path.join(mo2, 'ModOrganizer.ini')):
            mo2 = None                     # a remembered folder that moved or was deleted
        profile = profile or cfg.get('profile')
        self.instance, self.profile = mo2, profile
        t = time.time()
        if data:
            plugins_txt = None
            local = os.environ.get('LOCALAPPDATA', '')
            if mo2 and not any(os.path.isfile(os.path.join(local, g, 'plugins.txt')) for g in GAME_FOLDERS):
                plugins_txt = os.path.join(mo2, 'profiles', profile or VFS.selected_profile(mo2), 'plugins.txt')
            self.vfs = VFS.from_data(data, plugins_txt)
        elif mo2:
            self.vfs = VFS.from_mo2(mo2, profile)
        else:
            raise NoInstance('No Mod Organizer 2 instance found. Run Radiance from MO2 (add Radiance.exe as an '
                             'executable), or pick your MO2 folder (the one that contains ModOrganizer.ini).')
        self.out = out or cfg.get('output') or (os.path.join(mo2, 'mods', OUTPUT_MOD) if mo2 else
                                                os.path.join(TOOL_DIR, 'output'))
        self.reports = os.path.join(TOOL_DIR, 'reports')
        self.lo = LoadOrder(self.vfs, cache_dir=os.path.join(TOOL_DIR, 'cache'), log=log)
        self.rules_path = os.path.join(TOOL_DIR, 'light_rules.json')
        self.rules = load_rules(self.rules_path)
        self.settings = dict(DEFAULTS)
        self.settings.update(cfg.get('settings', {}))
        log(f'Load order: {self.vfs.label} - {len(self.lo.plugins)} plugins, {len(self.vfs.archives)} archives, '
            f'{len(self.lo.cells)} interior cells ({time.time() - t:.1f}s)')
        log(f'Output: {self.out}')

    # -- manifest ----------------------------------------------------------------------------------
    @property
    def manifest_path(self):
        return os.path.join(self.out, 'Radiance', 'manifest.json')

    def manifest(self):
        if os.path.isfile(self.manifest_path):
            with open(self.manifest_path, encoding='utf-8') as f:
                return json.load(f)
        return {'tool_version': TOOL_VERSION, 'cells': {}}

    def save_manifest(self, m):
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        tmp = self.manifest_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(m, f, indent=1, sort_keys=True)
        os.replace(tmp, self.manifest_path)

    # -- cells -------------------------------------------------------------------------------------
    def cells(self):
        man = self.manifest()['cells']
        out = []
        for c in self.lo.list_interiors():
            c['id'] = fk_str(c['key'])
            c['baked'] = c['id'] in man and os.path.isfile(probe_path(self.out, c['key']))
            out.append(c)
        return out

    def finish(self):
        """Everything besides the probes that the output mod needs: shader patch, light patch, runtime files."""
        from . import patchplugin, runtime, shaderpatch
        for step in (shaderpatch.install, patchplugin.build, runtime.install):
            try:
                step(self, self.log)
            except Exception as e:      # noqa: BLE001 - report and continue with the other outputs
                self.log(f'ERROR in {step.__module__}: {e!r}')
                self.log(traceback.format_exc(limit=4))

    def fingerprint(self, key):
        return quick_fingerprint(self.lo, key, settings_sig(self.settings, self.rules))

    def changed(self, keys=None, progress=None):
        """-> keys of already baked cells whose inputs (load order, tool version, mode) changed since their bake.
        Cells that were never baked are not included: 'changed' never starts baking new cells."""
        man = self.manifest()['cells']
        keys = keys or [c['key'] for c in self.lo.list_interiors()]
        out = []
        for i, k in enumerate(keys):
            ent = man.get(fk_str(k))
            if not ent:
                continue
            if not os.path.isfile(probe_path(self.out, k)) or ent.get('fingerprint') != self.fingerprint(k):
                out.append(k)
            if progress and i % 25 == 0:
                progress(i / len(keys))
        return out

    def bake(self, keys, preview=False, progress=None, should_stop=None):
        man = self.manifest()
        done, failed, skipped = 0, [], 0
        t0 = time.time()
        for i, k in enumerate(keys):
            if should_stop and should_stop():
                self.log('Stopped.')
                break

            def prog(f, i=i):
                if progress:
                    progress((i + f) / len(keys), fk_str(k))
            try:
                info, ctx = bake_cell(self.lo, k, self.out, self.settings, self.rules, self.log,
                                      report_dir=self.reports, progress=prog)
                ent = {k2: info[k2] for k2 in ('edid', 'name', 'plugin', 'fingerprint', 'file', 'bytes', 'time',
                                               'bricks', 'open_probes', 'seconds')}
                ent['calibration'] = info['calibration'].get('scale', 1.0)
                # light decisions for the patch plugin: [ref, class, LIGH flags, strength]
                ent['lights'] = [[L['ref'], L['class'], int(L['flags'], 16),
                                  round(sum(L['color']) * L['radius'], 1), L['keep_shadow'], L['live']]
                                 for L in info['lights']]
                man['cells'][info['cell']] = ent
                man['tool_version'] = TOOL_VERSION
                self.save_manifest(man)
                if preview:
                    from .preview import compare_report, validate
                    path = compare_report(ctx, os.path.join(self.reports, f'{k[0]}_{k[1]:06X}.png'), self.log)
                    self.log(f'  preview: {path}')
                    v = validate(ctx)
                    if v:
                        self.log(f"  accuracy vs path-traced reference: baked {v['baked_err'] * 100:.0f}% error, "
                                 f"original lighting {v['today_err'] * 100:.0f}% error; "
                                 f"brightness {v['baked_vs_today'] * 100:.0f}% of the original")
                done += 1
            except NothingToBake as e:
                skipped += 1
                self.log(f'[{fk_str(k)}] skipped: {e}')
                if fk_str(k) in man['cells']:       # was baked once, now empty: drop the old probes
                    del man['cells'][fk_str(k)]
                    self.save_manifest(man)
                    try:
                        os.remove(probe_path(self.out, k))
                    except OSError:
                        pass
            except Exception as e:      # noqa: BLE001 - keep going with the other cells
                failed.append(fk_str(k))
                self.log(f'[{fk_str(k)}] FAILED: {e!r}')
                self.log(traceback.format_exc(limit=3))
        self.finish()
        self.log(f'Done: {done} baked, {skipped} skipped (nothing to bake), {len(failed)} failed in {time.time() - t0:.0f}s')
        if failed:
            self.log('Failed: ' + ', '.join(failed))
        return done, failed
