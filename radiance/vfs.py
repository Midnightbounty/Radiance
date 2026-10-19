"""Game file system: which file wins, loose or archive, for a load order.

Two modes:
  * MO2 instance mode: resolves the profile's mod priority itself (works with or without MO2 running).
  * Data mode: one merged Data folder, e.g. when launched as an MO2 executable (USVFS shows the merged view).
"""
import configparser
import os
import re

from .bsa import BSA, norm

BASE_MASTERS = ['skyrim.esm', 'update.esm', 'dawnguard.esm', 'hearthfires.esm', 'dragonborn.esm', 'skyrimvr.esm']
ASSET_DIRS = ('meshes', 'textures', 'strings')
GAME_FOLDERS = ('Skyrim VR', 'Skyrim Special Edition', 'Skyrim Special Edition GOG')   # Documents / AppData


def _read_ini_archives(ini_paths):
    for p in ini_paths:
        if not p or not os.path.isfile(p):
            continue
        txt = open(p, encoding='utf-8', errors='replace').read()
        out = []
        for key in ('sResourceArchiveList', 'sResourceArchiveList2'):
            m = re.search(r'^\s*%s\s*=\s*(.*)$' % key, txt, re.M | re.I)
            if m:
                out += [a.strip() for a in m.group(1).split(',') if a.strip()]
        if out:
            return out
    return []


class VFS:
    def __init__(self, roots, plugins, ini_paths=(), label=''):
        """roots: list of (name, folder) from lowest to highest priority. plugins: active load order."""
        self.roots = roots
        self.label = label
        self.loose = {}      # norm path -> (root name, full path)
        self.root_files = {}  # top-level file name (lower) -> full path (plugins, archives)
        for name, folder in roots:
            if not os.path.isdir(folder):
                continue
            for e in os.scandir(folder):
                if e.is_file():
                    self.root_files[e.name.lower()] = e.path
            for sub in os.listdir(folder):
                if sub.lower() in ASSET_DIRS:
                    base = os.path.join(folder, sub)
                    for dp, _, fns in os.walk(base):
                        rel = os.path.relpath(dp, folder)
                        for fn in fns:
                            self.loose[norm(os.path.join(rel, fn))] = (name, os.path.join(dp, fn))
        # SE/AE have no skyrimvr.esm
        base = [p for p in BASE_MASTERS if p != 'skyrimvr.esm' or p in self.root_files]
        self.plugins = base + [p for p in plugins if p.lower() not in BASE_MASTERS]
        names =_read_ini_archives(ini_paths) or [
            'Skyrim - Misc.bsa', 'Skyrim - Shaders.bsa', 'Skyrim - Interface.bsa', 'Skyrim - Animations.bsa',
            'Skyrim - Meshes0.bsa', 'Skyrim - Meshes1.bsa', 'Skyrim - Sounds.bsa', 'Skyrim - Voices_en0.bsa',
            'Skyrim - Textures0.bsa', 'Skyrim - Textures1.bsa', 'Skyrim - Textures2.bsa', 'Skyrim - Textures3.bsa',
            'Skyrim - Textures4.bsa', 'Skyrim - Textures5.bsa', 'Skyrim - Textures6.bsa', 'Skyrim - Textures7.bsa',
            'Skyrim - Textures8.bsa', 'Skyrim - Patch.bsa']
        names.append('Skyrim_VR - Main.bsa')
        for p in self.plugins:
            stem = os.path.splitext(p)[0]
            names += [stem + '.bsa', stem + ' - Textures.bsa']
        self.archives = []
        self.archived = {}  # norm path -> archive index
        seen = set()
        for n in names:
            path = self.root_files.get(n.lower())
            if not path or n.lower() in seen:
                continue
            seen.add(n.lower())
            try:
                a = BSA(path)
            except (ValueError, OSError):
                continue
            idx = len(self.archives)
            self.archives.append(a)
            for k in a.files:
                if k.startswith(ASSET_DIRS):
                    self.archived[k] = idx

    # -- factories -------------------------------------------------------------------------------
    @staticmethod
    def _mo2_general(instance):
        ini = configparser.ConfigParser(interpolation=None, strict=False)
        ini.read(os.path.join(instance, 'ModOrganizer.ini'), encoding='utf-8')
        return ini['General']

    @classmethod
    def selected_profile(cls, instance):
        g = cls._mo2_general(instance)
        return re.sub(r'^@ByteArray\((.*)\)$', r'\1', g.get('selected_profile', 'Default'))

    @classmethod
    def from_mo2(cls, instance, profile=None):
        g = cls._mo2_general(instance)
        if profile is None:
            profile = cls.selected_profile(instance)
        game = re.sub(r'^@ByteArray\((.*)\)$', r'\1', g.get('gamePath', '')).replace('\\\\', '\\')
        pdir = os.path.join(instance, 'profiles', profile)
        mods = [l[1:].strip() for l in open(os.path.join(pdir, 'modlist.txt'), encoding='utf-8')
                if l.startswith('+') and not l.strip().endswith('_separator')][::-1]
        roots = [('<Data>', os.path.join(game, 'Data'))]
        roots += [(m, os.path.join(instance, 'mods', m)) for m in mods]
        roots.append(('<overwrite>', os.path.join(instance, 'overwrite')))
        plugins = [l.strip()[1:] for l in open(os.path.join(pdir, 'plugins.txt'), encoding='utf-8')
                   if l.startswith('*')]
        inis = [os.path.join(d, n) for d in (pdir, game) for n in ('skyrimvr.ini', 'skyrim.ini')]
        return cls(roots, plugins, inis, label=f'{instance} [{profile}]')

    @classmethod
    def from_data(cls, data_dir, plugins_txt=None):
        local = os.environ.get('LOCALAPPDATA', '')
        if plugins_txt is None:
            plugins_txt = next((p for p in (os.path.join(local, g, 'plugins.txt') for g in GAME_FOLDERS)
                                if os.path.isfile(p)), os.path.join(local, GAME_FOLDERS[0], 'plugins.txt'))
        plugins = [l.strip()[1:] for l in open(plugins_txt, encoding='utf-8') if l.startswith('*')]
        game = os.path.dirname(os.path.abspath(data_dir))
        docs = os.path.join(os.path.expanduser('~'), 'Documents', 'My Games')
        inis = [os.path.join(docs, g, n) for g in GAME_FOLDERS for n in ('skyrimvr.ini', 'skyrim.ini')]
        inis += [os.path.join(game, n) for n in ('skyrimvr.ini', 'skyrim.ini')]
        return cls([('<Data>', data_dir)], plugins, inis, label=data_dir)

    # -- access ----------------------------------------------------------------------------------
    def plugin_path(self, name):
        return self.root_files.get(name.lower())

    def exists(self, path):
        k = norm(path)
        return k in self.loose or k in self.archived

    def read(self, path):
        k = norm(path)
        hit = self.loose.get(k)
        if hit:
            with open(hit[1], 'rb') as f:
                return f.read()
        idx = self.archived.get(k)
        if idx is None:
            return None
        return self.archives[idx].read(k)

    def source(self, path):
        """Short identity of the winning file, for change detection."""
        k = norm(path)
        hit = self.loose.get(k)
        if hit:
            st = os.stat(hit[1])
            return f'loose|{st.st_size}|{int(st.st_mtime)}'    # same whether read via MO2 or its virtual Data
        idx = self.archived.get(k)
        if idx is None:
            return 'missing'
        a = self.archives[idx]
        st = os.stat(a.path)
        return f'{os.path.basename(a.path)}|{st.st_size}|{int(st.st_mtime)}'
