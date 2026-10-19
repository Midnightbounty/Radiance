"""Command line. `Radiance.py` with no command opens the GUI."""
import argparse
import json
import os
import sys

from .app import Session, guess_instance, load_config, parse_key


def _session(a, log):
    data = a.data
    if a.vfs and not data:
        # launched by MO2: the game's Data folder shows the whole virtual load order
        from .vfs import VFS
        inst = a.mo2 or load_config().get('mo2') or guess_instance()
        g = VFS._mo2_general(inst)
        game = g.get('gamePath', '').replace('@ByteArray(', '').rstrip(')').replace('\\\\', '\\')
        data = os.path.join(game, 'Data')
    return Session(mo2=a.mo2, profile=a.profile, data=data, out=a.out, log=log)


def select(sess, a):
    cells = sess.cells()
    by_id = {c['id'].lower(): c for c in cells}
    by_edid = {c['edid'].lower(): c for c in cells}
    by_name = {}
    for c in cells:
        by_name.setdefault(c['name'].lower(), c)
    keys = []
    wanted = list(a.cell or [])
    if a.cells_file:
        with open(a.cells_file, encoding='utf-8') as f:
            wanted += [l.strip() for l in f if l.strip()]
    for s in wanted:
        c = by_edid.get(s.lower()) or by_id.get(s.lower()) or by_name.get(s.lower())
        if c is None and ':' in s:
            k = parse_key(s)
            c = next((x for x in cells if x['key'] == k), None)
        if c is None:
            raise SystemExit(f'unknown cell: {s}')
        keys.append(c['key'])
    if a.search:
        t = a.search.lower()
        keys += [c['key'] for c in cells if t in c['edid'].lower() or t in c['name'].lower()]
    if a.all:
        keys = [c['key'] for c in cells]
    if a.changed:
        keys = sess.changed(keys or None)
    seen = set()
    return [k for k in keys if not (k in seen or seen.add(k))]


def main(argv=None):
    p = argparse.ArgumentParser(prog='RadianceCLI', description='Baked interior lighting for Skyrim (probe volumes).')
    p.add_argument('--mo2', help='MO2 instance folder (default: config.json or two levels above this tool)')
    p.add_argument('--profile', help='MO2 profile (default: the selected one)')
    p.add_argument('--data', help='use one merged Data folder instead of resolving MO2 mods')
    p.add_argument('--vfs', action='store_true', help='running under MO2: read the virtual Data folder')
    p.add_argument('--out', help='output mod folder (default: <instance>\\mods\\Radiance Output)')
    p.add_argument('--jsonl', action='store_true', help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest='cmd')
    sub.add_parser('gui', help='cell picker (default)')
    ls = sub.add_parser('list', help='list interior cells')
    ls.add_argument('--search')
    b = sub.add_parser('bake', help='bake cells')
    b.add_argument('--cell', nargs='*', help='EditorID, in-game name or FormID:Plugin (e.g. WhiterunBanneredMare)')
    b.add_argument('--cells-file', help='text file with one cell per line')
    b.add_argument('--search', help='every cell whose name or EditorID contains this text')
    b.add_argument('--all', action='store_true', help='every interior cell')
    b.add_argument('--changed', action='store_true',
                   help='only already baked cells whose mods, mode or tool version changed since their bake')
    b.add_argument('--preview', action='store_true', help='also render a before/after comparison image')
    b.add_argument('--rays', type=int)
    b.add_argument('--spacing', type=float)
    b.add_argument('--bounces', type=int)
    b.add_argument('--performance', action='store_true',
                   help='performance mode: fewer shadow-casting lights (small light leaks and blur accepted)')
    sub.add_parser('finish', help='rewrite the light patch, shader patch and runtime files for the baked cells')
    ch = sub.add_parser('changed', help='list baked cells that changed since their bake')
    ch.add_argument('--performance', action='store_true')
    a = p.parse_args(argv)

    if a.cmd in (None, 'gui'):
        from .gui import run
        return run(a)

    if a.jsonl:
        def log(msg):
            print(json.dumps({'log': msg}), flush=True)

        def progress(f, cell=''):
            print(json.dumps({'progress': f, 'cell': cell}), flush=True)
    else:
        def log(msg):
            print(msg, flush=True)

        def progress(f, cell=''):
            pass

    sess = _session(a, log)
    if a.cmd == 'list':
        for c in sess.cells():
            if a.search and a.search.lower() not in (c['edid'] + c['name']).lower():
                continue
            print(f"{c['id']:32s} {c['edid']:40s} {c['name']:40s} {'baked' if c['baked'] else ''}")
        return 0
    if a.cmd == 'finish':
        sess.finish()
        return 0
    if a.cmd == 'changed':
        sess.settings['mode'] = 'performance' if a.performance else 'quality'
        for k in sess.changed(progress=lambda f: progress(f, 'checking')):
            print(f'{k[1]:06X}:{k[0]}')
        return 0
    if a.cmd == 'bake':
        for k in ('rays', 'spacing', 'bounces'):
            if getattr(a, k) is not None:
                sess.settings[k] = getattr(a, k)
        sess.settings['mode'] = 'performance' if a.performance else 'quality'
        keys = select(sess, a)
        if not keys:
            log('Nothing to bake.')
            return 0
        log(f'Baking {len(keys)} cell(s)...')
        done, failed = sess.bake(keys, preview=a.preview, progress=progress)
        return 1 if failed else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
