"""Cell picker (tkinter). Bakes run in a child process so the window stays responsive and Stop works."""
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import shutil
from tkinter import filedialog, ttk

from . import runtime
from .app import TOOL_DIR, NoInstance, Session, save_config
from .bake import TOOL_VERSION

MODE_LABELS = ('Quality (best looking)', 'Performance (fewer shadow lights, VR)')
SETTINGS_FILE = os.path.join(TOOL_DIR, 'gui_settings.json')
NUMBERS = (('rays', 'Rays/probe', 512, 32, 4096, 32), ('spacing', 'Probe spacing (0=auto)', 0, 0, 256, 16),
           ('bounces', 'Bounces', 6, 1, 16, 1))
# what the baker writes into the output mod (Clear output folder removes these, nothing else)
OUTPUT_ITEMS = ('Radiance', 'Radiance.esp', 'Shaders', 'SKSE')

# dark theme
BG, PANEL, FIELD, BORDER = '#1e1f22', '#2b2d31', '#25272b', '#3c3f45'
FG, MUTED, ACCENT, SELECT, DANGER = '#e3e5e8', '#8b9099', '#4c8bf5', '#2f5596', '#c0392b'


def dark_title_bar(win):
    """Windows 10/11: dark window title bar (DWMWA_USE_IMMERSIVE_DARK_MODE)."""
    try:
        import ctypes
        win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        on = ctypes.c_int(1)
        for attr in (20, 19):           # 19 on Windows 10 before 20H1
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(on), ctypes.sizeof(on)) == 0:
                break
    except (AttributeError, OSError):
        pass


def apply_dark_theme(root):
    root.configure(bg=BG)
    s = ttk.Style(root)
    s.theme_use('clam')
    s.configure('.', background=BG, foreground=FG, fieldbackground=FIELD, bordercolor=BORDER, lightcolor=PANEL,
                darkcolor=BG, troughcolor=PANEL, selectbackground=SELECT, selectforeground=FG, insertcolor=FG,
                focuscolor=ACCENT, arrowcolor=FG)
    s.map('.', foreground=[('disabled', MUTED)])
    s.configure('TButton', background=PANEL, padding=(8, 3))
    s.map('TButton', background=[('disabled', BG), ('pressed', SELECT), ('active', BORDER)],
          foreground=[('disabled', MUTED)])
    s.configure('Danger.TButton', background='#5a2420')
    s.map('Danger.TButton', background=[('disabled', BG), ('pressed', DANGER), ('active', '#7a2e27')])
    s.configure('Treeview', background=FIELD, fieldbackground=FIELD, foreground=FG, rowheight=24, borderwidth=0)
    s.map('Treeview', background=[('selected', SELECT)], foreground=[('selected', FG)])
    s.configure('Treeview.Heading', background=PANEL, foreground=FG, relief='flat', padding=(6, 4))
    s.map('Treeview.Heading', background=[('active', BORDER)])
    for w in ('TEntry', 'TCombobox', 'TSpinbox'):
        s.configure(w, fieldbackground=FIELD, foreground=FG, background=PANEL, arrowcolor=FG, insertcolor=FG)
        s.map(w, fieldbackground=[('readonly', FIELD), ('disabled', BG)], foreground=[('disabled', MUTED)],
              background=[('active', BORDER), ('disabled', BG)], selectbackground=[('readonly', FIELD)],
              selectforeground=[('readonly', FG)])
    for w in ('TCheckbutton', 'TRadiobutton'):
        s.configure(w, background=BG, indicatorbackground=FIELD, indicatorforeground=FG, indicatorrelief='flat')
        s.map(w, background=[('active', BG)], indicatorbackground=[('selected', ACCENT), ('active', BORDER)])
    s.configure('Vertical.TScrollbar', background=PANEL, troughcolor=BG, bordercolor=BG, arrowcolor=FG)
    s.map('Vertical.TScrollbar', background=[('active', BORDER)])
    s.configure('Horizontal.TProgressbar', background=ACCENT, troughcolor=PANEL, bordercolor=BORDER)
    s.configure('Sash', background=BORDER, sashthickness=5)
    s.configure('Muted.TLabel', foreground=MUTED)
    # the drop-down list of comboboxes is a classic Tk listbox
    root.option_add('*TCombobox*Listbox.background', FIELD)
    root.option_add('*TCombobox*Listbox.foreground', FG)
    root.option_add('*TCombobox*Listbox.selectBackground', SELECT)
    root.option_add('*TCombobox*Listbox.selectForeground', FG)
    root.option_add('*TCombobox*Listbox.borderWidth', 0)
    dark_title_bar(root)


class DarkMessageBox:
    """Dark replacements for tkinter.messagebox (the native Windows boxes can't be themed). Same call style."""

    def _dialog(self, title, message, kind, buttons):
        root = tk._default_root
        win = tk.Toplevel(root, bg=BG)
        win.title(title)
        win.resizable(False, False)
        win.transient(root)
        result = {'value': buttons[-1][1]}
        body = ttk.Frame(win, padding=(18, 16, 18, 8))
        body.pack(fill='both', expand=True)
        icon = {'info': ('i', ACCENT), 'warning': ('!', '#e0a030'), 'error': ('×', DANGER),
                'question': ('?', ACCENT)}[kind]
        tk.Label(body, text=icon[0], bg=icon[1], fg='white', font=('Segoe UI', 14, 'bold'), width=2).pack(
            side='left', anchor='n', padx=(0, 14))
        ttk.Label(body, text=message, wraplength=420, justify='left').pack(side='left', fill='both')
        row = ttk.Frame(win, padding=(18, 4, 18, 14))
        row.pack(fill='x')

        def close(v):
            result['value'] = v
            win.destroy()
        for i, (label, value) in enumerate(reversed(buttons)):
            b = ttk.Button(row, text=label, command=lambda v=value: close(v))
            b.pack(side='right', padx=(6, 0))
            if i == 0:                  # default: the last button (No / OK) - never the destructive choice
                b.focus_set()
                win.bind('<Return>', lambda e, v=value: close(v))
        win.bind('<Escape>', lambda e: close(buttons[-1][1]))
        win.protocol('WM_DELETE_WINDOW', lambda: close(buttons[-1][1]))
        dark_title_bar(win)
        win.update_idletasks()
        if root is not None and root.winfo_viewable():
            x = root.winfo_rootx() + (root.winfo_width() - win.winfo_width()) // 2
            y = root.winfo_rooty() + (root.winfo_height() - win.winfo_height()) // 3
            win.geometry(f'+{max(x, 0)}+{max(y, 0)}')
        win.grab_set()
        win.wait_window()
        return result['value']

    def showinfo(self, title, message):
        self._dialog(title, message, 'info', [('OK', None)])

    def showwarning(self, title, message):
        self._dialog(title, message, 'warning', [('OK', None)])

    def showerror(self, title, message):
        self._dialog(title, message, 'error', [('OK', None)])

    def askyesno(self, title, message):
        return self._dialog(title, message, 'question', [('Yes', True), ('No', False)])


messagebox = DarkMessageBox()


class App:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.proc = None
        self.sess = None
        self.q = queue.Queue()
        self.cells = []
        self.status = {}          # id -> 'baked' / 'changed' / ''
        self.sort_col, self.sort_rev = 'name', False
        self.on_line = self.on_done = None
        self.temp_file = None
        self.stopped = False
        self.closing = False
        saved = self._load_settings()
        root.title(f"Radiance {TOOL_VERSION.replace('-', ' ')} - baked global illumination")
        root.geometry('1180x720')
        root.minsize(900, 500)
        root.protocol('WM_DELETE_WINDOW', self.close)

        top = ttk.Frame(root, padding=6)
        top.pack(fill='x')
        ttk.Label(top, text='Search:').pack(side='left')
        self.search = tk.StringVar()
        self.search.trace_add('write', lambda *_: self.refresh())
        e = ttk.Entry(top, textvariable=self.search, width=40)
        e.pack(side='left', padx=4)
        e.focus_set()
        self.only = tk.StringVar(value='all')
        for val, txt in (('all', 'All'), ('baked', 'Baked'), ('changed', 'Changed since bake'),
                         ('unbaked', 'Not baked')):
            ttk.Radiobutton(top, text=txt, value=val, variable=self.only, command=self.refresh).pack(side='left')
        self.count = ttk.Label(top, text='')
        self.count.pack(side='right')

        mid = ttk.PanedWindow(root, orient='vertical')
        mid.pack(fill='both', expand=True, padx=6)
        lf = ttk.Frame(mid)
        cols = ('name', 'edid', 'id', 'plugin', 'status')
        self.tree = ttk.Treeview(lf, columns=cols, show='headings', selectmode='extended')
        for c, w, t in (('name', 260, 'Name'), ('edid', 260, 'EditorID'), ('id', 220, 'Form'),
                        ('plugin', 200, 'Winning plugin'), ('status', 110, 'Status')):
            self.tree.heading(c, text=t, command=lambda c=c: self.sort(c))
            self.tree.column(c, width=w, anchor='w')
        sb = ttk.Scrollbar(lf, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self.tree.bind('<Double-1>', lambda e: self.open_preview())
        mid.add(lf, weight=3)
        logf = ttk.Frame(mid)
        self.log = tk.Text(logf, height=10, wrap='none', font=('Consolas', 9), bg=FIELD, fg=FG,
                           insertbackground=FG, selectbackground=SELECT, relief='flat', borderwidth=0,
                           highlightthickness=0, padx=6, pady=4)
        lsb = ttk.Scrollbar(logf, orient='vertical', command=self.log.yview)
        self.log.configure(yscrollcommand=lsb.set)
        self.log.pack(side='left', fill='both', expand=True)
        lsb.pack(side='right', fill='y')
        mid.add(logf, weight=1)

        opts = ttk.Frame(root, padding=6)
        opts.pack(fill='x')
        self.mode = tk.StringVar(value=saved.get('mode', MODE_LABELS[0]) if saved.get('mode') in MODE_LABELS
                                 else MODE_LABELS[0])
        ttk.Label(opts, text='Mode').pack(side='left')
        ttk.Combobox(opts, textvariable=self.mode, values=MODE_LABELS, state='readonly', width=34).pack(
            side='left', padx=(2, 14))
        self.numbers = {}
        for key, label, default, lo, hi, inc in NUMBERS:
            var = tk.StringVar(value=str(saved.get(key, default)))
            self.numbers[key] = (var, lo, hi, default)
            ttk.Label(opts, text=label).pack(side='left')
            ttk.Spinbox(opts, from_=lo, to=hi, increment=inc, textvariable=var, width=6).pack(side='left', padx=(2, 10))
        self.preview = tk.BooleanVar(value=bool(saved.get('preview', False)))
        ttk.Checkbutton(opts, text='Render before/after preview', variable=self.preview).pack(side='left')

        keys = ttk.Frame(root, padding=(6, 0, 6, 0))
        keys.pack(fill='x')
        ttk.Label(keys, text='In-game keys:').pack(side='left')
        self.key_vars, self.key_boxes = {}, []
        for which, label in (('toggle', 'Baked lighting on/off'), ('debug', 'Debug view')):
            ttk.Label(keys, text=label).pack(side='left', padx=(10, 2))
            var = tk.StringVar(value='')
            box = ttk.Combobox(keys, textvariable=var, values=list(runtime.KEYS), state='disabled', width=12)
            box.pack(side='left')
            box.bind('<<ComboboxSelected>>', lambda e: self.set_keys())
            self.key_vars[which] = var
            self.key_boxes.append(box)
        ttk.Label(keys, text='(takes effect the next time the game starts)', style='Muted.TLabel').pack(
            side='left', padx=10)

        btns = ttk.Frame(root, padding=6)
        btns.pack(fill='x')
        self.buttons = []
        for txt, cmd in (('Bake selected', self.bake_selected), ('Bake all interiors', self.bake_all),
                         ('Bake changed', self.bake_changed), ('Check for changes', self.check_changed),
                         ('Open preview', self.open_preview), ('Open output folder', self.open_output)):
            b = ttk.Button(btns, text=txt, command=cmd)
            b.pack(side='left', padx=2)
            self.buttons.append(b)
        self.stop_btn = ttk.Button(btns, text='Stop', command=self.stop, state='disabled')
        self.stop_btn.pack(side='left', padx=8)
        clear = ttk.Button(btns, text='Clear output folder', style='Danger.TButton', command=self.clear_output)
        clear.pack(side='right', padx=(6, 0))
        self.buttons.append(clear)
        self.prog = ttk.Progressbar(btns, mode='determinate', maximum=1.0)
        self.prog.pack(side='left', fill='x', expand=True, padx=6)
        self.prog_lbl = ttk.Label(btns, text='', width=28)
        self.prog_lbl.pack(side='left')

        self.write('Reading load order...')
        self._set_busy(True)
        root.after(50, self.load)
        root.after(100, self.poll)

    # -- settings --------------------------------------------------------------------------------
    def _load_settings(self):
        try:
            with open(SETTINGS_FILE, encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_settings(self):
        data = {'mode': self.mode.get(), 'preview': bool(self.preview.get())}
        for key, (var, lo, hi, default) in self.numbers.items():
            data[key] = self._number(key, quiet=True)
        try:
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=1)
        except OSError:
            pass

    def _number(self, key, quiet=False):
        var, lo, hi, default = self.numbers[key]
        try:
            v = int(float(var.get()))
        except ValueError:
            if quiet:
                return default
            raise ValueError(key)
        return max(lo, min(hi, v))

    # -- data ------------------------------------------------------------------------------------
    def load(self):
        mo2 = self.args.mo2
        while True:
            try:
                self.sess = Session(mo2=mo2, profile=self.args.profile, data=self.args.data,
                                    out=self.args.out, log=self.write)
                break
            except NoInstance as e:
                # not started from MO2 and nothing remembered: ask once, then remember the folder
                messagebox.showinfo('Radiance', f'{e}\n\nNext, select your Mod Organizer 2 folder.')
                mo2 = filedialog.askdirectory(title='Select your Mod Organizer 2 folder (contains ModOrganizer.ini)')
                if mo2 and os.path.isfile(os.path.join(mo2, 'ModOrganizer.ini')):
                    save_config(mo2=os.path.normpath(mo2))
                    continue
                if mo2:
                    messagebox.showerror('Radiance', 'That folder has no ModOrganizer.ini. For a global MO2 instance '
                                         'it is %LOCALAPPDATA%\\ModOrganizer\\<instance name>.')
                    continue
                err = 'No Mod Organizer 2 folder selected.'
            except BaseException as e:   # noqa: BLE001
                err = e
            messagebox.showerror('Radiance', f'Could not read the load order:\n{err}')
            self.closing = True
            self.root.destroy()
            return
        self._reload_cells()
        self._show_keys()
        self.refresh()                           # show the cells now, not only after a change check
        self._set_busy(False)
        if any(s == 'baked' for s in self.status.values()):
            self.check_changed(quiet=True)       # flag baked cells whose mods changed since their bake

    def _reload_cells(self):
        self.cells = self.sess.cells()
        man = self.sess.manifest()['cells']
        self.baked_times = {cid: ent.get('time') for cid, ent in man.items()}
        for c in self.cells:
            if not c['baked']:
                self.status[c['id']] = ''
            elif self.status.get(c['id']) != 'changed':
                self.status[c['id']] = 'baked'
        self._apply_sort()

    def refresh(self):
        t = self.search.get().lower().strip()
        only = self.only.get()
        self.tree.delete(*self.tree.get_children())
        n = 0
        for c in self.cells:
            st = self.status.get(c['id'], '')
            if (only == 'baked' and st != 'baked') or (only == 'changed' and st != 'changed') or \
                    (only == 'unbaked' and st != ''):
                continue
            if t and t not in c['name'].lower() and t not in c['edid'].lower() and t not in c['id'].lower():
                continue
            self.tree.insert('', 'end', iid=c['id'], values=(c['name'], c['edid'], c['id'], c['plugin'],
                                                            {'changed': 'changed - rebake'}.get(st, st)))
            n += 1
        self.count.configure(text=f'{n} of {len(self.cells)} interior cells')

    def sort(self, col):
        self.sort_rev = not self.sort_rev if self.sort_col == col else False
        self.sort_col = col
        self._apply_sort()
        self.refresh()

    def _apply_sort(self):
        idx = ('name', 'edid', 'id', 'plugin', 'status').index(self.sort_col)
        self.cells.sort(key=lambda c: str((c['name'], c['edid'], c['id'], c['plugin'],
                                           self.status.get(c['id'], ''))[idx]).lower(), reverse=self.sort_rev)

    # -- in-game keys (Radiance.ini in the output mod) ------------------------------------
    def _show_keys(self):
        self.keys = runtime.read_keys(self.sess.out)
        for which, vk in zip(('toggle', 'debug'), self.keys):
            self.key_vars[which].set(runtime.key_name(vk))

    def set_keys(self):
        new = tuple(runtime.KEYS.get(self.key_vars[w].get(), old) for w, old in zip(('toggle', 'debug'), self.keys))
        if new == self.keys:
            return
        if new[0] == new[1]:
            messagebox.showwarning('Radiance', 'Pick two different keys for on/off and the debug view.')
            self._show_keys()
            return
        try:
            runtime.write_keys(self.sess.out, *new)
        except OSError as e:
            messagebox.showerror('Radiance', f'Could not save the keys:\n{e}')
            self._show_keys()
            return
        self.keys = new
        names = [runtime.key_name(v) for v in new]
        self.write(f'In-game keys saved: {names[0]} = baked lighting on/off, {names[1]} = debug view '
                   '(SKSE\\Plugins\\Radiance.ini in the output mod).')
        if any(len(n) == 1 for n in names):
            messagebox.showinfo('Radiance', 'Letter and number keys are also game controls; '
                                'they will do both. Function keys (F1-F12) or the numpad are safer.')

    # -- actions ---------------------------------------------------------------------------------
    def base_cmd(self):
        exe = sys.executable
        if getattr(sys, 'frozen', False):
            # the console twin of Radiance.exe: bakes need a working stdout for progress (its window stays hidden)
            cmd = [os.path.join(TOOL_DIR, 'RadianceCLI.exe'), '--jsonl']
        else:
            if os.path.basename(exe).lower() == 'pythonw.exe':
                exe = os.path.join(os.path.dirname(exe), 'python.exe')
            cmd = [exe, os.path.join(TOOL_DIR, 'Radiance.py'), '--jsonl']
        # the bake works on exactly the load order and output this window shows
        given = {k: getattr(self.args, k, None) for k in ('mo2', 'profile', 'data', 'out')}
        if self.sess is not None:
            given.update(mo2=self.sess.instance or given['mo2'], profile=self.sess.profile or given['profile'],
                         out=self.sess.out)
        for k in ('mo2', 'profile', 'data', 'out'):
            if given[k]:
                cmd += [f'--{k}', given[k]]
        if getattr(self.args, 'vfs', False):
            cmd.append('--vfs')
        return cmd

    def bake_opts(self):
        """Command line options from the settings row; None (after telling the user) when a number is invalid."""
        try:
            nums = {k: self._number(k) for k in self.numbers}
        except ValueError as e:
            label = next(l for k, l, *_ in NUMBERS if k == str(e))
            messagebox.showwarning('Radiance', f'"{label}" must be a number.')
            return None
        self._save_settings()
        o = ['--rays', str(nums['rays']), '--spacing', str(nums['spacing']), '--bounces', str(nums['bounces'])]
        return o + (['--preview'] if self.preview.get() else []) + self.mode_args()

    def mode_args(self):
        return ['--performance'] if self.mode.get() == MODE_LABELS[1] else []

    def bake_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo('Radiance', 'Select one or more cells first (Ctrl/Shift-click).')
            return
        opts = self.bake_opts()
        if opts is None:
            return
        fd, self.temp_file = tempfile.mkstemp(suffix='.txt', prefix='radiance-')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sel))
        self.run(self.base_cmd() + ['bake', '--cells-file', self.temp_file] + opts, f'Baking {len(sel)} cell(s)',
                 bake=True)

    def bake_all(self):
        opts = self.bake_opts()
        if opts is None:
            return
        if messagebox.askyesno('Radiance', f'Bake all {len(self.cells)} interior cells?\n'
                               'This takes many hours. You can stop at any time; finished cells are kept.'):
            self.run(self.base_cmd() + ['bake', '--all'] + opts, 'Baking all interiors', bake=True)

    def bake_changed(self):
        opts = self.bake_opts()
        if opts is None:
            return
        self.run(self.base_cmd() + ['bake', '--all', '--changed'] + opts,
                 'Re-baking cells that changed since their bake', bake=True)

    def check_changed(self, quiet=False):
        self.changed_ids = []
        self.run(self.base_cmd() + ['changed'] + self.mode_args(), 'Checking baked cells for changes',
                 on_line=self._changed_line, on_done=lambda: self._changed_done(quiet))

    def _changed_line(self, line):
        if ':' in line and not line.startswith('{'):
            self.changed_ids.append(line.strip())

    def _changed_done(self, quiet):
        ch = set(self.changed_ids)
        for c in self.cells:
            if c['id'] in ch and self.status.get(c['id']) in ('baked', 'changed'):
                self.status[c['id']] = 'changed'
            elif self.status.get(c['id']) == 'changed':
                self.status[c['id']] = 'baked'
        if ch or not quiet:
            self.write(f'{len(ch)} baked cell(s) changed since their bake (mods, mode or tool version). '
                       '"Bake changed" re-bakes only those.')
        self.refresh()

    def open_preview(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo('Radiance', 'Select a cell first.')
            return
        local, plugin = sel[0].split(':', 1)
        path = os.path.join(TOOL_DIR, 'reports', f'{plugin.lower()}_{local}.png')
        if os.path.isfile(path):
            os.startfile(path)
        else:
            self.write(f'No preview for {sel[0]} yet: tick "Render before/after preview" and bake it.')

    def open_output(self):
        if self.sess is None:
            return
        os.makedirs(self.sess.out, exist_ok=True)
        os.startfile(self.sess.out)

    def clear_output(self):
        """Delete everything the baker wrote into the output mod (probes, light patch, shader, SKSE plugin).
        Only the baker's own items: MO2's meta.ini and anything unknown stay."""
        if self.sess is None or self.proc:
            return
        out = self.sess.out
        items = [n for n in (os.listdir(out) if os.path.isdir(out) else []) if n in OUTPUT_ITEMS]
        if not items:
            messagebox.showinfo('Radiance', 'The output folder is already empty.')
            return
        n = sum(1 for s in self.status.values() if s)
        if not messagebox.askyesno('Radiance',
                                   f'Delete everything in the output folder?\n\n{out}\n\n'
                                   f'This removes {n} baked cell(s), the light patch, the shader patch and the SKSE '
                                   'plugin. In game everything goes back to your normal lighting until you bake '
                                   'again. This cannot be undone.'):
            return
        failed = []
        for name in items:
            path = os.path.join(out, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except OSError as e:
                failed.append(f'{name}: {e.strerror or e}')
        self.status = {}
        self._reload_cells()
        self._show_keys()
        self.refresh()
        if failed:
            self.write('Clear output folder: some files could not be deleted: ' + '; '.join(failed))
            messagebox.showerror('Radiance', 'Some files could not be deleted (is the game running?):\n\n'
                                 + '\n'.join(failed))
        else:
            self.write(f'Output folder cleared ({n} baked cell(s) removed).')

    # -- child process ---------------------------------------------------------------------------
    def _set_busy(self, busy):
        for b in self.buttons:
            b.configure(state='disabled' if busy else 'normal')
        for b in self.key_boxes:
            b.configure(state='disabled' if busy or self.sess is None else 'readonly')
        self.stop_btn.configure(state='normal' if busy and self.proc else 'disabled')

    def run(self, cmd, title, on_line=None, on_done=None, bake=False):
        if self.proc or self.sess is None:
            return
        self.write(f'--- {title} ---')
        self.on_line, self.on_done = on_line, on_done
        self.is_bake = bake
        self.stopped = False
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding='utf-8', errors='replace', creationflags=flags)
        self._set_busy(True)
        self.prog['value'] = 0
        self.prog_lbl.configure(text='')
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        for line in proc.stdout:
            self.q.put(('line', line.rstrip('\n')))
        proc.wait()
        self.q.put(('done', proc.returncode))

    def stop(self):
        if self.proc:
            self.stopped = True
            self.proc.kill()
            self.write('Stopping...')

    def close(self):
        if self.proc:
            if not messagebox.askyesno('Radiance', 'A bake is running. Stop it and close?\n'
                                       'Finished cells are kept.'):
                return
            self.closing = True
            self.stop()
            return                                 # closes once the light patch is brought up to date
        self._save_settings()
        self.closing = True
        self.root.destroy()

    def _finished(self, code):
        was_bake, stopped = self.is_bake, self.stopped
        self.proc = None
        if self.temp_file:
            try:
                os.remove(self.temp_file)
            except OSError:
                pass
            self.temp_file = None
        self.prog_lbl.configure(text='stopped' if stopped else ('finished' if code == 0 else f'exit code {code}'))
        on_done, self.on_done = self.on_done, None
        if on_done and not stopped:
            on_done()
        if was_bake:
            # cells baked by this run are current again
            old = self.baked_times
            self._reload_cells()
            for cid, t in self.baked_times.items():
                if old.get(cid) != t:
                    self.status[cid] = 'baked'
            if stopped:
                # a killed bake never wrote the light patch / shader for the cells it finished: do it now
                self.write('Updating the light patch and shader for the finished cells...')
                self.run(self.base_cmd() + ['finish'], 'Finishing')
                return
        self._set_busy(False)
        self.refresh()
        if self.closing:
            self._save_settings()
            self.root.destroy()

    def poll(self):
        try:
            while True:
                try:
                    kind, val = self.q.get_nowait()
                except queue.Empty:
                    break
                if kind == 'line':
                    self._line(val)
                else:
                    self._finished(val)
            self.root.after(100, self.poll)
        except tk.TclError:
            return                                 # window closed

    def _line(self, line):
        try:
            msg = json.loads(line)
        except ValueError:
            msg = None
        if isinstance(msg, dict):
            if 'log' in msg:
                self.write(msg['log'])
            if 'progress' in msg:
                self.prog['value'] = msg['progress']
                self.prog_lbl.configure(text=f"{msg['progress'] * 100:.0f}%  {msg.get('cell', '')}")
            return
        if self.on_line:
            self.on_line(line)
        else:
            self.write(line)

    def write(self, msg):
        self.log.insert('end', msg + '\n')
        self.log.see('end')


def run(args):
    try:
        # own taskbar entry and icon (not python's) when run from source
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('Radiance.BakedGlobalIllumination')
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    try:
        root.tk.call('tk', 'scaling', 1.3)
    except tk.TclError:
        pass
    try:
        # title bar, taskbar and every dialog (default=) use the Radiance icon instead of Tk's feather
        root.iconbitmap(default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'radiance.ico'))
    except tk.TclError:
        pass
    apply_dark_theme(root)
    App(root, args)
    root.mainloop()
    return 0
