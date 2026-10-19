"""Radiance - baked global illumination for Skyrim interiors. Run without arguments for the cell picker.

Radiance.exe opens the window; RadianceCLI.exe is the same program for the command line (the window also uses it
to run bakes in the background). Without a console, output and errors go to RadianceBaker.log next to the exe.
"""
import os
import sys
import traceback

FROZEN = getattr(sys, 'frozen', False)
HERE = os.path.dirname(sys.executable) if FROZEN else os.path.dirname(os.path.abspath(__file__))
if not FROZEN:
    sys.path.insert(0, HERE)
# frozen: build/rthook_cache.py keeps numba's compiled-code cache in <Radiance>\cache\compiled

if __name__ == '__main__':
    if sys.stdout is None or sys.stderr is None:      # Radiance.exe / pythonw.exe: no console
        log = open(os.path.join(HERE, 'RadianceBaker.log'), 'w', encoding='utf-8', buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log
    try:
        from radiance.cli import main
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        msg = traceback.format_exc()
        print(msg, file=sys.stderr)
        try:
            import tkinter.messagebox
            tkinter.messagebox.showerror('Radiance', msg[-1500:])
        except Exception:      # noqa: BLE001
            pass
        sys.exit(1)
