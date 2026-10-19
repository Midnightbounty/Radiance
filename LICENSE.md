# Licences

## SKSE plugin (`plugin/`): GPL-3.0-or-later

`Radiance.dll` is built on CommonLibSSE NG / CommonLibVR, which is licensed GPL-3.0-or-later with a modding
exception, so the plugin source in `plugin/` is licensed **GPL-3.0-or-later** as well. The full licence text is in
[plugin/COPYING.txt](plugin/COPYING.txt).

## Everything else: source available, all rights reserved

The baker (`Radiance.py`, `radiance/`, `build/`) is published so that it can be reviewed and built. You may read it,
build it and run your own build for personal use. Please do not redistribute it or reuse it in other projects
without asking the author (MidnightBounty on Nexus Mods). Including the official release in mod lists, including
Wabbajack lists, is fine.

Third-party components bundled into the release (Python, NumPy, Numba/llvmlite/LLVM, Pillow, Tcl/Tk, spdlog, {fmt},
zlib) keep their own licences; the release includes them in its `LICENSES` folder.
