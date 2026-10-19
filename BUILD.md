# Building Radiance

The release contains two programs built from this repository:

- `Radiance.exe` / `RadianceCLI.exe`: the baker (Python, packaged with PyInstaller)
- `runtime\Radiance.dll`: the SKSE plugin (C++)

All steps are on Windows 10/11, 64-bit.

## 1. Baker (Radiance.exe, RadianceCLI.exe)

Requirements: Python 3.11 (64-bit).

```bat
python -m pip install -r requirements.txt
python Radiance.py                         :: runs the baker from source (window)
python Radiance.py --help                  :: command line
```

Package it the same way as the release (use a short output path; long paths break PyInstaller's icon/resource step):

```bat
python -m PyInstaller build\Radiance.spec --noconfirm --distpath C:\rbuild\dist --workpath C:\rbuild\work
```

The result is `C:\rbuild\dist\Radiance\` with `Radiance.exe` (window), `RadianceCLI.exe` (console; the window runs
bakes through it) and `_internal\` (Python runtime, numpy, numba/LLVM, Pillow, Tcl/Tk and the `radiance` package,
including its `.py` sources, which numba needs for its compiled-code cache).

`build\rthook_cache.py` and `radiance\numba_cache.py` keep numba's compiled-code cache in `<Radiance>\cache\compiled`
instead of `%LOCALAPPDATA%`. `build\make_icon.py` regenerates `radiance\radiance.ico`.

## 2. SKSE plugin (Radiance.dll)

Requirements: Visual Studio 2022 or newer with "Desktop development with C++" (includes CMake, Ninja and vcpkg), git.

```powershell
powershell -ExecutionPolicy Bypass -File plugin\build.ps1
```

`build.ps1` clones CommonLibVR (https://github.com/alandtse/CommonLibVR) into its build folder (default `C:\rb`),
installs the vcpkg dependencies (zlib, spdlog, fmt, ...) and builds `C:\rb\out\Radiance.dll`, one DLL for Skyrim SE,
AE and VR. The build folder is a short, neutral path on purpose: vcpkg/MSVC break on long paths, and source paths end
up inside the DLL's error messages.

## 3. Release folder

```
Radiance\
  Radiance.exe, RadianceCLI.exe, _internal\      <- C:\rbuild\dist\Radiance\
  runtime\Radiance.dll                           <- C:\rb\out\Radiance.dll
  README.txt                                     <- build\README.txt
  light_rules.example.json
  LICENSES\                                      <- licences of the bundled components
  source\                                        <- plugin\ and radiance\shader\Radiance.hlsli
```

The baker copies `runtime\Radiance.dll` into the user's output mod on every bake.
