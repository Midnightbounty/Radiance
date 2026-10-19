# Radiance - Baked Global Illumination for Interiors

Source code of **Radiance**, a Skyrim (SE / AE / VR) tool that path traces the light in interior cells and bakes it
for the user's own load order, plus the SKSE plugin that shows the result in game through a Community Shaders
(CSX, Open Shaders, Community Shaders) shader patch.

Download and user guide: Nexus Mods, "Radiance - Baked Global Illumination for Interiors".

## Layout

| Path | What it is |
|---|---|
| `Radiance.py` | Entry point of the baker (window and command line) |
| `radiance/` | The baker: plugin/BSA/NIF/DDS readers, scene builder, path tracer (numba), probe volumes, light patch, shader patch, GUI |
| `radiance/shader/Radiance.hlsli` | Shader include installed next to the patched Community Shaders `Lighting.hlsl` |
| `plugin/` | SKSE plugin `Radiance.dll` (C++, CommonLibVR) |
| `build/` | PyInstaller spec and runtime hook for `Radiance.exe` / `RadianceCLI.exe`, icon generator, end-user README |
| `light_rules.example.json` | Optional per-light overrides |

How to build everything: [BUILD.md](BUILD.md). Licences: [LICENSE.md](LICENSE.md).
