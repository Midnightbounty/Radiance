RADIANCE - Baked Global Illumination for Interiors
Version 0.5.0 beta  -  Skyrim VR (tested), Skyrim SE / AE (built for, not yet tested)

Radiance path traces the light in Skyrim's interiors and bakes it into your load order, the way DynDOLOD
generates LOD: you run the tool once, it writes an output mod, and in game every baked interior gets real
bounced light, colour bleeding and soft shadows instead of the flat ambient light and fake fill lights.
It works with whatever lighting mod you use; it reads your actual load order.


REQUIREMENTS
- Mod Organizer 2
- SKSE (SKSE VR for Skyrim VR)
- Address Library for SKSE Plugins (VR Address Library for Skyrim VR)
- One of these (Radiance patches whichever wins in your load order):
    Community Shaders Expanded (CSX)  - Skyrim VR, SE, AE
    Open Shaders                      - Skyrim VR, SE, AE
    Community Shaders                 - Skyrim SE, AE (it no longer supports VR)
  Other forks should work too: Radiance adapts its shader patch to the installed version.
- Windows 10 or 11, 64-bit


INSTALL
1. Extract the "Radiance" folder anywhere outside the game and outside "Program Files",
   for example  C:\Modding\Radiance  or a "tools" folder in your MO2 folder.
2. In MO2: Tools (the gears icon) > Executables > "+" > Add from file... > select Radiance.exe.
3. Start Radiance from MO2's executable list. The first start reads your load order (a few seconds).
   If you start Radiance outside MO2, it asks once for your MO2 folder (the one with ModOrganizer.ini).


USE
1. Pick the cells to bake (search box, filters, Ctrl/Shift-click) and press "Bake selected", or
   "Bake all interiors" (several hours for a large load order; you can stop at any time).
   The very first bake takes about a minute longer: Radiance compiles its light transport code once.
2. Mode: "Quality" (default) or "Performance" (fewer shadow-casting lights, for VR frame rate).
3. Radiance writes the mod "Radiance Output" into your MO2 mods folder. In MO2:
   - press Refresh (F5), enable "Radiance Output" in the left pane,
   - it must be BELOW (win over) your Community Shaders mod,
   - tick Radiance.esp in the right pane.
4. If the lighting does not change in game, clear the Community Shaders shader cache once
   (Community Shaders in-game menu: "Clear Shader Cache") and restart the game.

In game:
- F7 switches between Radiance and your original lighting, so you can compare.
- F8 cycles a debug view: off / magenta where Radiance is active / baked light only.
- Both keys can be changed in the Radiance window ("In-game keys").

Bake again ("Bake changed") after you change lighting or interior mods, and after updating your
Community Shaders mod (any bake re-patches the shader). Player homes you build later (Hearthfire)
need a bake after building.

"Clear output folder" deletes everything Radiance wrote, if you want to start over.


WHAT TO EXPECT
- Light comes from where it really is: pools of light near fires and windows, darker corners,
  colour bleeding from walls and floors. Each room keeps about the same overall brightness as your
  lighting mod intended.
- Rooms that were lit mostly by light shining through walls (unshadowed window lights) get darker,
  because that light cannot physically reach those places.
- Some lights are hidden while Radiance is on (fake fill lights and leaking window lights that the bake
  replaces). They are only hidden, never deleted: F7 shows them again, and nothing is written into
  your save games.
- Performance: usually the same or better than before (fewer shadow-casting lights).


KNOWN LIMITATIONS (BETA)
- Tested in game on Skyrim VR with Community Shaders Expanded. The shader patch is verified for CSX 3.20,
  Open Shaders 2.19 and Community Shaders 1.9 (VR and SE builds compile cleanly), but SE/AE and Open Shaders
  have not been played with yet - reports welcome.
- Baked light is soft (about 1 m resolution); sharp shadows come from the game's own shadowed lights.
- Light can bleed slightly through very thin walls between a bright and a dark room.
- Lights that scripts switch on and off are left as they are (not baked).
- Exteriors are not baked.


UNINSTALL
Disable or delete "Radiance Output" in MO2 and delete the Radiance folder. Nothing else is changed.


BUG REPORTS - please include:
- RadianceBaker.log (in the Radiance folder) and the text from the Radiance window's log,
- Radiance.log from Documents\My Games\Skyrim VR\SKSE (or Skyrim Special Edition\SKSE),
- your Community Shaders version and the cell you were in.


FILES
Radiance.exe       the Radiance window
RadianceCLI.exe    the same program for the command line (the window uses it to run bakes);
                   RadianceCLI.exe --help lists the commands
runtime\           the SKSE plugin Radiance.dll, copied into "Radiance Output" by every bake
source\            source code of the SKSE plugin (GPL-3.0, see LICENSES)
LICENSES\          licences of Radiance's components
light_rules.example.json   optional per-light overrides (copy to light_rules.json to use)
cache\, reports\, config.json, gui_settings.json   created while you use Radiance; safe to delete
