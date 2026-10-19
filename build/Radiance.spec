# PyInstaller build of Radiance: Radiance.exe (window) + RadianceCLI.exe (console), one shared runtime folder.
#   pyinstaller build/Radiance.spec --distpath <dist> --workpath <work>
# The package's .py files are shipped next to the bytecode: numba caches compiled code per source file.
import glob
import os

ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))
datas = [(f, 'radiance') for f in glob.glob(os.path.join(ROOT, 'radiance', '*.py'))]
datas += [(os.path.join(ROOT, 'radiance', 'shader', 'Radiance.hlsli'), 'radiance/shader'),
          (os.path.join(ROOT, 'radiance', 'radiance.ico'), 'radiance')]
ICON = os.path.join(ROOT, 'radiance', 'radiance.ico')     # build/make_icon.py
excludes = ['matplotlib', 'scipy', 'pandas', 'IPython', 'jupyter', 'notebook', 'pytest', 'setuptools',
            'pkg_resources', 'docutils', 'sphinx', 'cv2', 'torch', 'tensorflow', 'yaml', 'jinja2', 'markupsafe']

a = Analysis([os.path.join(ROOT, 'Radiance.py')], pathex=[ROOT], datas=datas,
             hiddenimports=['radiance.cli', 'radiance.gui', 'radiance.numba_cache'], excludes=excludes, noarchive=False,
             runtime_hooks=[os.path.join(SPECPATH, 'rthook_cache.py')])
pyz = PYZ(a.pure)
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Radiance', console=False, upx=False, icon=ICON)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name='RadianceCLI', console=True, upx=False, icon=ICON)
coll = COLLECT(gui, cli, a.binaries, a.datas, upx=False, name='Radiance')
