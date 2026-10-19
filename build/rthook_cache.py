# PyInstaller runtime hook: runs before any import. Keep numba's compiled-code cache inside the Radiance folder
# (otherwise a frozen app falls back to %LOCALAPPDATA%\numba, which deleting Radiance would leave behind).
import os
import sys

os.environ['NUMBA_CACHE_DIR'] = os.path.join(os.path.dirname(sys.executable), 'cache', 'compiled')
os.environ['NUMBA_CACHE_LOCATOR_CLASSES'] = 'radiance.numba_cache.RadianceCacheLocator'
