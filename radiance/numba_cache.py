"""Compiled-code cache for the packaged Radiance.exe.

numba normally caches next to each source file. In the packaged exe the source paths it sees are relative to the
bundle, so it would fall back to %LOCALAPPDATA%\\numba (left behind when Radiance is deleted). This locator resolves
them inside the bundle and keeps the cache in <Radiance>\\cache\\compiled (NUMBA_CACHE_DIR, set at startup by
build/rthook_cache.py together with NUMBA_CACHE_LOCATOR_CLASSES pointing here).
"""
import os
import sys

from numba.core.caching import UserProvidedCacheLocator


class RadianceCacheLocator(UserProvidedCacheLocator):
    @classmethod
    def from_function(cls, py_func, py_file):
        if not os.path.isabs(py_file):
            py_file = os.path.join(getattr(sys, '_MEIPASS', os.path.dirname(sys.executable)), py_file)
        return super().from_function(py_func, py_file)
