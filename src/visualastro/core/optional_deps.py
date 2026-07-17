"""
Author: Elko Gerville-Reache
Date Created: 2026-07-17
Date Modified: 2026-07-17
Description:
    All optional dependencies are imported here.
    This centralizes the logic in visualastro for
    what packages are availble at runtime for a user.
"""

try:
    from spectral_cube import SpectralCube
    _HAS_SPECTRAL_CUBE = True
except ImportError:
    SpectralCube = None
    _HAS_SPECTRAL_CUBE = False
