"""
Author: Elko Gerville-Reache
Date Created: 2026-07-27
Date Modified: 2026-07-27
Description:
    Optional spectral-cube package imports.
"""

from textwrap import dedent


try:
    from spectral_cube import SpectralCube
    from spectral_cube.lower_dimensional_structures import Slice
    _HAS_SPECTRAL_CUBE = True
except ImportError:
    SpectralCube = None
    Slice = None
    _HAS_SPECTRAL_CUBE = False


_SPECTRALCUBE_DEP = {
    'flag': _HAS_SPECTRAL_CUBE,
    'msg': dedent("""\
        spectral-cube is required for this function.
        Install via:
            CONDA :
                $ conda install conda-forge::spectral-cube
            PIP :
                $ pip install spectral-cube
    """)
}
