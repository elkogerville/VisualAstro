"""
Author: Elko Gerville-Reache
Date Created: 2026-07-20
Date Modified: 2026-07-20
Description:
    Optional specutils package imports.
"""

from textwrap import dedent


try:
    from specutils import SpectralAxis, SpectralRegion, Spectrum
    from specutils.fitting import fit_continuum as _fit_continuum
    from specutils.fitting import fit_generic_continuum as _fit_generic
    _HAS_SPECUTILS = True
except ImportError:
    SpectralAxis = None
    SpectralRegion = None
    Spectrum = None
    _fit_continuum = None
    _fit_generic = None
    _HAS_SPECUTILS = False


SPECUTILS_DEP = {
    'flag': _HAS_SPECUTILS,
    'msg': dedent("""\
        spectral-cube is required for this function.
        Install via:
            CONDA :
                $ conda install -c conda-forge specutils
            PIP :
                $ pip install specutils
    """)
}
