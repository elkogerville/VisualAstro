"""
Author: Elko Gerville-Reache
Date Created: 2026-07-27
Date Modified: 2026-07-27
Description:
    Optional astropy regions package imports.
"""

from textwrap import dedent


try:
    from regions import (
        PixCoord, EllipseAnnulusPixelRegion, EllipsePixelRegion
    )
    _HAS_REGIONS = True
except ImportError:
    PixCoord = None
    EllipseAnnulusPixelRegion = None
    EllipsePixelRegion = None
    _HAS_REGIONS = False


_REGIONS_DEP = {
    'flag': _HAS_REGIONS,
    'msg': dedent("""\
        astropy regions is required for this function.
        Install via:
            CONDA :
                $ conda install -c conda-forge regions
            PIP :
                $ python -m pip install regions
    """)
}
