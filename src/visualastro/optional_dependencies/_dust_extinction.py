"""
Author: Elko Gerville-Reache
Date Created: 2026-07-27
Date Modified: 2026-07-27
Description:
    Optional dust_extinction package imports.
"""

from textwrap import dedent


try:
    from dust_extinction.parameter_averages import M14, G23
    from dust_extinction.grain_models import WD01
    _HAS_DUST_EXTINCTION = True
except ImportError:
    M14 = None
    G23 = None
    WD01 = None
    _HAS_DUST_EXTINCTION = False


_DUST_EXTINCTION_DEP = {
    'flag': _HAS_DUST_EXTINCTION,
    'msg': dedent("""\
        dust_extinction is required for this function.
        Install via:
            CONDA :
                $ conda install conda-forge::dust_extinction
            PIP :
                $ pip install dust_extinction
    """)
}
