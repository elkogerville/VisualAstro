"""
Author: Elko Gerville-Reache
Date Created: 2026-07-27
Date Modified: 2026-07-27
Description:
    Optional colorspacious package imports.
"""

from textwrap import dedent


try:
    from colorspacious import cspace_convert, cspace_converter, deltaE
    _HAS_COLORSPACIOUS = True
except ImportError:
    cspace_convert = None
    cspace_converter = None
    deltaE = None
    _HAS_COLORSPACIOUS = False


_COLORSPACIOUS_DEP = {
    'flag': _HAS_COLORSPACIOUS,
    'msg': dedent("""\
        colorspacious is required for this function.
        Install via:
            CONDA :
                $ conda install conda-forge::colorspacious
            PIP :
                $ pip install colorspacious
    """)
}
