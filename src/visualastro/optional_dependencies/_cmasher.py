"""
Author: Elko Gerville-Reache
Date Created: 2026-07-29
Date Modified: 2026-07-29
Description:
    Optional cmasher package imports.
"""

from textwrap import dedent


try:
    import cmasher
    _HAS_CMASHER = True
except ImportError:
    cmasher = None
    _HAS_CMASHER = False


_CMASHER_DEP = {
    'flag': _HAS_CMASHER,
    'msg': dedent("""\
        cmasher is available to add legend labels to this plot.
        Install via:
            CONDA :
                $ conda install conda-forge::colorspacious
            PIP :
                $ pip install colorspacious
    """)
}
