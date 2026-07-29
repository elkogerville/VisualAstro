"""
Author: Elko Gerville-Reache
Date Created: 2026-07-28
Date Modified: 2026-07-28
Description:
    Optional reproject package imports.
"""

from textwrap import dedent


try:
    from reproject import reproject_exact, reproject_interp
    _HAS_REPROJECT = True
except ImportError:
    reproject_exact = None
    reproject_interp = None
    _HAS_REPROJECT = False


_REPROJECT_DEP = {
    'flag': _HAS_REPROJECT,
    'msg': dedent("""\
        reproject is required for this function.
        Install via:
            CONDA :
                $ conda install -c conda-forge reproject
            PIP :
                $ pip install reproject
    """)
}
