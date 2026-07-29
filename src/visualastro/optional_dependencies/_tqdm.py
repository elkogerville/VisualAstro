"""
Author: Audrey Whitmer
Date Created: 2026-07-27
Date Modified: 2026-07-28
Description:
    Optional tqdm package imports.
"""

from textwrap import dedent


try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
    _HAS_TQDM = False


_TQDM_DEP = {
    'flag': _HAS_TQDM,
    'msg': dedent("""\
        tqdm is available to add a progress bar to this operation!
        Install via:
            CONDA :
                $ conda install -c conda-forge tqdm
            PIP :
                $ pip install tqdm
    """)
}
