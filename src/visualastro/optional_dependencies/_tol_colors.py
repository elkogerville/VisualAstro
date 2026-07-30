"""
Author: Elko Gerville-Reache
Date Created: 2026-07-29
Date Modified: 2026-07-29
Description:
    Optional tol-colors package imports.
"""

from textwrap import dedent


try:
    import tol_colors
    _HAS_TOLCOLORS = True
except ImportError:
    tol_colors = None
    _HAS_TOLCOLORS = False
