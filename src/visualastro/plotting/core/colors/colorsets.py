"""
Author: Elko Gerville-Reache
Date Created: 2026-11-02
Date Modified: 2026-11-02
Description:
    VisualAstro colorset registry.
"""

from collections.abc import Callable, Iterator
from typing import Literal

import matplotlib as mpl

from visualastro.plotting.core.colors import (
    COLORSETS, COLORSET_ALIASES, plot_colors
)
