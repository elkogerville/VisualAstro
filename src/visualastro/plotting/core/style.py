"""
Author: Elko Gerville-Reache
Date Created: 2025-07-13
Date Modified: 2026-07-03
Description:
    Functions related to setting the plotting style.
"""

from contextlib import nullcontext

import matplotlib.pyplot as plt


def _style_context(stylepath):
    """
    Helper function to facilitate
    """
    return (
        plt.style.context(stylepath) if stylepath is not None else nullcontext()
    )
