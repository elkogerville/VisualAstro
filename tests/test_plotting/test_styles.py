"""
Author: Elko Gerville-Reache
Date Created: 2026-10-25
Date Modified: 2026-10-25
Description:
    Tests for plot styles.
"""

import matplotlib.pyplot as plt
import pytest
import visualastro as va


class TestPlotStyles:
    def test_styles_available(self):
        va_styles = va.styles.available
        mpl_styles = plt.style.available
        assert len(va_styles) > 0
        assert all([vs in mpl_styles for vs in va_styles])
