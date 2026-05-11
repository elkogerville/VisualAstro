"""
Author: Elko Gerville-Reache
Date Created: 2026-05-11
Date Modified: 2026-05-11
Description:
    Tests for io module.
Dependencies:
    - pytest
"""

import pytest

from visualastro.core.config import config
from visualastro.core.io import _param, _pop_kwargs


class TestIo:

    def test_params(self):
        """Test _params utility function"""
        param = _param('name', 2, 3)
        assert isinstance(param, tuple)
        assert param[0] == 'name' and param[1] == 2 and param[2] == 3
        with pytest.raises(ValueError):
            _param(1, 2, 3)

    def test_pop_kwargs(self):
        kwargs = {
            'color': 'r',
            'lw': 2,
            'ls': '--',
        }
        value = _pop_kwargs(kwargs, 'linewidth', config.linewidth)

        assert kwargs == {'color': 'r', 'ls': '--'}
        assert value == 2

        value = _pop_kwargs(kwargs, 'nonsense', config.style)
        assert kwargs == {'color': 'r', 'ls': '--'}
        assert value == config.style
