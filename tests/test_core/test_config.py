"""
Author: Elko Gerville-Reache
Date Created: 2026-05-11
Date Modified: 2026-05-11
Description:
    Tests for config module.
"""

from visualastro.core.config import (
    config,
    VisualAstroConfig,
    _resolve_default,
    _Unset,
    _UNSET
)


class TestConfig:

    def test_config(self):
        assert isinstance(_UNSET, _Unset)
        assert isinstance(config, VisualAstroConfig)

        default_style = config.style
        config.style = 'nan'
        assert config.style == 'nan'

        default_color = config.colors
        config.colors = 'b'
        assert config.colors == 'b'

        config.reset()
        assert config.style == default_style and config.colors == default_color

    def test_resolve_default(self):
        param = _UNSET
        value = _resolve_default(param, config.style)
        assert value == config.style

        param = 3
        value = _resolve_default(param, 4)
        assert value == 3
