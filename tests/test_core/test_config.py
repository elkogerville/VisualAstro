"""
Author: Elko Gerville-Reache
Date Created: 2026-05-11
Date Modified: 2026-05-11
Description:
    Tests for config module.
"""

from visualastro.core.configuration import (
    _UNSET,
    VisualAstroConfig,
    _resolve_default,
    _Unset,
    config,
)


class TestConfig:

    def test_config(self):
        assert isinstance(_UNSET, _Unset)
        assert isinstance(config, VisualAstroConfig)

        default_style = config.style
        config.style = 'nan'
        assert config.style == 'nan'

        default_color = config.color
        config.color = 'b'
        assert config.color == 'b'

        config.reset()
        assert config.style == default_style and config.color == default_color

    def test_resolve_default(self):
        param = _UNSET
        value = _resolve_default(param, config.style)
        assert value == config.style

        param = 3
        value = _resolve_default(param, 4)
        assert value == 3

    def test_reset(self):
        figsize = config.figsize
        config.figsize = (200, 200)
        assert config.figsize == (200, 200)
        assert config.figsize != figsize
        config.reset()
        assert config.figsize == figsize

    def test_subconfig_reset(self):
        config.figsize = (700, 700)
        config.axes.xlog = 70
        assert config.colorbar.label
        config.colorbar.label = False
        assert not config.colorbar.label

        config.colorbar.reset()
        assert config.colorbar.label
        assert config.figsize == (700, 700)
        assert config.axes.xlog == 70
        config.reset()
