"""
Author: Elko Gerville-Reache
Date Created: 2026-04-10
Date Modified: 2026-10-06
Description:
    Color transformation functions.
"""

from collections.abc import Sequence
import colorsys

from typing import Literal
from matplotlib import colors as mcolors
from matplotlib.typing import ColorType
import numpy as np

from visualastro.core.data import as_list, to_list
from visualastro.optional_dependencies.register import _require_dependency
from visualastro.optional_dependencies._colorspacious import cspace_convert
from visualastro.plotting.core.colors.utils import as_color, _convert_color
from visualastro.plotting.core.colors.definitions import RGBATuple, RGBTuple



def simulate_colorblindness(
    colors: ColorType | list[ColorType] | list[str | RGBTuple | RGBATuple],
    cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] = 'deuteranomaly',
    severity: int = 100,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> list[str | RGBTuple | RGBATuple]:
    """
    Simulate colorblindness perception of a color palette.

    Parameters
    ----------
    colors : ColorType | list[ColorType]
        Color or list of colors recognized by Matplotlib.
    cvd_type : {'deuteranomaly', 'protanomaly', 'tritanomaly'}, optional, default='deuteranomaly'
        Type of colorblindness to simulate. Can be shorthanded to {'d', 'p', 't'}.
    severity : int, optional, default=100
        Severity level (0-100). 100 = complete colorblindness.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    list[ColorType]
        List of ColorType as perceived by colorblind vision.
    """
    _require_dependency('colorspacious')
    if not 0 <= severity <= 100:
        raise ValueError(
            'severity must be >= 0 and <= 100!'
        )

    aliases = {
        'd': 'deuteranomaly',
        'p': 'protanomaly',
        't': 'tritanomaly'
    }
    colorblind_type = aliases.get(cvd_type, cvd_type)

    cvd_space = {
        'name': 'sRGB1+CVD',
        'cvd_type': colorblind_type,
        'severity': severity
    }

    # convert to RGB [0, 1]
    rgb = np.array(as_list(as_color(colors, fmt='rgb')))

    cvd_rgb = cspace_convert(rgb, cvd_space, 'sRGB1')
    cvd_rgb = np.clip(cvd_rgb, 0, 1)

    return [_convert_color(tuple(row), fmt=fmt) for row in cvd_rgb]


def _transform_colors(
    color: ColorType | Sequence[ColorType],
    transform: Literal['lighten', 'darken', 'saturate', 'desaturate'] | None = 'lighten',
    factor: int | float = 0.5,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> (
    str
    | RGBTuple
    | RGBATuple
    | list[str | RGBTuple | RGBATuple]
):
    """
    Lightens, darkens, saturates, or desaturates a color or list of
    colors. Mixes colors with white to lighten, mixes colors with
    black to darken, and moves colors towards grey to desaturate.

    `transform=None` returns color unchanged.

    Parameters
    ----------
    color : ColorType
        Matplotlib named color, hex color, HTML color, or RGB tuple.
    transform : {'lighten', 'darken', 'saturate', 'desaturate'} | None, optional, default='lighten'
        Method to modify the color. If `None`, returns `color` unchanged.
    factor : float | int, optional, default=0.5
        Modification strength. Must be between 0 and 1.

        * If `transform='lighten'`: Blending ratio with white.

            * `factor=0`: Original color
            * `factor=1`: Pure white

        * If `transform='darken'`: Blending ratio with black.

            * `factor=0`: Original color
            * `factor=1`: Pure black

        * If `transform='saturate'`: Saturation level in hsl space.

            * `factor=1`: Maximum saturation for each given color
            * `factor=0`: Grayscale

        * If `transform='desaturate'`: Desaturation amount.

            * `factor=0`: Original color
            * `factor=1`: Grayscale

    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    str | list[str] :
        If `fmt='hex'`.
    tuple[float, float, float] | list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    tuple[float, float, float, float] | list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.

    Raises
    ------
    ValueError
        If `factor` is not between 0 and 1.
    """
    if transform:
        method = {
            'lighten': _lighten_color,
            'darken': _darken_color,
            'saturate': _saturate_color,
            'desaturate': _desaturate_color,
        }.get(transform, None)
    else:
        method = None

    if transform is None or method is None:
        return as_color(color, fmt=fmt)

    colors = to_list(color)
    colors = [method(c, factor) for c in colors]

    return as_color(colors, fmt=fmt)


def lighten_colors(
    color: ColorType | Sequence[ColorType],
    factor: int | float = 0.5,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> str | RGBTuple | RGBATuple | list[str | RGBTuple | RGBATuple]:
    """
    Lighten a set of color(s) by mixing the each color with white.

    Parameters
    ----------
    color : ColorType | Sequence[ColorType]
        Color(s) to lighten.
    factor : int | float, optional, default=0.5
        Mixing factor. `1` results in all white, while `0`
        leaves `color` unchanged. Must be between 0 and 1.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    str | list[str] :
        If `fmt='hex'`.
    tuple[float, float, float] | list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    tuple[float, float, float, float] | list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.

    Raises
    ------
    ValueError
        If `factor` is not between 0 and 1.
    """
    if factor < 0 or factor > 1:
        raise ValueError('mixing factor with white must be between 0 and 1')

    colors = to_list(color)
    return as_color([_lighten_color(c, factor=factor) for c in colors], fmt=fmt)


def _lighten_color(color: ColorType, factor: int | float = 0.5) -> 'str':
    """Lightens the given Matplotlib color by mixing it with white."""
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    mixed = (1 - factor) * rgb + factor * white

    return mcolors.to_hex(tuple(mixed))


def darken_colors(
    color: ColorType | Sequence[ColorType],
    factor: int | float = 0.5,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> str | RGBTuple | RGBATuple | list[str | RGBTuple | RGBATuple]:
    """
    Darken a set of color(s) by mixing the each color with black.

    Parameters
    ----------
    color : ColorType | Sequence[ColorType]
        Color(s) to darken.
    factor : int | float, optional, default=0.5
        Mixing factor. `1` results in all black, while `0`
        leaves `color` unchanged. Must be between 0 and 1.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    str | list[str] :
        If `fmt='hex'`.
    tuple[float, float, float] | list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    tuple[float, float, float, float] | list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.

    Raises
    ------
    ValueError
        If `factor` is not between 0 and 1.
    """
    if factor < 0 or factor > 1:
        raise ValueError('mixing factor with black must be between 0 and 1')

    colors = to_list(color)
    return as_color([_darken_color(c, factor=factor) for c in colors], fmt=fmt)


def _darken_color(color: ColorType, factor: int | float = 0.5) -> 'str':
    """Darkens the given Matplotlib color by mixing it with black."""
    rgb = np.array(mcolors.to_rgb(color))
    black = np.array([0, 0, 0])
    mixed = (1 - factor) * rgb + factor * black

    return mcolors.to_hex(tuple(mixed))


def saturate_colors(
    color: ColorType | Sequence[ColorType],
    factor: int | float = 1,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> str | RGBTuple | RGBATuple | list[str | RGBTuple | RGBATuple]:
    """
    Saturate a set of color(s) by modifying the saturation level
    of each color in hls space.

    Parameters
    ----------
    color : ColorType | Sequence[ColorType]
        Color(s) to saturate.
    factor : int | float, optional, default=1
        Saturation level. `1` results in maximum saturation, while `0`
        returns `color` in grayscale. Must be between 0 and 1.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    str | list[str] :
        If `fmt='hex'`.
    tuple[float, float, float] | list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    tuple[float, float, float, float] | list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.

    Raises
    ------
    ValueError
        If `factor` is not between 0 and 1.
    """
    if factor < 0 or factor > 1:
        raise ValueError('saturation factor must be between 0 and 1')

    colors = to_list(color)
    return as_color(
        [_saturate_color(c, factor=factor) for c in colors], fmt=fmt
    )


def _saturate_color(color: ColorType, factor: int | float = 1) -> str:
    """Saturate a color by shifting the saturation in hls space."""
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    rgb_new = colorsys.hls_to_rgb(h, l, factor)

    return mcolors.to_hex(rgb_new)


def desaturate_colors(
    color: ColorType | Sequence[ColorType],
    factor: int | float = 0.5,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> str | RGBTuple | RGBATuple | list[str | RGBTuple | RGBATuple]:
    """
    Desaturate a set of color(s) by moving each color towards gray
    in hsl space.

    Parameters
    ----------
    color : ColorType | Sequence[ColorType]
        Color(s) to desaturate.
    factor : int | float, optional, default=0.5
        Desaturation level. `1` returns `color` in grayscale while
        `0` returns the colors unchanged. Must be between 0 and 1.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    str | list[str] :
        If `fmt='hex'`.
    tuple[float, float, float] | list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    tuple[float, float, float, float] | list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.

    Raises
    ------
    ValueError
        If `factor` is not between 0 and 1.
    """
    if factor < 0 or factor > 1:
        raise ValueError('desaturation factor must be between 0 and 1')

    colors = to_list(color)
    return as_color(
        [_desaturate_color(c, factor=factor) for c in colors], fmt=fmt
    )


def _desaturate_color(color: ColorType, factor: int | float = 0.5) -> str:
    """Desaturate a color by moving it toward gray."""
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s_new = s * (1 - factor)
    rgb_new = colorsys.hls_to_rgb(h, l, s_new)

    return mcolors.to_hex(rgb_new)
