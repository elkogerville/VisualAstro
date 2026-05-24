"""
Author: Elko Gerville-Reache
Date Created: 2025-10-20
Date Modified: 2026-04-08
Description:
    Visualastro configuration interface to update function defaults.
Dependencies:
    - astropy
    - matplotlib
    - numpy
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Literal, TypeVar

from astropy.wcs import WCS
from astropy.io.fits import Header
import astropy.units as u
from astropy.units import physical
from matplotlib.colors import Colormap
from matplotlib.transforms import Bbox
from matplotlib.typing import ColorType
import numpy as np
from numpy.typing import DTypeLike


T = TypeVar('T')

class _Unset(Enum):
    """
    Default placeholder sentinel value for
    visualastro functions.
    """
    UNSET = 'UNSET'

_UNSET = _Unset.UNSET


@dataclass(slots=True)
class AxesConfig:
    """matplotlib.axes config"""
    xpad: float = 0.05  # set_axis_limits() xpad
    ypad: float = 0.05 # set_axis_limits() ypad
    xlog: bool = False
    ylog: bool = False
    xlog_hist: bool = True
    ylog_hist: bool = True
    sharex: bool = False
    sharey: bool = False
    hspace = None
    wspace = None
    Nticks = None
    aspect = None

@dataclass(slots=True)
class ZorderLayers:
    gridlines: float = 0
    wcs_grid: float = 1
    contourf: float = 10
    plot_data: float = 20
    contour: float = 30
    vlines: float = 40
    hlines: float = 40
    regions: float = 50
    text: float = 60

@dataclass(slots=True)
class AXLineConfig:
    """ax.vline / ax.hline config"""
    linestyle: Literal['-', '--', '-.', ':', ''] = ':'
    linewidth: float = 1.0
    color: ColorType = 'k'
    alpha: float | None = 0.7
    zorder: float = 0

@dataclass(slots=True)
class LegendConfig:
    """ax.legend config"""
    handles: Sequence | None = None
    labels: Sequence[str] | None = None
    loc: str = 'best'
    ncols: int = 1
    fontsize: int | Literal['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'] = 13
    fancybox: bool = False
    framealpha: float = 0.8
    facecolor: Literal['inherit'] | ColorType = 'inherit'
    edgecolor: Literal['inherit'] | ColorType = 'w'
    title: str | None = None
    alignment: Literal['center', 'left', 'right'] = 'center'
    columnspacing: float = 2
    draggable: bool = True

@dataclass(slots=True)
class SavefigConfig:
    """plt.savefig config"""
    enabled: bool = False
    dpi: float = 600
    pdf_compression: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = 6
    transparent: bool = False
    bbox_inches: str | Bbox | None = 'tight'
    allowed_formats = {'eps', 'pdf', 'png', 'svg'}

@dataclass(slots=True)
class CurveFitConfig:
    """scipy.curve_fit config"""
    method: Literal['lm', 'trf', 'dogbox'] = 'trf'
    absolute_sigma: bool = False
    samples: int = 10000
    interpolate: bool = False
    interpolation_method: Literal['linear', 'cubic', 'cubic_spline'] = 'cubic_spline'
    error_interpolation_method: Literal['linear', 'cubic', 'cubic_spline'] = 'cubic_spline'

@dataclass(slots=True)
class ErrorBarConfig:
    """ax.errorbar config"""
    fmt: str = 'none' # use 'none' to plot errorbars without any data markers.
    colors: ColorType | None = None
    linewidth: float | None = 1
    capsize: float = 3 # cap length in points
    capthick: float | None = 1 # cap thickness in points
    barsabove: bool = False
    markeredgecolor: ColorType | None = None

@dataclass(slots=True)
class SpectralLineConfig:
    """spectral_line_marker config"""
    marker_direction: Literal['up', 'down'] = 'down'
    label_offset_points: tuple[float, float] = (0.0, 4.0)
    label_position: Literal['center', 'left', 'right'] = 'center'
    label_anchor: Literal['center', 'left', 'right', 'auto'] = 'auto'
    label_reference: Literal['marker', 'hline', 'auto'] = 'auto'
    label_rotation: float = 0

@dataclass(slots=True)
class HDUConfig:
    """HDU config"""
    index: int = 0
    error_extensions: list[str] = field(
        default_factory=lambda: [
        'ERR', 'ERROR', 'UNCERT'
    ])
    variance_extensions: list[str] = field(
        default_factory=lambda: [
        'VAR', 'VARIANCE', 'VAR_POISSON', 'VAR_RNOISE', 'STAT'
    ])


@dataclass(slots=True)
class VisualAstroConfig:
    """
    Global configuration object for controlling default behavior
    across the visualastro package.

    visualastro function parameters are often set to `_UNSET`,
    which at runtime gets resolved to the default hardcoded value
    set in `VisualAstroConfig`. Modifying this file will update
    the default values.

    Users can also modify attributes at runtime by modifying the config
    object attributes. This avoids permanently changing the default values.
    ie:
        >>> import visualastro as va
        >>> va.config.style = 'latex'
        >>> va.config.figsize = (6, 6)
    """
    # I/O params
    unit_mismatch: Literal['warn', 'ignore', 'raise'] | None = 'warn'
    default_dtype: DTypeLike = np.float64
    hdu_idx: int = 0
    print_info: bool = False
    transpose: bool = False
    mask_non_positive: bool = False
    mask_out_value: float = np.nan
    invert_wcs_if_transpose: bool = True
    target_wcs: Header | WCS | None = None
    hdu: HDUConfig = field(default_factory=HDUConfig)
    array_order: Literal['C', 'c', 'F', 'f', 'fortran'] = 'c'

    # figure params
    style: str = 'astro' # default style
    style_fallback: str = 'default' # style if default style fails
    figsize: tuple = (6, 6)
    reference_idx: int = 0 # which index is considered the reference for plot labels, cbars, etc..
    grid_figsize: tuple = (12, 6)
    figsize3d: tuple = (10, 10)
    # if _UNSET, defaults to `self.default_colorset`.
    # To define a custom default colorset,
    # define it in `get_colors` and change the `default_colorset`.
    colors: ColorType | int | Sequence[ColorType] | _Unset = _UNSET
    default_colorset: str = 'ibm_contrast' # see `get_colors` in plot_utils.py
    alpha: int = 1
    nrows: int = 1 # make_grid_plot() nrows
    ncols: int = 2 # make_grid_plot() ncols
    rasterized: bool = False # rasterize plot artists wherever possible
    show_plot: bool = True

    # figure grid params
    gridlines: bool = False
    grid_which: Literal['major', 'minor', 'both'] = 'major'
    grid_color: ColorType = 'k'
    grid_linestyle: 'str' = 'dotted'
    grid_linewidth: float = 0.8
    grid_alpha: float = 0.8

    wcs_grid: bool = False
    wcs_grid_color: ColorType = 'w'
    wcs_grid_linestyle: 'str' = 'dotted'
    wcs_grid_linewidth: float = 0.8
    wcs_grid_alpha: float = 0.8

    # data params
    normalize_data: bool = False

    # histogram params
    histtype: Literal['bar', 'barstacked', 'step', 'stepfilled'] = 'step'
    bins: int | Sequence[float] | str = 'auto'
    normalize_hist: bool = True

    # line2D params
    linestyle: Literal['solid', '-', 'dotted', ':', 'dashed', '--', 'dashdot', '-.'] = '-'
    linewidth: float = 0.8

    axline: AXLineConfig = field(default_factory=AXLineConfig)

    # scatter params
    scatter_size: float = 10
    marker: str = 'o'
    edgecolor: ColorType | None = None
    facecolor: ColorType | None = None

    # errorbar params
    errorbar: ErrorBarConfig = field(default_factory=ErrorBarConfig)

    # imshow params
    cmap: Colormap | str = 'turbo'
    origin: Literal['lower', 'upper'] = 'lower'
    norm: Literal['asinh', 'asinhnorm', 'log', 'power'] | None = 'asinh'
    linear_width: float = 1 # AsinhNorm linear width
    gamma: float = 0.5 # PowerNorm exponent
    vmin: float | None = None
    vmax: float | None = None
    percentile: tuple[float, float] | None = (3.0, 99.5)
    aspect: Literal['auto', 'equal'] | float | None = None

    axes: AxesConfig = field(default_factory=AxesConfig)
    zorder: ZorderLayers = field(default_factory=ZorderLayers)
    legend: LegendConfig = field(default_factory=LegendConfig)
    savefig: SavefigConfig = field(default_factory=SavefigConfig)

    # cbar params
    cbar: bool = True
    cbar_width: float = 0.03
    cbar_pad: float = 0.015
    cbar_tick_which = 'both'
    cbar_tick_dir = 'out'
    clabel = True

    # text params
    fontsize: float = 10
    text_color: str = 'k'
    text_loc: tuple[float, float] = (0.03, 0.03)

    # label params
    unit_bracket_style: Literal['round', 'square'] = 'square' # display units as [unit] instead of (unit)
    _unit_bracket_styles = {
        'round': ('(', ')'),
        'square': ('[', ']'),
    }
    right_ascension: str = 'Right Ascension'
    declination: str = 'Declination'
    highlight: bool = True
    show_type_label: bool = False
    show_unit_label: bool = True
    unit_label_format: Literal['latex', 'latex_inline', 'fits', 'unicode', 'console', 'vounit', 'cds', 'ogip'] = 'latex_inline'
    _PHYSICAL_TYPE_LABELS = {
        u.adu.physical_type: 'ADU',          # type: ignore
        u.count.physical_type: 'Counts',     # type: ignore
        u.electron.physical_type: 'Counts',  # type: ignore
        u.mag.physical_type: 'Magnitude',    # type: ignore
        physical.length: 'Distance',
        physical.power_density: 'Flux',
        physical.spectral_flux_density: 'Flux Density',
        physical.surface_brightness: 'Surface Brightness',
    }

    _SPECTRAL_TYPE_LABELS = {
        physical.length: 'Wavelength',
        physical.frequency: 'Frequency',
        physical.energy: 'Energy',
        physical.speed: 'Velocity',
    }

    # circles params
    circle_linewidth: float = 2
    circle_fill: bool = False
    ellipse_label_loc: tuple[float, float] = (0.03, 0.03)

    # Science Params
    # --------------
    # data params
    wavelength_unit = None
    radial_velocity: float | None = None

    # data cube params
    stack_cube_method: Literal['mean', 'median', 'sum', 'max', 'min', 'std'] = 'sum'

    # extract_cube_spectrum params
    spectra_rest_frequency: float | None = None
    flux_extract_method: Literal['mean', 'median', 'sum'] = 'mean'
    spectral_cube_extraction_mode = 'cube'
    spectrum_continuum_fit_method: str = 'fit_continuum'
    deredden_spectrum: bool = False
    plot_normalized_continuum: bool = False
    plot_continuum_fit: bool = False

    # plot_spectrum params
    plot_spectrum_text_loc: tuple[float, float] = (0.025, 0.95)

    # deredden spectra params
    Rv: float = 3.1 # Milky Way average
    Ebv: float = 0.19
    deredden_method: str | None = None
    deredden_region: str | None = None

    curve_fit: CurveFitConfig = field(default_factory=CurveFitConfig)

    # gaussian fitting params
    gaussian_model: Literal['gaussian', 'gaussian_line', 'gaussian_continuum'] = 'gaussian'
    return_gaussian_fit_parameters: bool = True
    print_gaussian_values: bool = True

    # reprojection parameters
    reproject_method: Literal['interp', 'exact'] = 'interp'
    return_footprint: bool = False
    reproject_block_size: Literal['auto'] | tuple | None = None
    reproject_parallel: bool = False

    # error propagation
    propagate_flux_error_method: Literal['mean', 'sum', 'median'] | None = None

    # Utils Params
    # ------------

    # pretty table params
    table_precision: int = 7
    table_sci_notation: bool = True
    table_column_pad: int = 3

    # numpy save
    save_format = '.npy'

    spectral_line_marker: SpectralLineConfig = field(default_factory=SpectralLineConfig)

    @property
    def elinewidth(self) -> float | None:
        return self.errorbar.linewidth

    @elinewidth.setter
    def elinewidth(self, value: float | None) -> None:
        self.errorbar.linewidth = value

    def reset(self):
        """
        Reset all configuration values to defaults.
        """
        default = type(self)()

        for f in fields(self):
            setattr(self, f.name, getattr(default, f.name))


config = VisualAstroConfig()


def _resolve_default(value: T | _Unset, fallback: T) -> T:
    """
    Fallback to a default configuration value if
    value is `_UNSET`.

    Parameters
    ----------
    value : T | _Unset
        Variable to be resolved.

    fallback : T
        Default value if `value` is `_UNSET`.
        Should be a `config` attribute.
        ie. `config.figsize`.

    Returns
    -------
    resolved_value : T
        Either `value` if value is not `_UNSET`
        or `fallback` if `value` is `_UNSET`.
    """
    if value is _UNSET:
        return fallback
    return value


def get_config_value(var, attribute):
    """
    Retrieve a configuration value, falling back to the
    default from `config` if `var` is None.

    Parameters
    ----------
    var : any
        User-specified value. If not None, this value is returned.
    attribute : str
        Name of the attribute to retrieve from `config` when `var` is None.

    Returns
    -------
    value : any
        The user-specified `var` if provided, otherwise the
        corresponding default value from `config`.
    """
    if var is None:
        return getattr(config, attribute)
    return var
