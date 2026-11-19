"""
Author: Elko Gerville-Reache
Date Created: 2025-05-23
Date Modified: 2026-11-15
Description:
    Spectra plotting functions.
"""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence
from typing import Literal

import astropy.units as u
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
import numpy as np
from astropy.units import Quantity
from matplotlib.colors import Colormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit

from visualastro.analysis.image_utils import stack_cube
from visualastro.analysis.spectra import (
    ExtractedPixelSpectra,
    GaussianFitResult,
    shift_to_rest_frame,
)
from visualastro.analysis.spectra import (
    gaussian as _gaussian,
)
from visualastro.analysis.spectra import (
    gaussian_continuum as _gaussian_continuum,
)
from visualastro.analysis.spectra import (
    gaussian_line as _gaussian_line,
)
from visualastro.analysis.spectra.manipulation import _masked_segments
from visualastro.analysis.spectra.utils import _get_spectral_axis
from visualastro.core.configuration import (
    _UNSET,
    _Unset,
    config,
)
from visualastro.core.data import get_value, to_array, to_list
from visualastro.core.io import savefig as savefigure
from visualastro.core.kwargs import (
    _extract_kwargs,
    _kwarg,
    _param,
    _resolve_kwargs,
)
from visualastro.core.numerical import interpolate as _interpolate
from visualastro.core.numerical import mask_within_range
from visualastro.core.sequences import _cycle, _unwrap_if_single
from visualastro.core.units import (
    convert_quantity,
    ensure_common_unit,
    get_unit,
)
from visualastro.datamodels.datacube import DataCube
from visualastro.optional_dependencies._spectralcube import SpectralCube
from visualastro.optional_dependencies._specutils import Spectrum
from visualastro.optional_dependencies._tqdm import tqdm
from visualastro.optional_dependencies.register import (
    _offer_dependency,
    _require_dependency,
)
from visualastro.plotting.base.plots import plot, step
from visualastro.plotting.core.axes import set_limits, subplots
from visualastro.plotting.core.axes.create import get_ax
from visualastro.plotting.core.colormaps import get_cmap
from visualastro.plotting.core.colors import get_colors, sample_cmap
from visualastro.plotting.core.legend import _expand_labels, legend
from visualastro.plotting.core.lines import axvlines
from visualastro.plotting.core.plot_input import (
    _resolve_color_alias,
    _set_prop_cycle_from_kwargs,
)
from visualastro.plotting.core.style import _style_context
from visualastro.plotting.interfaces.plot_utils import (
    _apply_plot_utils,
    _extract_plot_util_kwargs,
)

__all__ = [
    'extract_cube_spectrum',
    'extract_cube_pixel_spectra',
    'fit_gaussian_2_spec',
    'plot_spectrum',
    'plot_combine_spectrum',
    'plot_extracted_pixel_map',
]


_EXTRACT_METHODS: dict[str, str] = {
    'mean': 'mean',
    'median': 'median',
    'sum': 'sum',
}


def extract_cube_spectrum(
    cube: DataCube | SpectralCube | Sequence[DataCube | SpectralCube],
    flux_extract_method: Literal['mean', 'median', 'sum'] | _Unset = _UNSET,
    extract_mode: Literal['cube', 'slice', 'ray'] | _Unset = _UNSET,
    radial_vel: float | u.Quantity | None | _Unset = _UNSET,
    rest_freq: float | u.Quantity | None | _Unset = _UNSET,
    unit: str | u.Unit | None | _Unset = _UNSET,
    convention: str | None = None,
    **kwargs
) -> Spectrum | list[Spectrum]:
    """
    Extract 1D spectra from one or more data cubes by collapsing both spatial axes.

    Each cube is assumed to have axes ordered `(spectral, y, x)`. The spatial
    axes (1, 2) are collapsed with the chosen statistic.

    Parameters
    ----------
    cube : DataCube | SpectralCube
        Input SpectralCubes(s).
    flux_extract_method : {'mean', 'median', 'sum'} | None | _Unset, optional, default=_UNSET
        Statistic used to collapse the spatial axes.
        If None or `_UNSET`, uses `config.flux_extract_method`.
    extract_mode : {'cube', 'slice', 'ray'} | None | _Unset, optional, default=_UNSET
        Traversal strategy for the collapse. Trades memory for speed.

        - 'cube' : Load the entire cube in memory. Fastest, but blocked for
          very large cubes unless `cube.allow_huge_operations = True`.
        - 'slice' : Process slice-by-slice along the spectral axis. Recommended
          for moderately large cubes.
        - 'ray' : Process spaxel-by-spaxel. Lowest memory, slowest. Use for
          very large cubes or low-memory environments.

        If None or `_UNSET`, uses `config.spectral_cube_extraction_mode`.
    radial_vel : float | astropy.units.Quantity | None | _Unset, optional, default=_UNSET
        Radial velocity used to shift the spectral axis to the rest frame
        (`shift_to_rest_frame`). Floats are interpreted as km/s. If None,
        no shift is applied. If `_UNSET`, uses `config.radial_velocity`.
    rest_freq : float | astropy.units.Quantity | None | _Unset, optional, default=_UNSET
        Rest-frame frequency or wavelength stored on each `Spectrum` as
        `rest_value`. If None, no rest value is set.
        If `_UNSET`, uses `config.spectra_rest_frequency`.
    unit : str | astropy.units.Unit | None | _Unset, optional, default=_UNSET
        Target unit of the spectral axis. Conversion uses
        `astropy.units.spectral()` equivalencies. If None,
        no conversion is performed. If `_UNSET`, uses `config.wavelength_unit`.
    convention : str | None, optional, default=None
        Doppler convention ('optical', 'radio', or 'relativistic'), stored on
        each `Spectrum` as `velocity_convention`. If None, the specutils
        default is used.

    Returns
    -------
    Spectrum
        Collapsed 1D spectrum. The spectral axis is shifted by `radial_vel`
        and converted to `unit`. The flux carries the unit of `cube`.

    Raises
    ------
    ValueError
        If `flux_extract_method` is not one of 'mean', 'median', 'sum'.
    ImportError
        If `spectral-cube` or `specutils` is not installed.
    astropy.units.UnitConversionError
        If the spectral axis cannot be converted to `unit`.
    """
    _require_dependency('spectral-cube', 'specutils')
    params = _extract_kwargs(
        kwargs,
        params=[
            _param('flux_extract_method', flux_extract_method, config.flux_extract_method),
            _param('extract_mode', extract_mode, config.spectral_cube_extraction_mode),
            _param('radial_vel', radial_vel, config.radial_velocity),
            _param('rest_freq', rest_freq, config.spectra_rest_frequency),
            _param('unit', unit, config.wavelength_unit),
            _param('convention', convention, None),
        ],
    )

    method_name = _EXTRACT_METHODS.get(params['flux_extract_method'])
    if method_name is None:
        raise ValueError(
            f"Invalid flux_extract_method '{params['flux_extract_method']}'. "
            f'Choose from {list(_EXTRACT_METHODS)}.'
        )

    single = isinstance(cube, DataCube | SpectralCube)
    cubes = [cube] if single else list(cube)

    results = [_extract_single_cube_spectrum(c, **params) for c in cubes]

    return results[0] if single else results


def _extract_single_cube_spectrum(
    cube: DataCube | SpectralCube,
    flux_extract_method: Literal['mean', 'median', 'sum'],
    extract_mode: Literal['cube', 'slice', 'ray'],
    radial_vel: float | u.Quantity | None,
    rest_freq: float | u.Quantity | None,
    unit: str | u.Unit | None,
    convention: str | None
) -> Spectrum:

    if not isinstance(cube, (DataCube, SpectralCube)):
        raise TypeError(
            'Input `cube` must be a SpectralCube! '
            f'got: {type(cube).__name__}'
        )

    spectral_axis = shift_to_rest_frame(cube.spectral_axis, radial_vel)
    spectral_axis = convert_quantity(spectral_axis, unit, equivalencies=u.spectral())

    flux = u.Quantity(getattr(cube, flux_extract_method)(axis=(1, 2), how=extract_mode))

    return Spectrum(
        spectral_axis=spectral_axis,
        flux=flux,
        rest_value=rest_freq,
        velocity_convention=convention,
    )


def extract_cube_pixel_spectra(
    cube: DataCube | SpectralCube | Quantity | NDArray,
    *,
    idx: int | Sequence[int] | None = None,
    idx_range: tuple[int, int] | None = None,
    idx_drop: int | Sequence[int] | None = None,
    combine_spectra: bool = False,
    combine_method: Literal['sum', 'mean', 'median'] | _Unset = _UNSET,
    plot_spatial_map: bool = True,
    vline: float | Quantity | None = None,
    cmap: str | Colormap | _Unset = _UNSET,
    **kwargs,
) -> ExtractedPixelSpectra:
    """
    Extract per-pixel spectra from a spectral cube, keeping only spatial
    pixels that contain at least one non-NaN value along the spectral axis.

    Optionally compute and plot a combined spectrum derived from the
    extracted pixel spectra.

    NOTE : To ensure that the mapping between pixels and indices is correct,
    set `map_idx` to the correct cube slice!

    Parameters
    ----------
    cube : DataCube | SpectralCube | Quantity | ArrayLike
        Spectral cube with shape (T, N, M), where T is the spectral axis
        and (N,M) the spatial dimensions. If `cube` has no units, it is
        assined `u.dimensionless_unscaled`.
    idx : int | Sequence[int] | None None, optional, default=None
        Index or indices of the extracted per-pixel spectra to *select*.
        This controls which spatial pixels are extracted, combined,
        and plotted. If None, all valid spatial pixels are extracted.
        If an int, only that extracted spectrum is kept. If a sequence
        of ints (e.g., list, tuple, or NumPy array), only those extracted
        spectra are kept. The indices correspond to the ordering shown
        in the legend, where each entry is labeled as:
            <index>: (x=<x>, y=<y>)
    idx_range : tuple[int, int] | None, optional, default=None
        Inclusive range of indices of the extracted per-pixel spectra to *select*.
        This controls which spatial pixels are extracted, combined, and plotted.
        Equivalent to:
            `idx = np.arange(start, stop + 1)`
        Overrides `idx` if provided.
    idx_drop : int | Sequence[int] | None, optional, default=None
        Index or indices of extracted per-pixel spectra to remove *after*
        applying `idx` or `idx_range`. This allows excluding specific spectra
        (e.g., bad pixels) without redefining the full selection.
    combine_spectra : bool, optional, default=False
        If `True`, compute and plot a combined spectrum from all extracted
        pixel spectra using `combine_method`.
    combine_method : {'sum', 'mean', 'median'} | _Unset, optional, default=_UNSET
        Method used to combine per-pixel spectra when `plot_combined=True`.
        If `_UNSET`, uses `config.flux_extract_method`.
    plot_spatial_map : bool, optional, default=True
        If `True`, plot a 2D spatial map showing the locations of the
        extracted pixels, color-coded to match the spectral plot.
    vline : Quantity | float | None, optional
        If provided, draw a vertical dotted reference line at this wavelength.
        If unitless, the value is assumed to be in the same units as the
        spectral axis.
    cmap : str | Colormap | _Unset, optional, default=_UNSET
        Colormap used to sample per-spectrum colors.
        If `_UNSET`, uses `config.cmap`.
    style : str, optional, default=config.style
        Matplotlib style to use for plotting.
    background_cube : DataCube | SpectralCube | Quantity | ArrayLike | None, optional, default=None
        Background cube to be plotted for the spatial map. Overrides
        `cube`, and should have the same shape as `cube`. This is
        an experimental feature, and does not guarantee perfect alignment.
    map_idx : int | tuple[int, int] | None, optional, default=None
        Index or indices specifying which cube slice to plot
        for the spatial map plot:

        - `i` or `[i]` -> returns `cube[i]`
        - `[i, j]` -> returns `cube[i:j+1].sum(axis=0)`

        If `None`, uses `map_idx=0`.
    legend : bool, optional, default=True
        If `True`, displays the legend below the plot.
    figsize : tuple, optional, default=(12,6)
        Plot figsize.
    fontsize : float, optional, default=8
        Fontsize of the spatial map index legend in points.
    ncols : int, optional, default=8
        Number of columns for the legend.
    savefig : bool, optional, default=False
        If `True`, saves figure to disk.
    Any additional keyword arguments are forwarded to `plot_extracted_pixel_map`.

    Returns
    -------
    result : ExtractedPixelSpectra
        Container object holding the results of the extraction. It exposes
        the following attributes:

        - `spectra` : SpectrumPlus | list[SpectrumPlus]
          Extracted per-pixel spectra. If a single index is selected,
          this is returned as a single `SpectrumPlus` instance;
          otherwise, a list is returned.
        - `cube_array` : NDArray
          Copy of the original cube, with all pixels set to NaN
          other than the pixels corresponding to `extract_idx`.
        - `extract_idx` : ndarray of int, shape (N,)
          Indices of the extracted spectra corresponding to the ordering
          of valid spatial pixels.
        - `coords` : ndarray of int, shape (N, 2)
          Spatial coordinates of extracted pixels in `(y, x)` order.
        - `colors` : list
          Colors assigned to each extracted pixel/spectrum.
        - `labels` : list[str]
          Human-readable labels of the form `<idx>: (x=<x>, y=<y>)`.
        - `combined_spectrum` : SpectrumPlus | None
          Combined spectrum computed using `combine_method` if
          `combine_spectra=True`; otherwise None.
    """
    _require_dependency('spectral-cube', 'specutils')
    _offer_dependency('tqdm')
    params = _resolve_kwargs(
        kwargs,
        params=[
            _param('combine_method', combine_method, config.flux_extract_method),
            _param('cmap', cmap, config.cmap),
        ],
        additional_kwargs=[
            _kwarg('background_cube', None),
            _kwarg('map_idx', None),
            _kwarg('legend', True),
            _kwarg('figsize', (12,6)),
            _kwarg('style', config.style),
            _kwarg('ncols', 8),
            _kwarg('savefig', False),
        ]
    )

    cmap = get_cmap(params.cmap)

    if combine_spectra and params.combine_method not in {'sum', 'mean', 'median'}:
        raise ValueError(
            "`combine_method` must be one of {'sum', 'mean', 'median'} when "
            "`combine_spectra=True`."
        )

    spectral_axis = _get_spectral_axis(cube)
    if spectral_axis is None:
        raise ValueError('Could not determine spectral_axis from cube')

    data_unit = get_unit(cube) or u.dimensionless_unscaled

    data = to_array(cube, keep_unit=False)
    if data.ndim != 3:
        raise ValueError('cube must have shape (T, N, M)')

    error_cube = getattr(cube, 'error', None)

    if idx_range is not None:
        if (not isinstance(idx_range, (list, tuple, np.ndarray))
            or len(idx_range) != 2):
            raise TypeError(
                'range must be a sequence of two integers (start, stop)'
            )

        start, stop = map(int, idx_range)
        if stop < start:
            raise ValueError('range[1] must be >= range[0]')

        idx_set = set(range(start, stop + 1))

    elif idx is None:
        idx_set = None
    elif isinstance(idx, (int, np.integer)):
        idx_set = {int(idx)}
    elif isinstance(idx, (list, tuple, np.ndarray)):
        idx_set = {int(i) for i in idx}
    else:
        raise TypeError(
            'idx must be None, an int, or a sequence of ints'
        )

    # spatial pixels (spectrums) that are not entirely NaN
    valid_mask = ~np.all(np.isnan(data), axis=0)
    coords = np.column_stack(np.where(valid_mask))

    if idx_drop is not None:
        if isinstance(idx_drop, (int, np.integer)):
            drop_set = {int(idx_drop)}
        elif isinstance(idx_drop, (list, tuple, np.ndarray)):
            drop_set = {int(i) for i in idx_drop}
        else:
            raise TypeError(
                'idx_drop must be None, an int, or a sequence of ints'
            )

        if idx_set is None:
            idx_set = set(range(len(coords))) - drop_set
        else:
            idx_set = idx_set - drop_set

    if idx_set is None:
        extract_idx = np.arange(len(coords))
    else:
        extract_idx = np.array(
            [i for i in sorted(idx_set) if i < len(coords)],
            dtype=int,
        )
    if extract_idx.size == 0:
        raise ValueError('idx does not select any valid spectra')

    coords = coords[extract_idx]
    ys = coords[:, 0]
    xs = coords[:, 1]

    # the transpose is necessary to ensure that flux_matrix
    # has shape Npixels x T, so that looping over flux_matrix
    # gives you each spectrum along a column of voxels
    flux_matrix = data[:, ys, xs].T * data_unit
    spectra = [
        Spectrum(spectral_axis=spectral_axis, flux=flux)
        for flux in flux_matrix
    ]

    masked_data = np.full_like(data, np.nan)
    masked_data[:, ys, xs] = data[:, ys, xs]
    if (
        get_unit(masked_data) is None and
        data_unit != u.dimensionless_unscaled
    ):
        masked_data = masked_data * data_unit

    if error_cube is not None:
        err_data = to_array(error_cube, keep_unit=False)
        err_unit = get_unit(error_cube) or u.dimensionless_unscaled

        if err_data.shape != data.shape:
            raise ValueError('error cube must have same shape as cube')

        masked_err = np.full_like(err_data, np.nan)
        masked_err[:, ys, xs] = err_data[:, ys, xs]

        if (
            get_unit(masked_err) is None and
            err_unit != u.dimensionless_unscaled
        ):
            masked_err = masked_err * err_unit
    else:
        masked_err = None

    labels = [
        f'{i}: (x={x}, y={y})'
        for i, (y, x) in zip(extract_idx, coords)
    ]

    n_plot = len(spectra)
    combined_spec = None

    if combine_spectra:
        if params.combine_method == 'sum':
            combined_flux = np.nansum(flux_matrix, axis=0)
        elif params.combine_method == 'mean':
            combined_flux = np.nanmean(flux_matrix, axis=0)
        elif params.combine_method == 'median':
            combined_flux = np.nanmedian(flux_matrix, axis=0)
        else:
            raise ValueError(f'Unknown combine_method: {params.combine_method}')

        combined_spec = Spectrum(spectral_axis=spectral_axis, flux=combined_flux)

    with _style_context(params.style):
        fig, ax = plt.subplots(figsize=params.figsize)
        ax.set_autoscale_on(False)

        if isinstance(cmap, (list, tuple, np.ndarray)) and len(cmap) >= n_plot:
            colors = list(cmap[:n_plot])
        else:
            colors = sample_cmap(n_plot, cmap)

        for i in tqdm(range(len(spectra)), desc='plotting'):
            plot_spectrum(
                spectra[i],
                ax,
                color=[colors[i]],
                label=labels[i],
            )

        fluxes = [spec.flux for spec in spectra]

        if combine_spectra and isinstance(combined_spec, Spectrum):
            plot_spectrum(
                combined_spec,
                ax,
                color='k',
                ls='--',
                label=f'combined ({params.combine_method})',
            )
            fluxes.append(combined_spec.flux)

        axvlines(vline, ax=ax, unit=fluxes[0].unit)

        set_limits(
            ax=ax,
            xpad=0,
            **kwargs
        )

        if params.legend:
            legend(
                ncols=params.ncols,
                fontsize=8,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                frameon=False,
            )
        if params.savefig:
            savefigure(**kwargs)

        plt.show()

    result = ExtractedPixelSpectra(
        spectra=_unwrap_if_single(spectra),
        cube_array=masked_data,
        error_array=masked_err,
        extract_idx=extract_idx,
        coords=coords,
        colors=colors,
        labels=labels,
        combined_spectrum=combined_spec,
    )

    if plot_spatial_map:
        plot_extracted_pixel_map(
            result,
            cube=(
                data if params.background_cube is None
                else params.background_cube
            ),
            savefig=params.savefig,
            idx=params.map_idx,
            **kwargs
        )

    return result


def plot_extracted_pixel_map(
    extracted_pixel_spectra: ExtractedPixelSpectra,
    *,
    cube: DataCube | NDArray | Quantity | SpectralCube | None = None,
    **kwargs
):
    """
    Plot a 2D spatial map of extracted pixel locations from a spectral cube.

    This function visualizes which spatial pixels were selected during a
    per-pixel spectral extraction step. The spatial locations are overlaid
    on a 2D slice of the cube and color-coded to match the corresponding
    spectra in the spectral plot.

    NOTE : The spatial map may not be accurate unless `idx` is set properly!
    By default `idx=0`. Please set `idx` for optimal results.

    Parameters
    ----------
    extracted_pixel_spectra : ExtractedPixelSpectra
        Object containing the results of a pixel-spectra extraction.
        Must expose the following attributes:

        - `cube_array` : 3D NDArray
          NumPy 3D array to plot spatial map onto.
        - `extract_idx` : ArrayLike of int
          Indices of the extracted pixel spectra.
        - `coords` : ArrayLike, shape `(N, 2)`
          Spatial pixel coordinates in `(y, x)` order.
        - `colors` : sequence
          Colors assigned to each extracted pixel, matching the spectral plot.

    cube : DataCube | SpectralCube | Quantity | np.ndarray, optional, default=None
        Spectral cube with shape `(T, N, M)` or a 2D spatial slice
        with shape `(N, M)`. If 3D, either the first spectral slice
        is shown or a slice specified by `idx`. Overrides `cube_array`
        in `extracted_pixel_spectra`. This is not entirely supported and
        may cause misalignments since the spatial mapping was computed on
        `extracted_pixel_spectra.cube_array`.
    figsize : tuple, optional, default=(12, 6)
        Size of the figure in inches.
    style : str, optional
        Matplotlib style name.
    annotate : bool, optional, default=True
        If `True`, annotate each extracted pixel with its index.
    idx : int | list[int] | None, optional, default=None
        Index or indices specifying the slice along the first axis
        for the spatial map plot:

            - `i` -> returns `cube[i]`
            - `[i]` -> returns `cube[i]`
            - `[i, j]` -> returns `cube[i:j+1].sum(axis=0)`

        If `None`, uses `idx=0`.
    savefig : bool, optional, default=False
        If `True`, save the figure to disk using `savefig`.
    alpha : float, optional, default=0.8
        Alpha value for individual pixel colors.
    fontsize : float, optional, default=8
        Fontsize of annotations in points.
    Any additional keyword arguments are forwarded to `imshow`.

    Raises
    ------
    ValueError
        If `coords` does not have shape `(N, 2)`, if the number of
        coordinates and colors differ, or if `cube` has an invalid shape.
    IndexError
        If any extracted pixel coordinates fall outside the spatial
        dimensions of the cube.

    Notes
    -----
    - Spatial coordinates are interpreted in `(y, x)` order.
    - Extracted pixels are rendered as an RGBA overlay for efficient,
      vectorized plotting.
    - This function is intended to be used in conjunction with
      `extract_cube_pixel_spectra` and its returned extraction object.
    - For best results, set `idx` to the correct cube slice to preserve
      the correct mapping between pixels and indices.
    """
    params = _resolve_kwargs(
        kwargs,
        additional_kwargs=[
            _kwarg('annotate', True),
            _kwarg('idx', None),
            _kwarg('figsize', (12, 6)),
            _kwarg('style', config.style),
            _kwarg('alpha', 0.8),
            _kwarg('fontsize', 8),
            _kwarg('savefig', False),
        ]
    )

    extract_idx = extracted_pixel_spectra.extract_idx
    coords = extracted_pixel_spectra.coords
    colors = extracted_pixel_spectra.colors
    cube = extracted_pixel_spectra.cube_array if cube is None else cube

    extract_idx = np.asarray(extract_idx, dtype=int)
    coords = np.asarray(coords)
    colors = list(colors)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError('coords must have shape (N, 2) as (y, x)')

    if len(coords) != len(colors):
        raise ValueError('coords and colors must have the same length')

    background = to_array(cube, keep_unit=False)
    if background.ndim == 3:
        if params.idx is not None:
            background = stack_cube(
                background, idx=params.idx, method='sum', axis=0
            )
        else:
            background = background[0]
    elif background.ndim != 2:
        raise ValueError('cube must be a 3D!')

    ny, nx = background.shape
    ys, xs = coords[:, 0], coords[:, 1]

    if np.any(ys < 0) or np.any(ys >= ny) or np.any(xs < 0) or np.any(xs >= nx):
        raise IndexError('Some coords fall outside cube spatial dimensions')

    colors_rgba = np.array([to_rgba(c, alpha=params.alpha) for c in colors])
    overlay = np.zeros((ny, nx, 4), dtype=float)
    overlay[ys, xs] = colors_rgba

    with _style_context(params.style):
        fig, ax = plt.subplots(figsize=params.figsize)
        ax.imshow(
            background,
            origin='lower',
            cmap='gray',
        )

        ax.imshow(overlay, origin='lower')

        if params.annotate:
            for idx, y, x in zip(extract_idx, ys, xs):
                ax.text(
                    x, y, str(idx),
                    ha='center',
                    va='center',
                    fontsize=params.fontsize,
                    color='black',
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor='white',
                        edgecolor='none',
                        alpha=0.7
                    )
                )

        ax.set_xlabel('X pixel')
        ax.set_ylabel('Y pixel')
        ax.set_title(f'Extracted Pixel Locations (n={len(extract_idx)})')

        if params.savefig:
            savefigure(**kwargs)

        plt.show()


def plot_spectrum(
    spectrum: Spectrum,
    ax: maxes.Axes | None = None,
    plot_mode: Literal['step', 'plot'] = 'step',
    **kwargs,
) -> list[Line2D]:
    """
    Plot all spectra in a :class:`~specutils.Spectrum` on a
    Matplotlib :class:`~matplotlib.axes.Axes`.

    Is a thin wrapper of :func:`~visualastro.step` or :func:`~visualastro.plot`.
    Plots ``spectrum.spectral_axis`` and ``spectrum.flux``.
    Also sets ``xpad`` to 0 by default.

    Parameters
    ----------
    spectrum : specutils.Spectrum
        Spectrum to plot. ``spectrum.flux`` must be 1D (one line) or 2D with
        shape (n_spectra, n_spectral) (one line per row). All rows share
        the same spectral axis.
    ax : matplotlib.axes.Axes | None, optional, default=None
        Axes to plot on. If ``None``, resolved via ``get_ax``.
    plot_mode : {'step', 'plot'}, optional, default='step'
        Plotting backend, case-insensitive.

        - ``'step'``: calls :func:`~visualastro.step`.
        - ``'plot'``: calls :func:`~visualastro.plot`.

    **kwargs
        Additional keyword arguments forwarded to the selected backend
        (``visualastro.step`` or ``visualastro.plot``).

    Returns
    -------
    list[matplotlib.lines.Line2D]
        Plotted line objects, one per spectrum.

    Raises
    ------
    TypeError
        If `spectrum` is not a :class:`specutils.Spectrum`.
        If `ax` is not a Matplotlib Axes (raised by `get_ax`).
    ValueError
        If `spectrum.flux` has more than 2 dimensions.
        If `plot_mode` is not 'step' or 'plot'.
    """
    _require_dependency('specutils')

    if not isinstance(spectrum, Spectrum):
        raise TypeError(
            f'`spectrum` must be a `Spectrum`, got {type(spectrum).__name__}.'
        )
    if spectrum.flux.ndim > 2:
        raise ValueError(
            f'`spectrum.flux` must be 1D or 2D, got {spectrum.flux.ndim}D.'
        )

    plot_fn = {'step': step, 'plot': plot}.get(str(plot_mode).lower())
    if plot_fn is None:
        raise ValueError(
            f"`plot_mode` must be 'step' or 'plot', got {plot_mode!r}."
        )

    xpad = kwargs.pop('xpad', 0)
    return plot_fn(
        spectrum.spectral_axis, spectrum.flux, ax=ax, xpad=xpad, **kwargs
    )


def plot_combine_spectrum(
    spectra: Spectrum | Sequence[Spectrum],
    ax: maxes.Axes | None = None,
    spectral_edges: Sequence[float] | None = None,
    concatenate: bool = False,
    plot_normalize: bool = False,
    color: ColorType | Sequence[ColorType] | int | None = None,
    single_color: bool = True,
    label: str | Sequence[str] | None = None,
    plot_mode: Literal['step', 'plot'] = 'step',
    **kwargs,
) -> list[Line2D]:
    """
    Plot one or more spectra on `ax`, optionally stitched into one curve.

    Parameters
    ----------
    spectra : Spectrum | Sequence[Spectrum]
        Spectra to plot, in spectral order.
    ax : matplotlib.axes.Axes | None, optional, default=None
        Axes on which to plot. If None, uses the current axes
        (see `get_ax`).
    spectral_edges : Sequence[float | Quantity] | None, optional, default=None
        Edges ``[e0, ..., eN]`` along the spectral axis delimiting N
        spectra; see :func:`~visualastro.splice_spectra`. Quantities are
        converted to the spectral axis unit. Floats are assumed to be in that unit.
    concatenate : bool, optional, default=False
        If True, plot all spectra as one continuous curve
        (single artist, single color).
    plot_normalize : bool, optional, default=False
        If True, plot `spectrum.normalize` instead of `spectrum.flux`.
    color : ColorType | Sequence[ColorType] | int | None, optional, default=None
        Color specification passed to `get_colors`. If None, uses the
        axes property cycle.
    single_color : bool, optional, default=False
        If True, draw every segment with the color of the first segment.
        Redundant with `concatenate=True`.
    label : str | Sequence[str] | None, optional, default=None
        Legend label(s). A string labels the first segment only. A
        sequence must contain exactly one label per segment.
    plot_mode : {'step', 'plot'}, optional, default='step'
        Plotting backend, case-insensitive.

        - ``'step'``: calls :func:`~visualastro.step`.
        - ``'plot'``: calls :func:`~visualastro.plot`.

    **kwargs
        Additional keyword arguments. [fill in from the original docstring]

    Returns
    -------
    list of matplotlib.lines.Line2D
        One artist per plotted segment.

    Raises
    ------
    ValueError
        If `plot_mode` is invalid, or if `label` is a sequence whose
        length differs from the number of segments.
    """
    _require_dependency('specutils')

    params = _extract_kwargs(
        kwargs,
        params=[
            _param('color', color, None),
            _param('label', label, None),
        ],
        additional_kwargs=[
            _kwarg('linewidth', None),
            _kwarg('linestyle', None),
            _kwarg('alpha', None),
            _kwarg('rasterized', config.rasterized),
        ]
    )
    plot_params = _extract_plot_util_kwargs(kwargs)
    kwargs.update(params)

    spectra_list = to_list(spectra)
    spectral_axes = [spec.spectral_axis for spec in spectra_list]
    if spectral_edges is not None:
        spectral_axes.append(spectral_edges)

    ensure_common_unit(spectral_axes)
    ensure_common_unit([spec.flux for spec in spectra_list])

    segments = _masked_segments(spectra_list, spectral_edges, plot_normalize)
    if concatenate:
        segments = [(
            np.concatenate([w for w, _ in segments]),
            np.concatenate([f for _, f in segments]),
        )]

    if (color := kwargs['color']) is not None:
        kwargs['color'] = get_colors(color)

    ax = get_ax(ax)
    _set_prop_cycle_from_kwargs(ax, kwargs)

    plot_fn = {'step': step, 'plot': plot}.get(str(plot_mode).lower())
    if plot_fn is None:
        raise ValueError(
            f"`plot_mode` must be 'step' or 'plot', got {plot_mode!r}."
        )

    label = kwargs.pop('label')
    labels = _expand_labels(label, len(segments))
    artists: list[Line2D] = []

    for i, ((wave, flux), lab) in enumerate(zip(segments, labels, strict=True)):
        lines = plot_fn(get_value(wave), get_value(flux), ax=ax, label=lab, **kwargs)
        artists.extend(lines)
        if single_color and i == 0:
            kwargs['color'] = lines[0].get_color()

    plot_params.limits['xpad'] = kwargs.pop('xpad', 0)
    _apply_plot_utils(
        plot_params, ax,
        x=segments[0][0], y=segments[0][1],
        img=artists[0],
        label=labels[0],
        **kwargs
    )

    return artists


# Spectra Fitting Functions
# -------------------------
def fit_gaussian_2_spec(
    spectrum: Spectrum,
    p0: ArrayLike,
    *,
    model: Literal['gaussian', 'gaussian_line', 'gaussian_continuum'] | _Unset = _UNSET,
    spectral_range: tuple[float, float] | None = None,
    fit_method: Literal['lm', 'trf', 'dogbox'] | _Unset = _UNSET,
    absolute_sigma: bool | _Unset = _UNSET,
    yerror: Quantity | ArrayLike | None = None,
    interpolate: bool | _Unset = _UNSET,
    samples: int | _Unset = _UNSET,
    interp_method: Literal['cubic', 'cubic_spline', 'linear'] | _Unset = _UNSET,
    error_interp_method: Literal['cubic', 'cubic_spline', 'linear'] | _Unset = _UNSET,
    return_fit_params: bool | _Unset = _UNSET,
    plot_interp: bool = False,
    print_vals: bool | _Unset = _UNSET,
    **kwargs
) -> GaussianFitResult | tuple:
    """
    Fit a Gaussian-like model to a Spectrum, optionally including the continuum.

    Parameters
    ----------
    spectrum : Spectrum
        :class:`~specutils.Spectrum` object to be gaussian fitted.
    p0 : ArrayLike
        Initial guess for the Gaussian fit parameters.
        This should match the input arguments of the
        gaussian model (excluding the first argument
        which is the spectral axis). For `model='gaussian_continuum'`,
        the last element must be the continuum array, with the
        same shape as `spectrum.flux`. It is removed from `p0`
        before fitting.
    model : {'gaussian', 'gaussian_line', 'gaussian_continuum'} | _Unset, optional, default=_UNSET
        Type of Gaussian model to fit:

        - 'gaussian' : standard Gaussian
        - 'gaussian_line' : Gaussian with linear continuum
        - 'gaussian_continuum' : Gaussian with computed continuum array

        The continuum can be computed with `compute_continuum()`.
        If ``_UNSET``, uses `config.gaussian_model`.
    spectral_range : tuple[float, float] | None, optional, default=None
        (min, max) spectral axis range to restrict the fit, in the units
        of `spectrum.spectral_axis`. If ``None``, computes the min and max
        from the spectral axis.
    fit_method : {'lm', 'trf', 'dogbox'} | _Unset, optional, default=_UNSET
        Curve fitting algorithm used by `scipy.optimize.curve_fit`.
        If ``_UNSET``, uses ``config.curve_fit.method``.
    absolute_sigma : bool | _Unset, optional, default=_UNSET
        If True, the values provided in `yerror` are interpreted as absolute
        1σ uncertainties on the flux measurements. In this case, the returned
        covariance matrix reflects these absolute uncertainties, and parameter
        errors are reported in physical units.
        If False, the values in `yerror` are treated as relative weights only.
        The covariance matrix is rescaled such that the reduced χ² of the fit
        is unity, and the reported parameter uncertainties reflect relative
        errors rather than absolute measurement uncertainties.
        Set this to True when `yerror` represents well-calibrated observational
        uncertainties (e.g., photon-counting or pipeline-provided errors).
        Set this to False when `yerror` is used only for weighting the fit.
        If `_UNSET`, uses `config.curve_fit.absolute_sigma`.
    yerror : Quantity | ArrayLike | None, optional, default=None
        Flux uncertainties to be used in the fit, in the units of
        `spectrum.flux`. If None, uncertainties are ignored when
        computing the fit. This is passed to `curve_fit` as the
        `sigma` parameter.
    interpolate : bool | _Unset, optional, default=_UNSET
        Whether to interpolate the spectrum over a regular spectral axis grid.
        The number of samples is controlled by `samples`. If `_UNSET`, uses `config.curve_fit.interpolate`.
    samples : int | _Unset, optional, default=_UNSET
        Number of points in interpolated spectral axis grid. If
        `_UNSET`, uses `config.curve_fit.samples`.
    interp_method : {'cubic', 'cubic_spline', 'linear'} | _Unset, optional, default=_UNSET
        Interpolation method used. If `_UNSET`, uses
        `config.curve_fit.interpolation_method`.
    error_interp_method : {'cubic', 'cubic_spline', 'linear'} | _Unset, optional, default=_UNSET
        Method to interpolate yerror if provided. If `_UNSET`, uses
        `config.curve_fit.error_interpolation_method`.
    return_fit_params : bool | _Unset, optional, default=_UNSET
        If True, return full computed best-fit parameters for all parameters,
        including popt, pcov, and perr. If False, return only Flux, FWHM, and mu.
        If `_UNSET`, uses `config.return_gaussian_fit_parameters`.
    plot_interp : bool, optional, default=False
        If True, plot the interpolated spectrum. This is
        provided for debugging purposes.
    print_vals : bool | _Unset, optional, default=_UNSET
        If ``True``, print a table of best-fit parameters,
        errors, and computed quantities. If `_UNSET`,
        uses `config.print_gaussian_values`.

    **kwargs : dict, optional

        - `xlim` : tuple, optional, default=None
          Wavelength range for plotting. If None, uses `spectral_range`.
        - `savefig` : bool, optional, default=False
          If True, save current figure to disk.
        - Any parameter passed on to :func:`~visualastro.plot`

    Returns
    -------
    GaussianFitResult | GaussianHandles

        GaussianFitResult : dataclass (when `return_fit_params=True`):
            Contains all fitted parameters (amplitude, mu, sigma, and optionally
            slope/intercept for line models), derived quantities (flux, FWHM),
            and their 1σ uncertainties. Also includes raw fit outputs (popt, pcov, perr).
            See dataclass definition for complete attribute list.

        GaussianHandles : namedtuple (when `return_fit_params=False`):
            Simplified namedtuple with key results:

            - flux, FWHM, mu : Quantity
                Integrated flux, full width at half maximum, and center position.
            - flux_error, FWHM_error, mu_error : Quantity
                1σ uncertainties on the above quantities.
    """
    _require_dependency('specutils')

    params = _resolve_kwargs(
        kwargs,
        params=[
            _param('model', model, config.gaussian_model),
            _param('fit_method', fit_method, config.curve_fit.method),
            _param('absolute_sigma', absolute_sigma, config.curve_fit.absolute_sigma),
            _param('interpolate', interpolate, config.curve_fit.interpolate),
            _param('samples', samples, config.curve_fit.samples),
            _param('interp_method', interp_method, config.curve_fit.interpolation_method),
            _param('error_interp_method', error_interp_method, config.curve_fit.error_interpolation_method),
            _param('return_fit_params', return_fit_params, config.return_gaussian_fit_parameters),
            _param('print_vals', print_vals, config.print_gaussian_values),
        ],
        additional_kwargs=[
            _kwarg('color', _UNSET),
            _kwarg('savefig', config.savefig.enable),
        ]
    )

    spectral_axis = spectrum.spectral_axis
    flux = spectrum.flux
    spectral_unit = spectral_axis.unit
    flux_unit = flux.unit

    x0 = spectral_axis.to_value()
    y0 = flux.to_value()

    if yerror is not None:
        yerror = yerror.to_value()

    p0 = list(p0)

    # compute default wavelength range from wavelength
    if spectral_range is None:
        spectral_range = (np.nanmin(x0), np.nanmax(x0))

    if params.model == 'gaussian_continuum':
        # remove continuum values to ensure it is not
        # included as a free parameter during minimization
        continuum = np.asarray(p0.pop(-1))
    else:
        continuum = None

    # clip values outside wavelength range
    spectral_mask = mask_within_range(x0, spectral_range)

    x0 = x0[spectral_mask]
    y0 = y0[spectral_mask]
    if yerror is not None:
        yerror = yerror[spectral_mask]
    if params.model == 'gaussian_continuum' and continuum is not None:
        continuum = continuum[spectral_mask]

    finite_mask = np.isfinite(x0) & np.isfinite(y0)

    x0 = x0[finite_mask]
    y0 = y0[finite_mask]

    if yerror is not None:
        yerror = yerror[finite_mask]

    if params.model == 'gaussian_continuum' and continuum is not None:
        continuum = continuum[finite_mask]

    if len(x0) == 0:
        raise ValueError(
            f'No data points within spectral_range {spectral_range}'
        )

    # interpolate arrays
    if params.interpolate:
        # interpolate wavelength and flux arrays
        x, y = _interpolate(
            x0, y0,
            x_range=spectral_range,
            N_samples=params.samples,
            method=params.interp_method,
        )
        # interpolate y error values
        if yerror is not None:
            _, yerror = _interpolate(
                x0, yerror,
                x_range=spectral_range,
                N_samples=params.samples,
                method=params.error_interp_method
            )
        # interpolate continuum array
        if params.model == 'gaussian_continuum' and continuum is not None:
            _, continuum = _interpolate(
                x0, continuum,
                x_range=spectral_range,
                N_samples=params.samples,
                method=params.interp_method,
            )
    else:
        x, y = x0, y0

    if params.model == 'gaussian_continuum' and continuum is not None:
        def function(x, A, mu, sigma):
            return _gaussian_continuum(x, A, mu, sigma, continuum)

    elif params.model == 'gaussian_line':
        function = _gaussian_line

    else:
        function = _gaussian

    # fit gaussian model to data
    popt, pcov = curve_fit(
        function, x, y, p0,
        sigma=yerror,
        absolute_sigma=params.absolute_sigma,
        method=params.fit_method
    )

    # estimate errors
    perr = np.sqrt(np.diag(pcov))

    # extract physical quantities from model fitting
    amplitude: Quantity = popt[0] * flux_unit
    amplitude_error: Quantity = perr[0] * flux_unit

    mu: Quantity = popt[1] * spectral_unit
    mu_error: Quantity = perr[1] * spectral_unit

    sigma: Quantity = popt[2] * spectral_unit
    sigma_error: Quantity = perr[2] * spectral_unit

    # compute integrated flux, FWHM, and their errors
    integrated_flux: Quantity = amplitude * sigma * np.sqrt(2*np.pi)
    flux_error: Quantity = integrated_flux * np.sqrt(
        (amplitude_error / amplitude)**2 +
        (sigma_error / sigma)**2
    )

    FWHM: Quantity = 2*sigma * np.sqrt(2*np.log(2))
    FWHM_error: Quantity = 2*sigma_error * np.sqrt(2*np.log(2))

    if params.model == 'gaussian_line' and len(popt) > 3:
        m = popt[3] * (flux_unit / spectral_unit)
        m_error = perr[3] * (flux_unit / spectral_unit)
        b = popt[4] * flux_unit
        b_error = perr[4] * flux_unit
    else:
        m = None
        m_error = None
        b = None
        b_error = None

    if params.model == 'gaussian_continuum' and continuum is not None:
        continuum_at_mu = np.interp(mu.value, x, continuum)
        peak_flux = (amplitude.value + continuum_at_mu) * flux_unit
    else:
        peak_flux = function(mu.value, *popt) * flux_unit

    fit_config={
        'gaussian_model': params.model,
        'curve_fit_method': params.fit_method,
        'absolute_sigma': params.absolute_sigma,
        'interpolate': params.interpolate,
    }
    if params.interpolate:
        fit_config.update(
            interpolate_method=params.interp_method,
            error_interp_method=params.error_interp_method,
            interpolate_samples=params.samples
        )

    result = GaussianFitResult(
        amplitude=amplitude,
        amplitude_error=amplitude_error,
        mu=mu,
        mu_error=mu_error,
        sigma=sigma,
        sigma_error=sigma_error,
        flux=integrated_flux,
        flux_error=flux_error,
        FWHM=FWHM,
        FWHM_error=FWHM_error,
        slope=m,
        slope_error=m_error,
        intercept=b,
        intercept_error=b_error,
        peak_height=peak_flux,
        popt=popt,
        pcov=pcov,
        perr=perr,
        p0=p0,
        fit_config=fit_config
    )

    fig, ax = subplots(figsize=kwargs.pop('figsize', config.figsize))

    xlim = kwargs.pop('xlim', spectral_range)
    colors = get_colors(_resolve_color_alias(kwargs, params.color))

    if plot_interp and params.interpolate:
        plot(x, y, ax=ax, color=_cycle(colors, 2), label='Interpolated')

    plot(
        x0*spectral_unit if spectral_unit is not None else x0,
        y0*flux_unit if flux_unit is not None else y0,
        ax=ax,
        color=_cycle(colors, 0),
        label=kwargs.pop('label', 'Spectrum'),
        **kwargs,
    )

    gaussian = function(x, *popt)
    ax.plot(
        x, gaussian,
        color=_cycle(colors, 1),
        label='Gaussian Model'
    )
    set_limits(ax=ax, xlim=xlim)

    legend(ax=ax)

    if params.savefig:
        savefigure()
    plt.show()

    if params.print_vals:
        result.pretty_print()

    if params.return_fit_params:
        return result
    else:
        default_return = ['flux', 'FWHM', 'mu', 'flux_error', 'FWHM_error', 'mu_error']
        GaussianHandles = namedtuple('GaussianFit', default_return)

        return GaussianHandles(
            integrated_flux, FWHM, mu, flux_error, FWHM_error, mu_error
        )
