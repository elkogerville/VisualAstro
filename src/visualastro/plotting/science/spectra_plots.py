"""
Author: Elko Gerville-Reache
Date Created: 2025-05-23
Date Modified: 2026-06-28
Description:
    Spectra science functions.
"""

from collections import namedtuple
from collections.abc import Sequence
from typing import Literal

import astropy.units as u
from astropy.units import Quantity
from matplotlib.colors import Colormap, to_rgba
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from spectral_cube import SpectralCube
from specutils import SpectralAxis
from specutils.spectra import Spectrum
from tqdm import tqdm

from visualastro.analysis.image_utils import stack_cube
from visualastro.analysis.spectra_utils import (
    GaussianFitResult,
    ExtractedPixelSpectra,
    deredden_flux,
    fit_continuum,
    gaussian as _gaussian,
    gaussian_continuum as _gaussian_continuum,
    gaussian_line as _gaussian_line,
    shift_by_radial_vel,
    _get_spectral_axis,
    _spectral_axis_2_array,
)
from visualastro.core.config import (
    get_config_value,
    config,
    _Unset,
    _UNSET,
    _resolve_default
)
from visualastro.core.io import savefig
from visualastro.core.kwargs import (
    _pop_kwargs, _param, _kwarg, _resolve_kwargs
)
from visualastro.core.numerical import interpolate as _interpolate
from visualastro.core.numerical_utils import (
    get_value,
    mask_within_range,
    to_array,
    to_list,
    _cycle,
    _unwrap_if_single
)
from visualastro.core.units import (
    ensure_common_unit,
    convert_quantity,
    get_unit,
)
from visualastro.datamodels.datacube import DataCube
from visualastro.datamodels.spectrumplus import SpectrumPlus
from visualastro.plotting.science.wcs_plots import imshow
from visualastro.plotting.core.colormaps import get_cmap
from visualastro.plotting.core.colors import (
    get_colors,
    sample_cmap,
    _lighten_color
)
from visualastro.plotting.core.axes import (
    set_axis_labels, set_axis_limits
)
from visualastro.plotting.core.style import _style_context
from visualastro.plotting.core.utils import (
    plot_vlines,
    _get_zorder
)


def extract_cube_spectra(
    cubes,
    flux_extract_method=_UNSET,
    extract_mode=None,
    fit_method=_UNSET,
    region=None,
    radial_vel=_UNSET,
    rest_freq=_UNSET,
    deredden=_UNSET,
    unit=_UNSET,
    emission_line=None,
    plot_continuum=_UNSET,
    plot_norm_continuum=_UNSET,
    **kwargs
):
    '''
    Extract 1D spectra from one or more data cubes, with optional continuum normalization,
    dereddening, and plotting.

    Parameters
    ----------
    cubes : DataCube, SpectralCube, or list of cubes
        Input cube(s) from which to extract spectra. The data must either be
        a SpectralCube, or a DataCube containing a SpectralCube.
    flux_extract_method : {'mean', 'median', 'sum'} or None, default=None
        Method for extracting the flux. If None, uses `config.flux_extract_method`.
    extract_mode : {'cube', 'slice', 'ray'} or None, default=None
        Specifies how the spectral cube should be traversed during flux
        extraction. This controls memory usage and performance for large cubes.
            - 'cube' :
                Load and operate on the entire cube in memory. This is the
                simplest mode but may be slow or disabled for very large datasets
                unless `cube.allow_huge_operations = True` is set.
            - 'slice' :
                Process the cube slice-by-slice along the spectral axis. This
                avoids loading the full cube into memory and is recommended for
                moderately large datasets.
            - 'ray' :
                Traverse the cube voxel-by-voxel ('ray-wise'), minimizing memory
                load at the cost of speed. Recommended for extremely large cubes
                or low-memory environments.
        If None, uses `config.spectral_cube_extraction_mode`.
    fit_method : {'fit_continuum', 'generic'} or None, optional, default=None
        Method used to fit the continuum. If None, uses `config.spectrum_continuum_fit_method`.
    region : array-like or None, optional, default=None
        Wavelength or pixel region(s) to use when `fit_method='fit_continuum'`.
        Ignored for other methods. This allows the user to specify which
        regions to include in the fit. Removing strong peaks is preferable to
        avoid skewing the fit up or down.
        Ex: Remove strong emission peak at 7um from fit
        region = [(6.5*u.um, 6.9*u.um), (7.1*u.um, 7.9*u.um)]
    radial_vel : float or None, optional, default=`_UNSET`
        Radial velocity in km/s to shift the spectral axis.
        Astropy units are optional. If None, ignores the radial velocity.
        If `_UNSET`, uses `config.radial_velocity`.
    rest_freq : float or None, optional, default=`_UNSET`
        Rest-frame frequency or wavelength of the spectrum. If None,
        ignores the rest frequency for unit conversions. If `_UNSET`,
        uses `config.spectra_rest_frequency`.
    deredden : bool or None, optional, default=None
        Whether to apply dereddening to the flux using deredden_flux().
        If None, uses `config.deredden_spectrum`.
    unit : str, astropy.units.Unit, or None, optional, default=`_UNSET`
        Desired units for the wavelength axis. Converts the default
        units if possible. If None, does not try and convert. If `_UNSET`,
        uses `config.wavelength_unit`.
    emission_line : str, optional, default=None
        Name of an emission line to annotate on the plot.
    plot_continuum : bool or None, optional, default=None
        Whether to overplot the continuum fit. If None, uses `config.plot_continuum_fit`.
    plot_norm_continuum : bool or None, optional, default=None
        Whether to plot the normalized extracted spectra. If None,
        uses `config.plot_normalized_continuum`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `how` : str, optional, default=`config.spectral_cube_extraction_mode`
            Alias for `extract_mode`.
        - `convention` : str, optional
            Doppler convention.
        - `Rv` : float, optional, default=`config.deredden.Rv`
            Dereddening parameter.
        - `Ebv` : float, optional, default=`config.deredden.Ebv`
            Dereddening parameter.
        - `deredden_method` : str, optional, default=`config.deredden.method`
            Extinction law to use.
        - `deredden_region` : str, optional, default=`config.deredden.region`
            Region/environment for WD01 extinction law.
        - `figsize` : tuple, optional, default=`config.figsize`
            Figure size for plotting.
        - `style` : str, optional, default=`config.style`
            Plotting style.
        - `savefig` : bool, optional, default=`config.savefig.enable`
            Whether to save the figure to disk.
        - `dpi` : int, optional, default=`config.savefig.dpi`
            Figure resolution for saving.
        - `rasterized` : bool, default=`config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `colors`, `color` or `c` : list of colors or None, optional, default=None
            Colors to use for each dataset. If None, default
            color cycle is used.
        - `linestyles`, `linestyle`, `ls` : str or list of str, default=`config.linestyle`
            Line style of plotted lines. Accepted styles: {'-', '--', '-.', ':', ''}.
        - `linewidths`, `linewidth`, `lw` : float or list of float, optional, default=`config.linewidth`
            Line width for the plotted lines.
        - `alphas`, `alpha`, `a` : float or list of float default=`config.alpha`
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        - `zorders`, `zorder` : float, default=None
            Order of line placement. If None, will increment by 1 for
            each additional line plotted.
        - `cmap` : str, optional, default=`config.cmap`
            Colormap to use if `colors` is not provided.
        - `xlim` : tuple, optional, default=None
            Wavelength range to display.
        - `ylim` : tuple, optional
            Flux range to display.
        - `labels`, `label`, `l` : str or list of str, default=None
            Legend labels.
        - `loc` : str, default=`config.legend.loc`
            Location of legend.
        - `xlabel` : str, optional
            Label for the x-axis.
        - `ylabel` : str, optional
            Label for the y-axis.
        - `text_loc` : list of float, optional, default=`config.text_loc`
            Location for emission line annotation text in axes coordinates.
        - `unit_bracket_style` : Literal['round', 'square'], optional, default=`config.unit_bracket_style`
            If `'round`' displays spectra units as (unit). If `'square`' as [unit].

    Returns
    -------
    SpectrumPlus or list of SpectrumPlus
        Single object if one cube is provided, list if multiple cubes are provided.
    '''
    params = _resolve_kwargs(
        kwargs,
        params=[
            _param('flux_extract_method', flux_extract_method, config.flux_extract_method),
            _param('extract_mode', extract_mode, config.spectral_cube_extraction_mode),
            _param('fit_method', fit_method, config.spectrum_continuum_fit_method),
            _param('radial_vel', radial_vel, config.radial_velocity),
            _param('rest_freq', rest_freq, config.spectra_rest_frequency),
            _param('deredden', deredden, config.deredden_spectrum),
            _param('unit', unit, config.wavelength_unit),
            _param('plot_continuum', plot_continuum, config.plot_continuum_fit),
            _param('plot_norm_continuum', plot_norm_continuum, config.plot_normalized_continuum),
        ],
        additional_kwargs=[
            _kwarg('convention', None),
            _kwarg('Rv', config.deredden.Rv),
            _kwarg('Ebv', config.deredden.Ebv),
            _kwarg('deredden_method', config.deredden.method),
            _kwarg('deredden_region', config.deredden.region),
            _kwarg('figsize', config.figsize),
            _kwarg('style', config.style),
            _kwarg('savefigure', config.savefig.enable),
            _kwarg('dpi', config.savefig.dpi),
        ]
    )

    methods = {
        'mean': lambda cube: cube.mean(axis=(1, 2), how=params.extract_mode),
        'median': lambda cube: cube.median(axis=(1, 2), how=params.extract_mode),
        'sum': lambda cube: cube.sum(axis=(1, 2), how=params.extract_mode)
    }

    extract_method = methods.get(params.flux_extract_method, None)
    if extract_method is None:
        raise ValueError(
            f"Invalid flux_extract_method '{flux_extract_method}'. "
            f'Choose from {list(methods.keys())}.'
        )

    cubes = to_list(cubes)
    ensure_common_unit(cubes)

    extracted_spectra = []

    for cube in cubes:

        spectral_axis = shift_by_radial_vel(cube.spectral_axis, params.radial_vel)
        spectral_axis = convert_quantity(spectral_axis, params.unit, equivalencies=u.spectral())

        flux = extract_method(cube)
        # convert to u.Quantity
        flux = flux.value * flux.unit

        if params.deredden:
            flux = deredden_flux(
                spectral_axis, flux, params.Rv, params.Ebv,
                params.deredden_method, params.deredden_region
            )

        spectrum = Spectrum(
            spectral_axis=spectral_axis,
            flux=flux,
            rest_value=params.rest_freq,
            velocity_convention=params.convention
        )

        continuum = fit_continuum(spectrum, params.fit_method, region)

        flux_normalized = spectrum / continuum

        extracted_spectra.append(SpectrumPlus(
            spectrum=spectrum,
            normalized=flux_normalized.flux,
            continuum=continuum,
            fit_method=params.fit_method,
            region=region
        ))

    with _style_context(params.style):
        fig, ax = plt.subplots(figsize=params.figsize)
        _ = plot_spectra(
            extracted_spectra,
            ax=ax,
            plot_continuum=params.plot_continuum,
            plot_norm_continuum=params.plot_norm_continuum,
            emission_line=emission_line,
            **kwargs
        )
        if params.savefigure:
            savefig(dpi=params.dpi)
        plt.show()

    return extracted_spectra


def extract_cube_pixel_spectra(
    cube: DataCube | SpectralCube | Quantity | NDArray,
    *,
    idx: int | Sequence[int] | None = None,
    idx_range: tuple[int, int] | None = None,
    idx_drop: int | Sequence[int] | None = None,
    combine_spectra: bool = False,
    combine_method: Literal['sum', 'mean', 'median'] | _Unset = _UNSET,
    plot_spatial_map: bool = True,
    vline: float | int | Quantity | None = None,
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
    cube : DataCube, SpectralCube, Quantity or ArrayLike
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
    background_cube : DataCube, SpectralCube, Quantity, ArrayLike | None, default=None
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
        Font size of the spatial map index legend.
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

        * `spectra` : SpectrumPlus or list of SpectrumPlus
            Extracted per-pixel spectra. If a single index is selected,
            this is returned as a single `SpectrumPlus` instance;
            otherwise, a list is returned.
        * `cube_array` : NDArray
            Copy of the original cube, with all pixels set to NaN
            other than the pixels corresponding to `extract_idx`.
        * `extract_idx` : ndarray of int, shape (N,)
            Indices of the extracted spectra corresponding to the ordering
            of valid spatial pixels.
        * `coords` : ndarray of int, shape (N, 2)
            Spatial coordinates of extracted pixels in `(y, x)` order.
        * `colors` : list
            Colors assigned to each extracted pixel/spectrum.
        * `labels` : list of str
            Human-readable labels of the form `<idx>: (x=<x>, y=<y>)`.
        * `combined_spectrum` : SpectrumPlus or None
            Combined spectrum computed using `combine_method` if
            `combine_spectra=True`; otherwise None.
    """
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
        SpectrumPlus(
            Spectrum(spectral_axis=spectral_axis, flux=flux)
        )
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

        combined_spec = SpectrumPlus(
            Spectrum(spectral_axis=spectral_axis, flux=combined_flux)
        )

    with _style_context(params.style):
        fig, ax = plt.subplots(figsize=params.figsize)
        ax.set_autoscale_on(False)

        if isinstance(cmap, (list, tuple, np.ndarray)) and len(cmap) >= n_plot:
            colors = list(cmap[:n_plot])
        else:
            colors = sample_cmap(n_plot, cmap)

        for i in tqdm(range(len(spectra)), desc='plotting'):
            plot_spectra(
                spectra[i],
                ax,
                color=[colors[i]],
                label=labels[i],
                plot_continuum=False,
            )

        fluxes = [spec.flux for spec in spectra]

        if combine_spectra and isinstance(combined_spec, SpectrumPlus):
            plot_spectra(
                combined_spec,
                ax,
                color='k',
                ls='--',
                label=f'combined ({params.combine_method})',
                plot_continuum=False
            )
            fluxes.append(combined_spec.flux)

        plot_vlines(vline, ax, fluxes[0].unit)

        set_axis_limits(
            spectral_axis,
            fluxes,
            ax=ax,
            xpad=0,
            **kwargs
        )

        if params.legend:
            ax.legend(
                ncols=params.ncols,
                fontsize=8,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                frameon=False,
            )
        if params.savefig:
            savefig(**kwargs)

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
                masked_data if params.background_cube is None
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

        * `cube_array` : 3D NDArray
          Numpy 3D array to plot spatial map onto.
        * `extract_idx` : array-like of int
          Indices of the extracted pixel spectra.
        * `coords` : array-like, shape `(N, 2)`
        Spatial pixel coordinates in `(y, x)` order.
        * `colors` : sequence
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

            * `i` -> returns `cube[i]`
            * `[i]` -> returns `cube[i]`
            * `[i, j]` -> returns `cube[i:j+1].sum(axis=0)`

        If `None`, uses `idx=0`.
    savefig : bool, optional, default=False
        If `True`, save the figure to disk using `savefig`.
    alpha : float, optional, default=0.8
        Alpha value for individual pixel colors.
    fontsize : float, optional, default=8
        Fontsize of annotations.
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
    * Spatial coordinates are interpreted in `(y, x)` order.
    * Extracted pixels are rendered as an RGBA overlay for efficient,
      vectorized plotting.
    * This function is intended to be used in conjunction with
      `extract_cube_pixel_spectra` and its returned extraction object.
    * For best results, set `idx` to the correct cube slice to preserve
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
        imshow(
            background,
            ax,
            origin='lower',
            cmap='gray',
            colorbar=False,
            **kwargs
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
            savefig(**kwargs)

        plt.show()


# Spectra Plotting Functions
# --------------------------
def plot_spectra(
    extracted_spectra=None,
    ax=None,
    plot_continuum=_UNSET,
    plot_norm_continuum=_UNSET,
    emission_line=None,
    wavelength=None,
    flux=None,
    continuum=None,
    color=_UNSET,
    vline=None,
    **kwargs
):
    '''
    Plot one or more extracted spectra on a matplotlib Axes.

    Parameters
    ----------
    extracted_spectrums : SpectrumPlus or list of SpectrumPlus, optional
        Pre-computed spectrum object(s) to plot. If not provided, `wavelength`
        and `flux` must be given.
    ax : matplotlib.axes.Axes
        Axis to plot on.
    plot_norm_continuum : bool, optional, default=None
        If True, plot normalized flux instead of raw flux.
        If None, uses `plot_normalized_continuum`.
    plot_continuum : bool, optional, default=None
        If True, overplot continuum fit. If None, uses
        the default value set by `config.plot_continuum_fit`.
    emission_line : str, optional, default=None
        Label for an emission line to annotate on the plot.
    wavelength : array-like, optional, default=None
        Wavelength array (required if `extracted_spectrums` is None).
    flux : array-like, optional, default=None
        Flux array (required if `extracted_spectrums` is None).
    continuum : array-like, optional, default=None
        Fitted continuum array.
    colors : list of colors, str, or None, optional, default=None
        Colors to use for each scatter group or dataset.
        If None, uses `config.default_colorset`.
    vline : Quantity or float or None, optional
        If provided, draw a vertical dotted reference line at this wavelength.
        If unitless, the value is assumed to be in the same units as the
        spectral axis.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `rasterized` : bool, default=`config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `color` or `c` : list of colors or None, optional, default=None
            Aliases for `colors`.
        - `linestyles`, `linestyle`, `ls` : str or list of str, default=`config.linestyle`
            Line style of plotted lines. Accepted styles: {'-', '--', '-.', ':', ''}.
        - `linewidths`, `linewidth`, `lw` : float or list of float, optional, default=`config.linewidth`
            Line width for the plotted lines.
        - `alphas`, `alpha`, `a` : float or list of float default=`config.alpha`
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        - `zorders`, `zorder` : float, default=None
            Order of line placement. If None, will increment by 1 for
            each additional line plotted.
        - `cmap` : str, optional, default=`config.cmap`
            Colormap to use if `colors` is not provided.
        - `xlim` : tuple, optional, default=None
            Wavelength range to display.
        - `ylim` : tuple, optional
            Flux range to display.
        - `labels`, `label`, `l` : str or list of str, default=None
            Legend labels.
        - `loc` : str, default=`config.legend.loc`
            Location of legend.
        - `xlabel` : str, optional
            Label for the x-axis.
        - `ylabel` : str, optional
            Label for the y-axis.
        - `text_loc` : list of float, optional, default=`config.text_loc`
            Location for emission line annotation text in axes coordinates.
        - `unit_bracket_style` : Literal['round', 'square'], optional, default=`config.unit_bracket_style`
            If `'round`' displays spectra units as (unit). If `'square`' as [unit].

    Returns
    -------
    lines : Line2D or list of Line2D, or PlotSpectrum
        The plotted line object(s) created by `Axes.plot`.

        - If `plot_continuum` is False, returns a single `Line2D` object
          or a list of `Line2D` objects corresponding to the main spectrum.
        - If `plot_continuum` is True, returns a `PlotSpectrum` named tuple
          with the following fields:
            * `lines` : Line2D or list of Line2D
              The plotted spectrum line(s).
            * `continuum_lines` : Line2D or list of Line2D
              The plotted continuum fit line(s), if available.
    '''
    params = _resolve_kwargs(
        kwargs,
        params=[
            _param('plot_continuum', plot_continuum, config.plot_continuum_fit),
            _param('plot_norm_continuum', plot_norm_continuum, config.plot_normalized_continuum),
            _param('color', color, config.color),
        ],
        additional_kwargs=[
            _kwarg('linestyle', None),
            _kwarg('linewidth', None),
            _kwarg('alpha', None),
            _kwarg('zorder', None),
            _kwarg('cmap', config.cmap),
            _kwarg('bad_color', None),
            _kwarg('label', None),
            _kwarg('xlabel', None),
            _kwarg('ylabel', None),
            _kwarg('xlim', None),
            _kwarg('ylim', None),
            _kwarg('xpad', config.axes.xpad),
            _kwarg('ypad', config.axes.ypad),
            _kwarg('loc', config.legend.loc),
            _kwarg('text_loc', config.plot_spectra_text_loc),
            _kwarg('unit_bracket_style', config.unit_bracket_style),
            _kwarg('rasterized', config.rasterized),
        ]
    )
    cmap = get_cmap(params.cmap, params.bad_color)
    colors = get_colors(params.color, cmap=cmap)
    fit_colors = [_lighten_color(c) for c in colors]

    alphas = to_list(params.alpha)
    labels = to_list(params.label)
    linestyles = to_list(params.linestyle)
    linewidths = to_list(params.linewidth)
    zorders = to_list(params.zorder)

    if ax is None:
        raise ValueError('ax must be a matplotlib axes object!')

    # construct SpectrumPlus if user passes in wavelength and flux
    if extracted_spectra is None:
        # disable normalization because the user provided raw arrays
        params.plot_norm_continuum = False

        # normalize continuum_fit into a list
        if isinstance(continuum, (list, tuple)):
            continuum_list = list(continuum)
        else:
            continuum_list = [continuum]

        # case 1: single wavelength/flux array
        if (
            isinstance(wavelength, (np.ndarray, Quantity, SpectralAxis)) and
            isinstance(flux, (np.ndarray, Quantity))
        ):
            extracted_spectra = SpectrumPlus(
                spectral_axis=wavelength,
                flux=flux,
                continuum=continuum_list[0]
            )
        # case 2: multiple arrays
        elif (
            isinstance(wavelength, (list, tuple)) and
            isinstance(flux, (list, tuple)) and
            len(wavelength) == len(flux)
        ):
            extracted_spectra = [
                SpectrumPlus(
                    spectral_axis=w,
                    flux=f,
                    continuum=continuum_list[i % len(continuum_list)]
                )
                for i, (w, f) in enumerate(zip(wavelength, flux))
            ]
        else:
            raise ValueError(
                'Either pass `extracted_spectra`, or provide matching '
                '`wavelength` and `flux` arguments. \nFor multiple spectra, '
                'use lists of wavelength and flux arrays with equal length.'
            )

    extracted_spectra = to_list(extracted_spectra)
    ensure_common_unit(extracted_spectra, on_mismatch=config.unit_mismatch)

    if emission_line is not None:
        ax.text(
            params.text_loc[0], params.text_loc[1],
            f'{emission_line}', transform=ax.transAxes
        )

    lines = []
    fit_lines = []
    wavelength_list = []
    flux_list = []
    flux_mask_list = []

    for i, extracted_spectrum in enumerate(extracted_spectra):

        w = extracted_spectrum.spectral_axis
        if params.plot_norm_continuum:
            f = extracted_spectrum.normalized
        else:
            f = extracted_spectrum.flux
        wave_arr = _spectral_axis_2_array(w)
        flux_arr = to_array(f)

        mask = mask_within_range(wave_arr, xlim=params.xlim)
        wavelength_list.append(w[mask])
        flux_list.append(f)
        flux_mask_list.append(flux_arr[mask])

        color = _cycle(colors, i)
        fit_color = _cycle(fit_colors, i)
        linestyle = _cycle(linestyles, i)
        linewidth = _cycle(linewidths, i)
        alpha = _cycle(alphas, i)
        zorder = _get_zorder(zorders, i, config.zorder.plot_data)
        label = labels[i] if (_cycle(labels, i) is not None and i < len(labels)) else None

        l = ax.plot(
            wave_arr[mask], flux_arr[mask],
            color=color,
            ls=linestyle,
            lw=linewidth,
            alpha=alpha,
            zorder=zorder,
            label=label,
            rasterized=params.rasterized
        )

        if params.plot_continuum and extracted_spectrum.continuum is not None:
            if params.plot_norm_continuum:
                continuum = extracted_spectrum.continuum/extracted_spectrum.continuum
            else:
                continuum = extracted_spectrum.continuum
            fl = ax.plot(
                wave_arr[mask], to_array(continuum)[mask],
                color=fit_color,
                ls=linestyle,
                lw=linewidth,
                alpha=alpha,
                rasterized=params.rasterized
            )

            fit_lines.append(fl)

        lines.append(l)

    set_axis_limits(
        wavelength_list, flux_mask_list,
        ax=ax,
        xlim=params.xlim, ylim=params.ylim,
        xpad=0
    )
    set_axis_labels(
        _cycle(wavelength_list, config.reference_idx),
        _cycle(flux_list, config.reference_idx),
        ax,
        params.xlabel, params.ylabel,
        unit_bracket_style=params.unit_bracket_style
    )

    plot_vlines(vline, ax, _cycle(extracted_spectra, config.reference_idx).unit)

    if labels[0] is not None:
        ax.legend(loc=params.loc)

    if params.plot_continuum:
        PlotHandles = namedtuple('PlotSpectrum', ['lines', 'continuum_lines'])
        return PlotHandles(lines, fit_lines)

    return lines


def plot_combine_spectrum(extracted_spectra, ax, idx=0, wave_cuttofs=None,
                          concatenate=False, return_spectra=False,
                          plot_normalize=False, use_samecolor=True,
                          colors=_UNSET, **kwargs):
    '''
    Allows for easily plotting multiple spectra and stiching them together into
    one `SpectrumPlus` object.

    Parameters
    ----------
    extracted_spectra : list of `SpectrumPlus`/`Spectrum`, or list of list of `SpectrumPlus`/`Spectrum`
        List of spectra to plot. Each element should contain wavelength and flux attributes,
        and optionally the normalize attribute.
    ax : matplotlib.axes.Axes
        Axis on which to plot the spectra.
    idx : int, optional, default=0
        Index to select a specific spectrum if elements of `extracted_spectra` are lists.
        This is useful when extracting spectra from multiple regions at once.
        Ex:
            spec_1 = [spectrum1, spectrum2]
            spec_2 = [spectrum3, spectrum4]
            extracted_spectra = [spec_1[idx], spec_2[idx]]
    wave_cuttofs : list of float, optional, default=None
        Wavelength limits of each spectra used to mask spectra when stiching together.
        If provided, should contain the boundary wavelengths in sequence (e.g., [λ₀, λ₁, λ₂, ...λₙ]).
        Note:
            If N spectra are provided, ensure there are N+1 limits. For each i spectra, the
            program will define the limits as `wave_cuttofs[i]` < `spectra[i]` < `wave_cuttofs[i+1]`.
    concatenate : bool, optional, default=False
        If True, concatenate all spectra and plot as a single continuous curve.
    return_spectra : bool, optional, default=False
        If True, return the concatenated `SpectrumPlus` object instead of only plotting.
        If True, `concatenate` is set to True.
    plot_normalize : bool, optional, default=False
        If True, plot the normalized flux instead of the raw flux.
    use_samecolor : bool, optional, default=True
        If True, use the same color for all spectra. If `concatenate` is True,
        `use_samecolor` is also set to True.
    colors : list of colors, str, or None, optional, default=None
        Colors to use for each scatter group or dataset.
        If None, uses `config.default_colorset`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `rasterized` : bool, default=`config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - ylim : tuple, optional, default=None
            y-axis limits as (ymin, ymax).
        - `color` or `c` : list of colors or None, optional, default=None
            Aliases for `colors`.
        - `linestyles`, `linestyle`, `ls` : str or list of str, default=`config.linestyle`
            Line style of plotted lines. Accepted styles: {'-', '--', '-.', ':', ''}.
        - `linewidths`, `linewidth`, `lw` : float or list of float, optional, default=`config.linewidth`
            Line width for the plotted lines.
        - `alphas`, `alpha`, `a` : float or list of float default=`config.alpha`
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        - cmap : str, optional, default=`config.cmap`
            Colormap name for generating colors.
        - label : str, optional, default=None.
            Label for the plotted spectrum.
        - loc : str, optional, default=`config.legend.loc`
            Legend location (e.g., 'best', 'upper right').
        - xlabel, ylabel : str, optional, default=None
            Axis labels.
        - unit_bracket_style : Literal['round', 'square'], optional, default=`config.unit_bracket_style`
            If `'round`' displays spectra units as (unit). If `'square`' as [unit].

    Returns
    -------
    SpectrumPlus or None
        If `return_spectra` is True, returns the concatenated spectrum.
        Otherwise, returns None.

    Notes
    -----
    - If `concatenate` is True, all spectra are merged and plotted as one line.
    - If `wave_cuttofs` is provided, each spectrum is masked to its corresponding
    wavelength interval before plotting.
    '''
    # ---- KWARGS ----
    # figure params
    rasterized = kwargs.get('rasterized', config.rasterized)
    ylim = kwargs.get('ylim', None)
    # line params
    colors = _pop_kwargs(kwargs, 'color', 'c', default=colors)
    linestyles = _pop_kwargs(kwargs, 'linestyles', 'linestyle', 'ls', default=None)
    linewidths = _pop_kwargs(kwargs, 'linewidths', 'linewidth', 'lw', default=None)
    alphas = _pop_kwargs(kwargs, 'alphas', 'alpha', 'a', default=None)
    cmap = kwargs.get('cmap', config.cmap)
    # labels
    label = kwargs.get('label', None)
    loc = kwargs.get('loc', config.legend.loc)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    unit_bracket_style = kwargs.get('unit_bracket_style', config.unit_bracket_style)

    # get default config values
    colors = _resolve_default(colors, config.color)
    linestyles = get_config_value(linestyles, 'linestyle')
    linewidths = get_config_value(linewidths, 'linewidth')
    alphas = get_config_value(alphas, 'alpha')

    # ensure units match and that extracted_spectra is a list
    extracted_spectra = to_list(extracted_spectra)
    ensure_common_unit(extracted_spectra)
    # hardcode behavior to avoid breaking
    if return_spectra:
        concatenate = True
    if concatenate:
        use_samecolor = True

    # set plot style and colors
    colors = get_colors(colors, cmap=cmap)

    wave_list = []
    flux_list = []
    wavelength_lims = []
    for i, spectrum in enumerate(extracted_spectra):
        # index spectrum if list
        spectrum = spectrum[idx] if isinstance(spectrum, list) else spectrum
        # extract wavelength and flux
        wavelength = spectrum.spectral_axis
        flux = spectrum.normalize if plot_normalize else spectrum.flux
        # compute minimum and maximum wavelength values
        wmin = np.nanmin(get_value(wavelength))
        wmax = np.nanmax(get_value(wavelength))
        wavelength_lims.append( [wmin, wmax] )
        # mask wavelength and flux if user passes in limits
        if wave_cuttofs is not None:
            wave_min: float = wave_cuttofs[i]
            wave_max: float = wave_cuttofs[i+1]
            mask = mask_within_range(get_value(wavelength), (wave_min, wave_max))
            wavelength = wavelength[mask]
            flux = flux[mask]

        c = colors[0] if use_samecolor else colors[i%len(colors)]
        # only plot a label for combined spectrum, not each sub spectra
        l = label if label is not None and i == len(extracted_spectra)-1 else None
        # append to lists if concatenate
        if concatenate:
            wave_list.append(wavelength)
            flux_list.append(flux)
        # plot spectrum if not concatenating
        else:
            ax.plot(wavelength, flux, color=c,
                    label=l, ls=linestyles,
                    lw=linewidths, alpha=alphas,
                    rasterized=rasterized)
    # plot entire spectrum if concatenate
    if concatenate:
        wavelength = np.concatenate(wave_list)
        flux = np.concatenate(flux_list)

        ax.plot(get_value(wavelength),
                get_value(flux),
                color=c, label=l, ls=linestyles,
                lw=linewidths, alpha=alphas,
                rasterized=rasterized)

    set_axis_labels(wavelength, flux, ax, xlabel, ylabel, unit_bracket_style)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if wave_cuttofs is None:
        xmin = min(l[0] for l in wavelength_lims)
        xmax = max(l[1] for l in wavelength_lims)
    else:
        xmin = wave_cuttofs[0]
        xmax = wave_cuttofs[-1]
    ax.set_xlim(xmin, xmax)

    if label is not None:
        ax.legend(loc=loc)

    if return_spectra:
        extracted_spectrum = SpectrumPlus(wavelength, flux)

        return extracted_spectrum


# Spectra Fitting Functions
# -------------------------
def fit_gaussian_2_spec(
    extracted_spectrum, p0, *, model=None, spectral_range=None,
    fit_method=None, absolute_sigma=None, yerror=None,
    interpolate=_UNSET, samples=None, interp_method=None,
    error_interp_method=None, return_fit_params=None,
    plot_interp=False, print_vals=None, **kwargs
):
    '''
    Fit a Gaussian-like model to a Spectrum, optionally including the continuum.

    Parameters
    ----------
    extracted_spectrum : SpectrumPlus or Spectrum
        Spectrum object to be gaussian fitted.
    p0 : list
        Initial guess for the Gaussian fit parameters.
        This should match the input arguments of the
        gaussian model (excluding the first argument
        which is wavelength).
    model : {'gaussian', 'gaussian_line', 'gaussian_continuum'} or None, default=None
        Type of Gaussian model to fit:
        - 'gaussian' : standard Gaussian
        - 'gaussian_line' : Gaussian with linear continuum
        - 'gaussian_continuum' : Gaussian with computed continuum array
        The continuum can be computed with fit_continuum().
        If None, uses `config.gaussian_model`.
    spectral_range : array-like or None, optional, default=None
        (min, max) wavelength range to restrict the fit.
        If None, computes the min and max from the wavelength.
    fit_method : {'lm', 'trf', 'dogbox'} or None, optional, default=None
        Curve fitting algorithm used by `scipy.optimize.curve_fit`.
        If None, uses `config.curve_fit.method`.
    absolute_sigma : boolean, optional, default=None
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
        If None, uses `config.curve_fit.absolute_sigma`.
    yerror : array-like or None, optional, default=None
        Flux uncertainties to be used in the fit. If None,
        uncertainties are ignored when computing the fit.
        This is passed to `curve_fit` as the `sigma` parameter.
    interpolate : bool | _Unset, default=_UNSET
        Whether to interpolate the spectrum over a regular wavelength grid.
        The number of samples is controlled by `samples`. If `_UNSET`, uses `config.curve_fit.interpolate`.
    samples : int or None, default=None
        Number of points in interpolated wavelength grid. If
        None, uses `config.curve_fit.samples`.
    interp_method : {'cubic', 'cubic_spline', 'linear'} or None, default=None
        Interpolation method used. If None, uses `config.curve_fit.interpolation_method`.
    error_interp_method : {'cubic', 'cubic_spline', 'linear'} or None, default=None
        Method to interpolate yerror if provided. If None, uses
        the default value set by `config.curve_fit.error_interpolation_method`.
    return_fit_params : bool or None, default=None
        If True, return full computed best-fit parameters for all parameters,
        including popt, pcov, and perr. If False, return only Flux, FWHM, and mu.
        If None, uses `config.return_gaussian_fit_parameters`.
    plot_interp : bool, default=False
        If True, plot the interpolated spectrum. This is
        provided for debugging purposes.
    print_vals : bool or None, default=None
        If True, print a table of best-fit parameters,
        errors, and computed quantities. If None, uses `config.print_gaussian_values`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `figsize` : list or tuple, optional, default=`config.figsize`
            Figure size.
        - `style` : str or {'astro', 'latex', 'minimal', 'default'}, optional, default=`config.style`
            Plot style used. Can either be a matplotlib mplstyle
            or an included visualastro style.
        - `xlim` : tuple, optional, default=None
            Wavelength range for plotting. If None, uses `wave_range`.
        - `plot_type` : {'plot', 'scatter'}, optional, default='plot'
            Matplotlib plotting style to use.
        - `label` : str, optional, default=None
            Spectrum legend label.
        - `xlabel` : str, optional, default=None
            Plot x-axis label.
        - `ylabel` : str, optional, default=None
            Plot y-axis label.
        - `colors` : str or list, optional, default=`config.color`
            Plot colors. If None, will use default visualastro color colorset.
        - `unit_bracket_style` : Literal['round', 'square'], optional, default=`config.unit_bracket_style`
            If `'round`' displays spectra units as (unit). If `'square`' as [unit].
        - `savefig` : bool, optional, default=`config.savefig.enable`
            If True, save current figure to disk.
        - `dpi` : float or int, optional, default=`config.savefig.dpi`
            Resolution in dots per inch.

    Returns
    -------
    GaussianFitResult or GaussianHandles

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
    '''
    # ---- KWARGS ----
    # figure params
    figsize = kwargs.get('figsize', config.figsize)
    style = kwargs.get('style', config.style)
    xlim = kwargs.get('xlim', None)
    plot_type = kwargs.get('plot_type', 'plot')
    # labels
    label = kwargs.get('label', None)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    colors = _pop_kwargs(kwargs, 'colors', 'color', 'c', default=None)
    unit_bracket_style = kwargs.get('unit_bracket_style', config.unit_bracket_style)
    # savefig
    savefig = kwargs.get('savefig', config.savefig.enable)
    dpi = kwargs.get('dpi', config.savefig.dpi)

    # get default config values
    colors = get_config_value(colors, 'colors')
    model = get_config_value(model, 'gaussian_model')
    fit_method = _resolve_default(fit_method, config.curve_fit.interpolation_method)
    absolute_sigma = _resolve_default(absolute_sigma, config.curve_fit.absolute_sigma)
    interpolate = _resolve_default(interpolate, config.curve_fit.interpolate)
    interp_method = _resolve_default(interp_method, config.curve_fit.interpolation_method)
    error_interp_method = _resolve_default(error_interp_method, config.curve_fit.error_interpolation_method)
    samples = _resolve_default(samples, config.curve_fit.samples)
    return_fit_params = get_config_value(return_fit_params, 'return_gaussian_fit_parameters')
    print_vals = get_config_value(print_vals, 'print_gaussian_values')

    # ensure arrays are not quantity objects
    spectral_axis = extracted_spectrum.spectral_axis
    flux = extracted_spectrum.flux

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

    if model == 'gaussian_continuum':
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
    if model == 'gaussian_continuum' and continuum is not None:
        continuum = continuum[spectral_mask]

    finite_mask = np.isfinite(x0) & np.isfinite(y0)

    x0 = x0[finite_mask]
    y0 = y0[finite_mask]

    if yerror is not None:
        yerror = yerror[finite_mask]

    if model == 'gaussian_continuum' and continuum is not None:
        continuum = continuum[finite_mask]

    if len(x0) == 0:
        raise ValueError(
            f'No data points within spectral_range {spectral_range}'
        )

    # interpolate arrays
    if interpolate:
        # interpolate wavelength and flux arrays
        x, y = _interpolate(
            x0, y0, spectral_range, samples, method=interp_method
        )
        # interpolate y error values
        if yerror is not None:
            _, yerror = _interpolate(
                x0, yerror, spectral_range, samples, method=error_interp_method
            )
        # interpolate continuum array
        if model == 'gaussian_continuum' and continuum is not None:
            _, continuum = _interpolate(
                x0, continuum, spectral_range, samples, method=interp_method
            )
    else:
        x, y = x0, y0

    if model == 'gaussian_continuum' and continuum is not None:
        def function(x, A, mu, sigma):
            return _gaussian_continuum(x, A, mu, sigma, continuum)

    elif model == 'gaussian_line':
        function = _gaussian_line

    else:
        function = _gaussian

    # fit gaussian model to data
    popt, pcov = curve_fit(
        function, x, y, p0,
        sigma=yerror,
        absolute_sigma=absolute_sigma,
        method=fit_method
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

    if model == 'gaussian_line' and len(popt) > 3:
        m = popt[3] * (flux_unit / spectral_unit)
        m_error = perr[3] * (flux_unit / spectral_unit)
        b = popt[4] * flux_unit
        b_error = perr[4] * flux_unit
    else:
        m = None
        m_error = None
        b = None
        b_error = None

    if model == 'gaussian_continuum' and continuum is not None:
        continuum_at_mu = np.interp(mu.value, x, continuum)
        peak_flux = (amplitude.value + continuum_at_mu) * flux_unit
    else:
        peak_flux = function(mu.value, *popt) * flux_unit

    fit_config={
        'gaussian_model': model,
        'curve_fit_method': fit_method,
        'absolute_sigma': absolute_sigma,
        'interpolate': interpolate,
    }
    if interpolate:
        fit_config.update(
            interpolate_method=interp_method,
            error_interp_method=error_interp_method,
            interpolate_samples=samples
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

    # set plot style and colors
    colors = get_colors(colors)

    with _style_context(style):
        fig, ax = plt.subplots(figsize=figsize)
        # determine plot type
        plt_plot = {
            'plot': ax.plot,
            'scatter': ax.scatter
        }.get(plot_type, ax.plot)
        # plot interpolated data
        if plot_interp and interpolate:
            plt_plot(
                x, y, c=colors[2%len(colors)], label='Interpolated'
            )
        # plot original data
        # clip values outisde of plotting range
        xlim = spectral_range if xlim is None else xlim
        plot_mask = mask_within_range(x0, xlim)
        gauss_mask = mask_within_range(x, xlim)
        label = label if label is not None else 'Spectrum'

        plt_plot(x0[plot_mask], y0[plot_mask],
                 c=colors[0%len(colors)], label=label)
        # plot gaussian model
        gaussian = function(x, *popt)
        ax.plot(
            x,
            gaussian,
            c=colors[1%len(colors)],
            label='Gaussian Model'
        )
        # set axis labels and limits
        set_axis_labels(
            spectral_axis, flux, ax, xlabel, ylabel, unit_bracket_style
        )
        set_axis_limits(
            [x0[plot_mask], x[gauss_mask]],
            [y0[plot_mask], gaussian[gauss_mask]],
            ax=ax, xlim=xlim
        )

        plt.legend()
        if savefig:
            savefig(dpi=dpi)
        plt.show()

    if print_vals:
        result.pretty_print(**kwargs)

    if return_fit_params:
        return result
    else:
        default_return = ['flux', 'FWHM', 'mu', 'flux_error', 'FWHM_error', 'mu_error']
        GaussianHandles = namedtuple('GaussianFit', default_return)

        return GaussianHandles(
            integrated_flux, FWHM, mu, flux_error, FWHM_error, mu_error
        )
