"""
Author: Elko Gerville-Reache
Date Created: 2025-05-23
Date Modified: 2026-03-11
Description:
    Plotting functions for 2D and 3D astronomical images.
"""

from typing import Literal
import warnings

import astropy.units as u
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization.wcsaxes.core import WCSAxes
from astropy.wcs import WCS
import matplotlib.axes as maxes
from matplotlib.colors import Colormap
from matplotlib.image import AxesImage
import numpy as np
from numpy.typing import NDArray
from spectral_cube import SpectralCube
from spectral_cube.lower_dimensional_structures import Slice

from visualastro.analysis.image_utils import stack_cube
from visualastro.core.config import (
    config,
    _Unset,
    _UNSET
)
from visualastro.core.io import _kwarg, _param, _resolve_kwargs
from visualastro.core.numerical_utils import (
    as_list,
    get_data,
    get_value,
    to_list,
    _cycle,
)
from visualastro.core.units import ensure_common_unit
from visualastro.datamodels.datacube import DataCube
from visualastro.plotting.core.colors import get_cmap
from visualastro.plotting.core.interface import (
    _apply_plot_utils, _extract_plot_util_kwargs
)
from visualastro.plotting.core.image_utils import compute_imshow_scale
from visualastro.plotting.science.spectra_plot_utils import spectral_axis_label


warnings.filterwarnings('ignore', category=AstropyWarning)


def imshow(
    images: DataCube | SpectralCube | NDArray | u.Quantity | list[DataCube | SpectralCube | NDArray | u.Quantity],
    ax: maxes.Axes | WCSAxes,
    idx: int | tuple[int, int] | list[int | tuple[int, int] | None] | None = None,
    vmin: float | _Unset = _UNSET,
    vmax: float | _Unset = _UNSET,
    norm: Literal['asinh', 'asinhnorm', 'log', 'power', 'twoslope', 'linear'] | None | _Unset = _UNSET,
    percentile: tuple[float, float] | _Unset = _UNSET,
    stack_method: Literal['mean', 'median', 'sum', 'max', 'min', 'std'] | _Unset = _UNSET,
    origin: Literal['lower', 'upper'] | _Unset = _UNSET,
    cmap: Colormap | str | _Unset = _UNSET,
    aspect: Literal['auto', 'equal'] | float | None | _Unset = _UNSET,
    mask_non_pos: bool | _Unset = _UNSET,
    axis: int = 0,
    **kwargs
) -> list[AxesImage]:
    """
    Display 2D image data with optional overlays and customization.

    Parameters
    ----------
    datas : DataCube | SpectralCube | NDArray | u.Quantity | list of such
        Image or list of images to plot. Each image should be 2D (Ny, Nx),
        or 3D (Nz, Nx, Ny) if using `idx` to slice the cube.
    ax : WCSAxes | matplotlib.axes.Axes
        The axes on which to draw the slice.
    idx : int | tuple[int, int] | None | list[int | tuple[int, int] | None]
        Spectral axis indices to slice. Tuples denote [start, end] (inclusive)
        and are collapsed via `stack_method`. If int, applies to all cubes; if list,
        each entry specifies indices for the corresponding cube. `None` collapses
        the entire spectral axis via `stack_method`.
    vmin, vmax: float | None | _Unset, optional, default=`_UNSET`
        Lower / upper limits for colormap scaling; overides `percentile`.
        If `None`, values are determined from `percentile[0]` for `vmin`
        and `percentile[1]` for `vmax`. If `_UNSET`, uses `config.vmin`
        and `config.vmax`.
    norm : str | None | _Unset, optional, default=_UNSET
        Normalization algorithm for colormap scaling.

        * `'asinh'` -> asinh stretch using `ImageNormalize`
        * `'asinhnorm'` -> asinh stretch using `AsinhNorm`
        * `'log'` -> logarithmic scaling using `LogNorm`
        * `'power'` -> power-law normalization using `PowerNorm`
        * `'twoslope'` -> twoslope normalization use `TwoSlopeNorm`
        * `'linear'`, `'none'`, or `None` -> no normalization applied

        If `_UNSET`, uses `config.norm`.
    percentile : tuple[float, float] | _Unset, default=_UNSET
        Default percentile range used to determine `vmin` and `vmax`.
        If `None`, use no percentile stretch (as long as `vmin` and
        `vmax`, are None). If `_UNSET`, uses `config.percentile`.
    stack_method : {'mean', 'median', 'sum', 'max', 'min', 'std'} | _Unset, default=_UNSET
        Stacking method. This controls how the cube is collapsed into a 2D
        image for plotting. The axis along to flatten is controlled by `axis`.
        If `_UNSET`, uses `config.stack_cube_method`.
    origin : {'upper', 'lower'} | _Unset, default=_UNSET
        Pixel origin convention for imshow. If `_UNSET`, uses `config.origin`.
    cmap : Colormap | str | list[Colormap | str] | _Unset, default=_UNSET
        Colormap(s) to use for plotting. If `_UNSET`, uses `config.cmap`.
    aspect : {'auto', 'equal'} | float | None | _Unset, optional, default=_UNSET
        Aspect ratio passed to imshow, shortcut for `Axes.set_aspect`. `'auto'`
        results in fixed axes with the aspect adjusted to fit the axes. `'equal``
        sets an aspect ratio of 1. `None` defaults to `'equal'`, however, if the
        image uses a transform that does not contain the axes data transform,
        then `None` means to not modify the axes aspect at all. If `_UNSET`,
        uses `config.aspect`.
    mask_non_pos : bool | _Unset, optional, default=_UNSET
        If `True`, mask out non-positive data values. Useful for displaying
        log scaling of images with non-positive values. If `_UNSET`, uses
        `config.mask_non_positive`. The mask value is set by `mask_out_val`.
        axis : int, optional, default=0
    axis : int, optional, default=0
        Axis used for collapsing 3D cubes into a 2D image.
    rasterized : bool, optional, default=config.rasterized
        Whether to rasterize plot artists. Rasterization
        converts the artist to a bitmap when saving to
        vector formats (e.g., PDF, SVG), which can
        significantly reduce file size for complex plots.
    mask_out_val : float, optional, default=config.mask_out_value
        Value to use when masking out non-positive values.
        ie: np.nan, 1e-6, np.inf
    linear_width : float, optional, default=config.linear_width
        The effective width of the linear region, beyond which the
        transformation becomes asymptotically logarithmic.
        Used for `norm='asinhnorm'`.
    gamma : float, optional, default=config.gamma
        Power law exponent. Used for `norm='power'`.
    vcenter : float, optional, default=None
        Center point of normalization. Must be in between `vmin` and `vmax`.
        If `None`, is the midpoint between `vmin` and `vmax`.
    invert_xaxis, invert_yaxis : bool, optional, default=False
        Invert the x or y axis if True.

    - `text_loc` : list of float, optional, default=`config.text_loc`
        Relative axes coordinates for text placement when
        plotting interactive ellipses.
    - `text_color` : str, optional, default=`config.text_color`
        Color of the ellipse annotation text.
    - `xlabel` : str, optional, default=None
        X-axis label.
    - `ylabel` : str, optional, default=None
        Y-axis label.
    - `colorbar` : bool, optional, default=`config.colorbar.enable`
        Add colorbar if True.
    - `clabel` : str or bool, optional, default=`config.colorbar.label`
        Colorbar label. If True, use default label; if None or False, no label.
    - `cbar_width` : float, optional, default=`config.colorbar.width`
        Width of the colorbar.
    - `cbar_pad` : float, optional, default=`config.colorbar.pad`
        Padding between plot and colorbar.
    - `mask_out_val` : float, optional, default=`config.mask_out_value`
        Value to use when masking out non-positive values.
        Ex: np.nan, 1e-6, np.inf
    - `circles` : list, optional, default=None
        List of Circle objects (e.g., `matplotlib.patches.Circle`) to overplot on the axes.
    - `ellipses` : list, optional, default=None
        List of Ellipse objects (e.g., `matplotlib.patches.Ellipse`) to overplot on the axes.
        Single Ellipse objects can also be passed directly.
    - `points` : array-like, shape (2,) or (N, 2), optional, default=None
        Coordinates of points to overplot. Can be a single point `[x, y]`
        or a list/array of points `[[x1, y1], [x2, y2], ...]`.
        Points are plotted as red stars by default.
    - `plot_ellipse` : bool, optional, default=False
        If True, plot an interactive ellipse overlay. Requires an interactive backend.

    Returns
    -------
    images : list[matplotlib.image.AxesImage]
        list of `AxesImage` objects returned by the `ax.imshow`
        call for each image in `images`.
    """
    params = _resolve_kwargs(
        kwargs,
        params=[
            _param('vmin', vmin, config.vmin),
            _param('vmax', vmax, config.vmax),
            _param('norm', norm, config.norm),
            _param('percentile', percentile, config.percentile),
            _param('stack_method', stack_method, config.stack_cube_method),
            _param('origin', origin, config.origin),
            _param('cmap', cmap, config.cmap),
            _param('aspect', aspect, config.aspect),
            _param('mask_non_pos', mask_non_pos, config.mask_non_positive),
        ],
        additional_kwargs=[
            _kwarg('unit_fmt', config.unit_label_format),
            _kwarg('rasterized', config.rasterized),
            _kwarg('mask_out_val', config.mask_out_value),
            _kwarg('stack_method', config.stack_cube_method),
            _kwarg('linear_width', config.linear_width),
            _kwarg('gamma', config.gamma),
            _kwarg('vcenter', None),
            _kwarg('invert_xaxis', False),
            _kwarg('invert_yaxis', False),
            _kwarg('rotate_wcs_labels', False),
        ],
    )
    plot_params = _extract_plot_util_kwargs(kwargs)

    images = to_list(images)
    idxs = [idx] if isinstance(idx, list) else as_list(idx)
    cmaps = to_list(params.cmap)

    ref_unit = ensure_common_unit(images)

    if isinstance(ax, WCSAxes) and origin == 'upper':
        warnings.warn(
            "origin cannot be 'upper' if ax is a WCSAxes! "
            "setting invert_yaxis=True and origin='lower'!"
        )
        origin = 'lower'
        params.invert_yaxis = True

    image_list = []

    for i, image in enumerate(images):
        data_slice = stack_cube(
            image, idx=_cycle(idxs, i), method=stack_method, axis=axis
        )
        data = get_value(data_slice)

        if mask_non_pos:
            data = np.where(data > 0.0, data, params.mask_out_val)

        img_norm, cube_vmin, cube_vmax = compute_imshow_scale(
            data=data,
            norm=params.norm,
            vmin=params.vmin,
            vmax=params.vmax,
            percentile=params.percentile,
            linear_width=params.linear_width,
            gamma=params.gamma,
            vcenter=params.vcenter,
        )

        cm = get_cmap(_cycle(cmaps, i))

        imshow_kwargs = dict(kwargs)
        if norm is None:
            imshow_kwargs.pop('norm', None)
            imshow_kwargs.update(vmin=cube_vmin, vmax=cube_vmax)
        else:
            imshow_kwargs.pop('vmin', None)
            imshow_kwargs.pop('vmax', None)
            imshow_kwargs.update(norm=img_norm)

        im = ax.imshow(
            data,
            cmap=cm,
            origin=params.origin,
            aspect=params.aspect,
            rasterized=params.rasterized,
            **imshow_kwargs
        )

        image_list.append(im)

    if params.invert_xaxis: ax.invert_xaxis()
    if params.invert_yaxis: ax.invert_yaxis()

    if params.rotate_wcs_labels:
        if isinstance(ax, WCSAxes):
            for coord in ax.coords:
                coord_type = coord.coord_type
                coord_index = coord.coord_index

                # set label based on coordinate type
                if coord_type == 'longitude':
                    coord.set_axislabel(config.right_ascension_label)
                elif coord_type == 'latitude':
                    coord.set_axislabel(config.declination_label)

                # determine which pixel axis this world coordinate
                # primarily affects by checking the PC/CD matrix
                wcs = ax.wcs
                if isinstance(wcs, WCS) and hasattr(wcs, 'pixel_to_world_values'):
                    # check which pixel axis (0=x, 1=y) this world axis varies most with
                    # look at the absolute values in the transformation matrix
                    if hasattr(wcs.wcs, 'pc'):
                        pc_matrix = wcs.wcs.pc
                    elif hasattr(wcs.wcs, 'cd'):
                        pc_matrix = wcs.wcs.cd
                    else:
                        pc_matrix = None

                    if pc_matrix is not None:
                        # for this world axis, see which pixel axis it affects most
                        world_axis_row = pc_matrix[coord_index, :]
                        dominant_pixel_axis = abs(world_axis_row).argmax()

                        # dominant_pixel_axis: 0 = x-axis (horizontal), 1 = y-axis (vertical)
                        if dominant_pixel_axis == 0:
                            coord.set_axislabel_position('b')
                            coord.set_ticklabel(rotation=0)
                        elif dominant_pixel_axis == 1:
                            coord.set_axislabel_position('l')
                            coord.set_ticklabel(rotation=90)

    _apply_plot_utils(
        plot_params, ax,
        im_list=image_list,
        ref_unit=ref_unit,
    )

    return image_list


def plot_spectral_cube(
    cubes: DataCube | SpectralCube | list[DataCube | SpectralCube],
    idx: int | tuple[int, int] | None | list[int | tuple[int, int] | None],
    ax: WCSAxes | maxes.Axes,
    vmin: float | _Unset = _UNSET,
    vmax: float | _Unset = _UNSET,
    norm: Literal['asinh', 'asinhnorm', 'log', 'power', 'twoslope', 'linear'] | None | _Unset = _UNSET,
    percentile: tuple[float, float] | None | _Unset = _UNSET,
    stack_method: Literal['mean', 'median', 'sum', 'max', 'min', 'std'] | _Unset = _UNSET,
    radial_vel: float | _Unset = _UNSET,
    spectral_unit: u.UnitBase | None = None,
    cmap: Colormap | str | list[Colormap | str] | _Unset = _UNSET,
    mask_non_pos: bool | _Unset = _UNSET,
    axis: int = 0,
    **kwargs
) -> list[AxesImage]:
    """
    Plot a spectral slice from one or more spectral cubes on a WCSAxes.

    Parameters
    ----------
    cubes : DataCube | SpectralCube | list of such
        One or more spectral cubes to plot. All cubes should have consistent units.
    idx : int | tuple[int, int] | None | list[int | tuple[int, int] | None]
        Spectral axis indices to slice. Tuples denote [start, end] (inclusive)
        and are collapsed via `stack_method`. If int, applies to all cubes; if list,
        each entry specifies indices for the corresponding cube. `None` collapses
        the entire spectral axis via `stack_method`.
    ax : WCSAxes | matplotlib.axes.Axes
        The axes on which to draw the slice.
    vmin, vmax: float | None | _Unset, optional, default=`_UNSET`
        Lower / upper limits for colormap scaling; overides `percentile`.
        If `None`, values are determined from `percentile[0]` for `vmin`
        and `percentile[1]` for `vmax`. If `_UNSET`, uses `config.vmin`
        and `config.vmax`.
    norm : str | None | _Unset, optional, default=_UNSET
        Normalization algorithm for colormap scaling.

        * `'asinh'` -> asinh stretch using `ImageNormalize`
        * `'asinhnorm'` -> asinh stretch using `AsinhNorm`
        * `'log'` -> logarithmic scaling using `LogNorm`
        * `'power'` -> power-law normalization using `PowerNorm`
        * `'twoslope'` -> twoslope normalization use `TwoSlopeNorm`
        * `'linear'`, `'none'`, or `None` -> no normalization applied

        If `_UNSET`, uses `config.norm`.
    percentile : tuple[float, float] | _Unset, default=_UNSET
        Default percentile range used to determine `vmin` and `vmax`.
        If `None`, use no percentile stretch (as long as `vmin` and
        `vmax`, are None). If `_UNSET`, uses `config.percentile`.
    stack_method : {'mean', 'median', 'sum', 'max', 'min', 'std'} | _Unset, default=_UNSET
        Stacking method. This controls how the cube is collapsed into a 2D
        image for plotting. The axis along to flatten is controlled by `axis`.
        If `_UNSET`, uses `config.stack_cube_method`.
    radial_vel : float | _Unset, optional, default=_UNSET
        Radial velocity in km/s to shift the spectral axis.
        Astropy units are optional. If `_UNSET`, uses `config.radial_velocity`.
    spectral_unit : u.UnitBase | str, optional, default=None
        Desired spectral axis unit for labeling. If `None`, detects unit automatically.
    cmap : Colormap | str | list[Colormap | str] | _Unset, default=_UNSET
        Colormap(s) to use for plotting. If `_UNSET`, uses `config.cmap`.
    mask_non_pos : bool | _Unset, optional, default=_UNSET
        If `True`, mask out non-positive data values. Useful for displaying
        log scaling of images with non-positive values. If `_UNSET`, uses
        `config.mask_non_positive`. The mask value is set by `mask_out_val`.
    axis : int, optional, default=0
        Axis used for collapsing 3D cubes into a 2D image.
    spectral_label : bool, optional, default=True
        If `True`, annotate the figure with the slice's spectral axis value,
        ie. the wavelength of the cube's spectral_axis.
    as_title : bool, optional, default=False
         If `True`, display spectral slice label as plot title.
    unit_fmt : str | optional, default=config.unit_label_format
        Format for the colorbar and spectral unit label. Accepted formats are
        `'latex'`, `'latex_inline'`, `'fits'`, `'unicode'`, `'console'`, `'vounit'`,
        `'cds'`, or `'ogip'`.
    emission_line : str | None, optional, default=None
        Optional emission line label prepended to the spectral axis label.
    rasterized : bool, optional, default=config.rasterized
        Whether to rasterize plot artists. Rasterization
        converts the artist to a bitmap when saving to
        vector formats (e.g., PDF, SVG), which can
        significantly reduce file size for complex plots.
    mask_out_val : float, optional, default=config.mask_out_value
        Value to use when masking out non-positive values.
        ie: np.nan, 1e-6, np.inf
    linear_width : float, optional, default=config.linear_width
        The effective width of the linear region, beyond which the
        transformation becomes asymptotically logarithmic.
        Used for `norm='asinhnorm'`.
    gamma : float, optional, default=config.gamma
        Power law exponent. Used for `norm='power'`.
    vcenter : float, optional, default=None
        Center point of normalization. Must be in between `vmin` and `vmax`.
        If `None`, is the midpoint between `vmin` and `vmax`.
    highlight : bool, optional, default=config.highlight
        Whether to highlight interactive ellipse or wavelength label if plotted.
    text_loc : list[float], optional, default=config.text_loc
        Axes coordinates for overlay text placement in figure coordinates.
    text_color : ColorType, default=config.text_color
        Color of overlay text.

    - `colorbar` : bool, default=`config.colorbar.enable`
        Whether to add a colorbar.
    - `cbar_width` : float, default=`config.colorbar.width`
        Width of the colorbar.
    - `cbar_pad` : float, default=`config.colorbar.pad`
        Padding between axes and colorbar.
    - `clabel` : str, bool, or None, default=`config.colorbar.label`
        Label for colorbar. If True, automatically generate from cube unit.
    - `xlabel` : str, default=`config.right_ascension_label`
        X axis label.
    - `ylabel` : str, default=`config.declination_label`
        Y axis label.
    - `ellipses` : list or None, default=None
        Ellipse objects to overlay on the image.


    Returns
    -------
    images : list[matplotlib.image.AxesImage]
        list of `AxesImage` objects returned by the `ax.imshow`
        call for each cube in `cubes`.

    Notes
    -----
    - If multiple cubes are provided, they are overplotted in sequence
    on the wcs of `cubes[config.reference_idx]`.
    """
    params = _resolve_kwargs(
        kwargs,
        params=[
            _param('vmin', vmin, config.vmin),
            _param('vmax', vmax, config.vmax),
            _param('norm', norm, config.norm),
            _param('percentile', percentile, config.percentile),
            _param('stack_method', stack_method, config.stack_cube_method),
            _param('radial_velocity', radial_vel, config.radial_velocity),
            _param('cmap', cmap, config.cmap),
            _param('mask_non_pos', mask_non_pos, config.mask_non_positive),
        ],
        additional_kwargs=[
            _kwarg('spectral_label', True),
            _kwarg('as_title', False),
            _kwarg('unit_fmt', config.unit_label_format),
            _kwarg('emission_line', None),
            _kwarg('rasterized', config.rasterized),
            _kwarg('mask_out_val', config.mask_out_value),
            _kwarg('linear_width', config.linear_width),
            _kwarg('gamma', config.gamma),
            _kwarg('vcenter', None),
        ],
        copy_kwargs=[
            _kwarg('plot_ellipse', False),
            _kwarg('highlight', config.highlight),
            _kwarg('text_loc', config.text_loc),
            _kwarg('text_color', config.text_color),
        ]
    )
    plot_params = _extract_plot_util_kwargs(kwargs)

    cubes = to_list(cubes)
    idxs = [idx] if isinstance(idx, list) else as_list(idx)
    cmaps = to_list(params.cmap)

    ref_unit = ensure_common_unit(cubes, on_mismatch=config.unit_mismatch)

    images = []

    for i, cube in enumerate(cubes):
        cube = get_data(cube)
        if not isinstance(cube, (SpectralCube, Slice)):
            raise ValueError(
                'Input cubes must contain a SpectralCube! '
                'For non SpectralCube data, use imshow.'
            )

        cube_slice = stack_cube(
            cube, idx=_cycle(idxs, i), method=stack_method, axis=axis
        )
        data = get_value(cube_slice)

        if mask_non_pos:
            data = np.where(data > 0.0, data, params.mask_out_val)

        cube_norm, cube_vmin, cube_vmax = compute_imshow_scale(
            data=data,
            norm=params.norm,
            vmin=params.vmin,
            vmax=params.vmax,
            percentile=params.percentile,
            linear_width=params.linear_width,
            gamma=params.gamma,
            vcenter=params.vcenter,
        )

        cm = get_cmap(_cycle(cmaps, i))

        imshow_kwargs = dict(kwargs)
        if norm is None:
            imshow_kwargs.pop('norm', None)
            imshow_kwargs.update(vmin=cube_vmin, vmax=cube_vmax)
        else:
            imshow_kwargs.pop('vmin', None)
            imshow_kwargs.pop('vmax', None)
            imshow_kwargs.update(norm=cube_norm)

        im = ax.imshow(
            data,
            origin='lower',
            cmap=cm,
            rasterized=params.rasterized,
            **imshow_kwargs
        )

        images.append(im)

    if params.plot_ellipse:
        params.spectral_label = False

    # plot wavelength/frequency of current spectral slice, and emission line
    if params.spectral_label:
        spectral_axis_label(
            _cycle(cubes, config.reference_idx),
            idx=_cycle(idxs, config.reference_idx),
            ax=ax,
            ref_unit=spectral_unit,
            radial_vel=params.radial_velocity,
            emission_line=params.emission_line,
            as_title=params.as_title,
            highlight=params.highlight,
            text_loc=params.text_loc,
            text_color=params.text_color,
            unit_fmt=params.unit_fmt
        )

    _apply_plot_utils(
        plot_params, ax,
        im_list=images,
        ref_unit=ref_unit,
    )

    return images
