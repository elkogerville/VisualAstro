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
from spectral_cube import SpectralCube

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
from visualastro.plotting.core.colors import get_cmap
from visualastro.plotting.core.interface import (
    _apply_plot_utils, _extract_plot_util_kwargs
)
from visualastro.plotting.core.plot_utils import compute_imshow_scale
from visualastro.plotting.science.spectra_plot_utils import spectral_axis_label


warnings.filterwarnings('ignore', category=AstropyWarning)


def imshow(
    images,
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
):
    """
    Display 2D image data with optional overlays and customization.

    Parameters
    ----------
    datas : np.ndarray or list of np.ndarray
        Image array or list of image arrays to plot. Each array should
        be 2D (Ny, Nx) or 3D (Nz, Nx, Ny) if using 'idx' to slice a cube.
    ax : matplotlib.axes.Axes or WCSAxes
        Matplotlib axis on which to plot the image(s).
    idx : int or list of int, optional, default=None
        Index for slicing along the first axis if 'datas'
        contains a cube.
        - i -> returns cube[i]
        - [i] -> returns cube[i]
        - [i, j] -> returns the sum of cube[i:j+1] along axis 0
        If 'datas' is a list of cubes, you may also pass a list of
        indeces.
        ex: passing indeces for 2 cubes-> [[i,j], k].
    vmin : float or None, optional, default=`_UNSET`
        Lower limit for colormap scaling; overides `percentile[0]`.
        If None, values are determined from `percentile[0]`.
        If `_UNSET`, uses the default value in `config.vmin`.
    vmax : float or None, optional, default=`_UNSET`
        Upper limit for colormap scaling; overides `percentile[1]`.
        If None, values are determined from `percentile[1]`.
        If `_UNSET`, uses the default value in `config.vmax`.
    norm : str or None, optional, default=`_UNSET`
        Normalization algorithm for colormap scaling.
        - 'asinh' -> asinh stretch using 'ImageNormalize'
        - 'asinhnorm' -> asinh stretch using 'AsinhNorm'
        - 'log' -> logarithmic scaling using 'LogNorm'
        - 'powernorm' -> power-law normalization using 'PowerNorm'
        - 'linear', 'none', or None -> no normalization applied
        If `_UNSET`, uses the default value in `config.norm`.
    percentile : list or tuple of two floats, or None, default=`_UNSET`
        Default percentile range used to determine 'vmin' and 'vmax'.
        If `_UNSET`, uses default value from `config.percentile`.
        If None, use no percentile stretch.
    stack_method : {'mean', 'median', 'sum', 'max', 'min', 'std'}, default=None
        Stacking method. If None, uses the default value set
        by ``config.stack_cube_method``.
    origin : {'upper', 'lower'} or None, default=None
        Pixel origin convention for imshow. If None,
        uses the default value from `config.origin`.
    cmap : str, list of str or None, default=None
        Matplotlib colormap name or list of colormaps, cycled across images.
        If None, uses the default value from `config.cmap`.
        ex: ['turbo', 'RdPu_r']
    aspect : {'auto', 'equal'}, float, or None, optional, default=`_UNSET`
        Aspect ratio passed to imshow, shortcut for `Axes.set_aspect`. 'auto'
        results in fixed axes with the aspect adjusted to fit the axes. 'equal`
        sets an aspect ratio of 1. None defaults to 'equal', however, if the
        image uses a transform that does not contain the axes data transform,
        then None means to not modify the axes aspect at all. If `_UNSET`,
        uses the default value from `config.aspect`.
    mask_non_pos : bool or None, optional, default=None
        If True, mask out non-positive data values. Useful for displaying
        log scaling of images with non-positive values. If None, uses the
        default value set by `config.mask_non_positive`.
    wcs_grid : bool or None, optional, default=None
        If True, display WCS grid ontop of plot. Requires
        using WCSAxes for `ax`. If None, uses the default
        value set by `config.wcs_grid`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `rasterized` : bool, default=`config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `invert_xaxis` : bool, optional, default=False
            Invert the x-axis if True.
        - `invert_yaxis` : bool, optional, default=False
            Invert the y-axis if True.
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
        - `center` : list of float, optional, default=[Nx//2, Ny//2]
            Center of the default interactive ellipse (x, y).
        - `w` : float, optional, default=X//5
            Width of the default interactive ellipse.
        - `h` : float, optional, default=Y//5
            Height of the default interactive ellipse.

    Returns
    -------
    images : matplotlib.image.AxesImage or list of matplotlib.image.AxesImage
            Image object if a single array is provided, otherwise a list of image
            objects created by `imshow`.
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
    idxs = as_list(idx)
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
        image = get_data(image)

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
                    coord.set_axislabel(config.right_ascension)
                elif coord_type == 'latitude':
                    coord.set_axislabel(config.declination)

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
    cubes,
    idx: int | tuple[int, int] | None = None,
    ax: WCSAxes | None = None,
    vmin: float | _Unset = _UNSET,
    vmax: float | _Unset = _UNSET,
    norm: Literal['asinh', 'asinhnorm', 'log', 'power', 'twoslope', 'linear'] | None | _Unset = _UNSET,
    percentile: tuple[float, float] | _Unset = _UNSET,
    stack_method=None,
    radial_vel: float | None = None,
    spectral_unit=None,
    cmap: Colormap | str | _Unset = _UNSET,
    mask_non_pos=None,
    **kwargs
) -> list[AxesImage]:
    """
    Plot a single spectral slice from one or more spectral cubes.

    Parameters
    ----------
    cubes : DataCube, SpectralCube, or list of such
        One or more spectral cubes to plot. All cubes should have consistent units.
    idx : int or None, optional, default=None
        Index along the spectral axis corresponding to the slice to plot.
        If None, collapses the entire cube into a 2D map according
        to ``stack_method``.
    ax : matplotlib.axes.Axes or WCSAxes
        The axes on which to draw the slice.
    vmin : float or None, optional, default=`_UNSET`
        Lower limit for colormap scaling; overides `percentile[0]`.
        If None, values are determined from `percentile[0]`.
        If `_UNSET`, uses the default value in `config.vmin`.
    vmax : float or None, optional, default=`_UNSET`
        Upper limit for colormap scaling; overides `percentile[1]`.
        If None, values are determined from `percentile[1]`.
        If `_UNSET`, uses the default value in `config.vmax`.
    norm : str or None, optional, default=`_UNSET`
        Normalization algorithm for colormap scaling.

        * `'asinh'` -> asinh stretch using `ImageNormalize`
        * `'asinhnorm'` -> asinh stretch using `AsinhNorm`
        * `'log'` -> logarithmic scaling using `LogNorm`
        * `'powernorm'` -> power-law normalization using `PowerNorm`
        * `'linear'`, `'none'`, or `None` -> no normalization applied

        If `_UNSET`, uses the default value in `config.norm`.
    percentile : list or tuple of two floats, or None, default=`_UNSET`
        Default percentile range used to determine `vmin` and `vmax`.
        If None, use no percentile stretch (as long as vmin/vmax are None).
        If `_UNSET`, uses default value from `config.percentile`.
    stack_method : {'mean', 'median', 'sum', 'max', 'min', 'std'}, default=None
        Stacking method. If None, uses the default value set
        by ``config.stack_cube_method``.
    radial_vel : float or None, optional, default=None
        Radial velocity in km/s to shift the spectral axis.
        Astropy units are optional. If None, uses the default
        value set by `config.radial_velocity`.
    spectral_unit : astropy.units.Unit or str, optional, default=None
        Desired spectral axis unit for labeling.
    cmap : str, list or tuple of str, or None, default=None
        Colormap(s) to use for plotting. If None,
        uses `config.cmap`.
    mask_non_pos : bool or None, optional, default=None
        If True, mask out non-positive data values. Useful for displaying
        log scaling of images with non-positive values. If None, uses the
        default value set by `config.mask_non_positive`.
    wcs_grid : bool or None, optional, default=None
        If True, display WCS grid ontop of plot. Requires
        using WCSAxes for `ax`. If None, uses the default
        value set by `config.wcs_grid`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `rasterized` : bool, default=`config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `title` : bool, default=False
            If True, display spectral slice label as plot title.
        - `emission_line` : str or None, default=None
            Optional emission line label to display instead of slice value.
        - `text_loc` : list of float, default=`config.text_loc`
            Relative axes coordinates for overlay text placement.
        - `text_color` : str, default=`config.text_color`
            Color of overlay text.
        - `colorbar` : bool, default=`config.colorbar.enable`
            Whether to add a colorbar.
        - `cbar_width` : float, default=`config.colorbar.width`
            Width of the colorbar.
        - `cbar_pad` : float, default=`config.colorbar.pad`
            Padding between axes and colorbar.
        - `clabel` : str, bool, or None, default=`config.colorbar.label`
            Label for colorbar. If True, automatically generate from cube unit.
        - `xlabel` : str, default=`config.right_ascension`
            X axis label.
        - `ylabel` : str, default=`config.declination`
            Y axis label.
        - `spectral_label` : bool, default=True
            Whether to draw spectral slice value as a label.
        - `highlight` : bool, optional, default=`config.highlight`
            Whether to highlight interactive ellipse or wavelength label if plotted.
        - `mask_out_val` : float, optional, default=`config.mask_out_value`
            Value to use when masking out non-positive values.
            Ex: np.nan, 1e-6, np.inf
        - `ellipses` : list or None, default=None
            Ellipse objects to overlay on the image.
        - `plot_ellipse` : bool, default=False
            If True, plot a default or interactive ellipse.
        - `center` : list of two ints, default=[Nx//2, Ny//2]
            Center of default ellipse.
        - `w`, `h` : float, default=X//5, Y//5
            Width and height of default ellipse.
        - `angle` : float or None, default=None
            Angle of ellipse in degrees.

    Returns
    -------
    images : matplotlib.image.AxesImage or list of matplotlib.image.AxesImage
            Image object if a single array is provided, otherwise a list of image
            objects created by `ax.imshow`.

    Notes
    -----
    - If multiple cubes are provided, they are overplotted in sequence.
    """
    params = _resolve_kwargs(
        kwargs,
        params=[
            _param('vmin', vmin, config.vmin),
            _param('vmax', vmax, config.vmax),
            _param('norm', norm, config.norm),
            _param('percentile', percentile, config.percentile),
            _param('cmap', cmap, config.cmap),
        ],
        additional_kwargs=[
            _kwarg('as_title', False),
            _kwarg('unit_fmt', config.unit_label_format),
            _kwarg('emission_line', None),
            _kwarg('rasterized', config.rasterized),
            _kwarg('spectral_label', True),
            _kwarg('mask_out_val', config.mask_out_value),
            _kwarg('stack_method', config.stack_cube_method),
            _kwarg('radial_velocity', config.radial_velocity),
            _kwarg('mask_non_pos', config.mask_non_positive),
            _kwarg('linear_width', config.linear_width),
            _kwarg('gamma', config.gamma),
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
    cmaps = to_list(params.cmap)

    ref_unit = ensure_common_unit(cubes, on_mismatch=config.unit_mismatch)

    if not isinstance(ax, WCSAxes) or ax is None:
        raise ValueError(
            'ax must be a WCSAxes instance!'
        )

    images = []

    for i, cube in enumerate(cubes):
        cube = get_data(cube)
        if not isinstance(cube, SpectralCube):
            raise ValueError(
                'Input cubes must contain a SpectralCube! '
                'For non SpectralCube data, use imshow.'
            )

        # return data cube slices
        cube_slice = stack_cube(
            cube, idx=idx, method=stack_method, axis=0
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
        )

        cm = get_cmap(_cycle(cmaps, i))

        imshow_kwargs = dict(kwargs)
        if norm is None:
            imshow_kwargs.pop('norm', None)
            imshow_kwargs.update(vmin=cube_vmin, vmax=cube_vmax)
        else:
            imshow_kwargs.pop('vmin', None)
            imshow_kwargs.pop('vmax', None)
            imshow_kwargs['norm'] = cube_norm

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
            cubes[0], idx, ax,
            ref_unit=spectral_unit,
            radial_vel=radial_vel,
            emission_line=params.emission_line,
            as_title=params.as_title,
            highlight=params.highlight,
            text_loc=params.text_loc,
            text_color=params.text_color,
        )

    _apply_plot_utils(
        plot_params, ax,
        im_list=images,
        ref_unit=ref_unit,
        colorbar=True
    )

    return images
