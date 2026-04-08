"""
Author: Elko Gerville-Reache
Date Created: 2025-05-23
Date Modified: 2026-03-11
Description:
    Plotting functions for 2D and 3D images.
Dependencies:
    - astropy
    - matplotlib
    - numpy
    - spectral-cube
Module Structure:
    - Datacube I/O Functions
        Functions for loading datacubes into visualastro.
    - Cube Plotting Functions
        Functions for plotting datacubes
"""

import warnings
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization.wcsaxes.core import WCSAxes
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from spectral_cube import SpectralCube
from visualastro.analysis.image_utils import stack_cube
from visualastro.core.config import (
    get_config_value,
    config,
    _UNSET
)
from visualastro.core.numerical_utils import (
    get_data,
    get_value,
    to_array,
    to_list,
    _unwrap_if_single
)
from visualastro.core.units import (
    ensure_common_unit,
    get_unit,
    to_latex_unit
)
from visualastro.plotting.plot_utils import (
    add_colorbar,
    compute_imshow_scale,
    plot_circles,
    plot_ellipses,
    plot_interactive_ellipse,
    plot_points,
    get_imshow_norm,
    get_vmin_vmax,
)
from visualastro.plotting.spectra_plot_utils import (
    spectral_axis_label
)


warnings.filterwarnings('ignore', category=AstropyWarning)


def imshow(
    datas, ax, idx=None, vmin=_UNSET,
    vmax=_UNSET, norm=_UNSET,
    percentile=_UNSET, stack_method=None,
    origin=None, cmap=None, aspect=_UNSET,
    mask_non_pos=None, wcs_grid=None, **kwargs
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
        - `colorbar` : bool, optional, default=`config.cbar`
            Add colorbar if True.
        - `clabel` : str or bool, optional, default=`config.clabel`
            Colorbar label. If True, use default label; if None or False, no label.
        - `cbar_width` : float, optional, default=`config.cbar_width`
            Width of the colorbar.
        - `cbar_pad` : float, optional, default=`config.cbar_pad`
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
    # ---- KWARGS ----
    # figure params
    rasterized = kwargs.get('rasterized', config.rasterized)
    invert_xaxis = kwargs.get('invert_xaxis', False)
    invert_yaxis = kwargs.get('invert_yaxis', False)
    # labels
    text_loc = kwargs.get('text_loc', config.text_loc)
    text_color = kwargs.get('text_color', config.text_color)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    colorbar = kwargs.get('colorbar', config.cbar)
    clabel = kwargs.get('clabel', config.clabel)
    cbar_width = kwargs.get('cbar_width', config.cbar_width)
    cbar_pad = kwargs.get('cbar_pad', config.cbar_pad)
    # mask out value
    mask_out_val = kwargs.get('mask_out_val', config.mask_out_value)
    # plot objects
    circles = kwargs.get('circles', None)
    points = kwargs.get('points', None)
    # plot ellipse
    ellipses = kwargs.get('ellipses', None)
    plot_ellipse = kwargs.get('plot_ellipse', False)
    # default ellipse parameters
    datas = to_list(datas)
    data = to_array(datas[0])
    X, Y = data.shape[-2:]
    center = kwargs.get('center', [X//2, Y//2])
    w = kwargs.get('w', X//5)
    h = kwargs.get('h', Y//5)

    # get default config values
    vmin = config.vmin if vmin is _UNSET else vmin
    vmax = config.vmax if vmax is _UNSET else vmax
    norm = config.norm if norm is _UNSET else norm
    stack_method = get_config_value(stack_method, 'stack_cube_method')
    percentile = config.percentile if percentile is _UNSET else percentile
    origin = get_config_value(origin, 'origin')
    cmap = get_config_value(cmap, 'cmap')
    aspect = config.aspect if aspect is _UNSET else aspect
    mask_non_pos = get_config_value(mask_non_pos, 'mask_non_positive')
    wcs_grid = get_config_value(wcs_grid, 'wcs_grid')

    # ensure inputs are iterable or conform to standard
    ensure_common_unit(datas)
    cmap = cmap if isinstance(cmap, (list, np.ndarray, tuple)) else [cmap]
    if idx is not None:
        idx = idx if isinstance(idx, (list, np.ndarray, tuple)) else [idx]

    # if wcsaxes are used, origin can only be 'lower'
    if isinstance(ax, WCSAxes) and origin == 'upper':
        origin = 'lower'
        invert_yaxis = True

    images = []

    # loop over data list
    for i, data in enumerate(datas):
        # ensure data is an array
        data = to_array(data)
        # slice data with index if provided
        if idx is not None:
            data = stack_cube(
                data, idx=idx[i%len(idx)], method=stack_method, axis=0
            )

        if mask_non_pos:
            data = np.where(data > 0.0, data, mask_out_val)

        img_norm, vmin, vmax = compute_imshow_scale(
            data, norm, vmin, vmax, percentile, **kwargs
        )

        # imshow image
        if img_norm is None:
            im = ax.imshow(
                data, origin=origin, vmin=vmin,
                vmax=vmax, cmap=cmap[i%len(cmap)],
                aspect=aspect, rasterized=rasterized
            )
        else:
            im = ax.imshow(
                data,
                origin=origin,
                norm=img_norm,
                cmap=cmap[i%len(cmap)],
                aspect=aspect,
                rasterized=rasterized
            )

        images.append(im)

    # overplot
    plot_circles(circles, ax)
    plot_points(points, ax)
    plot_ellipses(ellipses, ax)
    if plot_ellipse:
        plot_interactive_ellipse(center, w, h, ax, text_loc, text_color)

    # invert axes
    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()

    # rotate tick labels and set optional grid
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
            wcs: WCS = ax.wcs
            if hasattr(wcs, 'pixel_to_world_values'):
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

        if wcs_grid:
            ax.coords.grid(True, color='white', ls='dotted')

    # set axes labels
    if isinstance(xlabel, str):
        ax.set_xlabel(xlabel)
    if isinstance(ylabel, str):
        ax.set_ylabel(ylabel)
    # add colorbar
    cbar_unit = to_latex_unit(get_unit(datas[0]))
    if clabel is True:
        clabel = cbar_unit if cbar_unit is not None else None
    if colorbar:
        add_colorbar(images[0], ax, cbar_width, cbar_pad, clabel)

    images = _unwrap_if_single(images)

    return images


def plot_spectral_cube(cubes, idx=None, ax=None, vmin=_UNSET,
                       vmax=_UNSET, norm=_UNSET,
                       percentile=_UNSET, stack_method=None,
                       radial_vel=None, spectral_unit=None, cmap=None,
                       mask_non_pos=None, wcs_grid=None, **kwargs):
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
        - 'asinh' -> asinh stretch using 'ImageNormalize'
        - 'asinhnorm' -> asinh stretch using 'AsinhNorm'
        - 'log' -> logarithmic scaling using 'LogNorm'
        - 'powernorm' -> power-law normalization using 'PowerNorm'
        - 'linear', 'none', or None -> no normalization applied
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
        uses the default value set by `config.cmap`.
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
        - `colorbar` : bool, default=`config.cbar`
            Whether to add a colorbar.
        - `cbar_width` : float, default=`config.cbar_width`
            Width of the colorbar.
        - `cbar_pad` : float, default=`config.cbar_pad`
            Padding between axes and colorbar.
        - `clabel` : str, bool, or None, default=`config.clabel`
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
    # check cube units match and ensure cubes is iterable
    cubes = to_list(cubes)
    ref_unit = ensure_common_unit(cubes)
    # ---- Kwargs ----
    # fig params
    rasterized = kwargs.get('rasterized', config.rasterized)
    as_title = kwargs.get('as_title', False)
    # labels
    emission_line = kwargs.pop('emission_line', None)
    text_loc = kwargs.get('text_loc', config.text_loc)
    text_color = kwargs.get('text_color', config.text_color)
    colorbar = kwargs.get('colorbar', config.cbar)
    cbar_width = kwargs.get('cbar_width', config.cbar_width)
    cbar_pad = kwargs.get('cbar_pad', config.cbar_pad)
    clabel = kwargs.get('clabel', config.clabel)
    xlabel = kwargs.get('xlabel', config.right_ascension)
    ylabel = kwargs.get('ylabel', config.declination)
    draw_spectral_label = kwargs.get('spectral_label', True)
    highlight = kwargs.get('highlight', config.highlight)
    # mask out value
    mask_out_val = kwargs.get('mask_out_val', config.mask_out_value)
    # plot ellipse
    ellipses = kwargs.get('ellipses', None)
    plot_ellipse = kwargs.get('plot_ellipse', False)
    _, X, Y = get_data(cubes[0]).shape
    center = kwargs.get('center', [X//2, Y//2])
    w = kwargs.get('w', X//5)
    h = kwargs.get('h', Y//5)
    angle = kwargs.get('angle', None)

    # get default config values
    vmin = config.vmin if vmin is _UNSET else vmin
    vmax = config.vmax if vmax is _UNSET else vmax
    norm = config.norm if norm is _UNSET else norm
    percentile = config.percentile if percentile is _UNSET else percentile
    stack_method = get_config_value(stack_method, 'stack_cube_method')
    radial_vel = get_config_value(radial_vel, 'radial_velocity')
    cmap = get_config_value(cmap, 'cmap')
    mask_non_pos = get_config_value(mask_non_pos, 'mask_non_positive')
    wcs_grid = get_config_value(wcs_grid, 'wcs_grid')

    if not isinstance(ax, WCSAxes):
        raise ValueError(
            'ax must be a WCSAxes instance!'
        )

    images = []
    cmap = cmap if isinstance(cmap, (list, np.ndarray, tuple)) else [cmap]

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
            data = np.where(data > 0.0, data, mask_out_val)

        cube_norm, vmin, vmax = compute_imshow_scale(
            data, norm, vmin, vmax, percentile, **kwargs
        )

        # imshow data
        if norm is None:
            im = ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax,
                           cmap=cmap[i%len(cmap)], rasterized=rasterized)
        else:
            im = ax.imshow(data, origin='lower', cmap=cmap[i%len(cmap)],
                           norm=cube_norm, rasterized=rasterized)

        images.append(im)

    # determine unit of colorbar
    cbar_unit = to_latex_unit(ref_unit)
    # set colorbar label
    if clabel is True:
        clabel = cbar_unit if cbar_unit is not None else None
    # set colorbar
    if colorbar:
        add_colorbar(
            images[0], ax, cbar_width, cbar_pad, clabel, rasterized=rasterized
        )

    if ellipses is not None:
        plot_ellipses(ellipses, ax)

    if plot_ellipse:
        plot_interactive_ellipse(
            center, w, h, ax, text_loc,
            text_color, highlight,
            rotation_step=kwargs.get('rotation_step', 5)
        )
        draw_spectral_label = False

    # plot wavelength/frequency of current spectral slice, and emission line
    if draw_spectral_label:
        spectral_axis_label(
            cubes[0], idx, ax,
            ref_unit=spectral_unit,
            radial_vel=radial_vel,
            emission_line=emission_line,
            as_title=as_title,
            **kwargs
        )

    # set axes labels
    ax.coords['ra'].set_axislabel(xlabel)
    ax.coords['dec'].set_axislabel(ylabel)
    ax.coords['dec'].set_ticklabel(rotation=90)
    if wcs_grid:
        ax.coords.grid(
            True,
            color=config.wcs_grid_color,
            alpha=config.grid_alpha,
            ls=config.wcs_grid_linestyle
        )

    images = _unwrap_if_single(images)

    return images
