import os
from functools import partial
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.visualization.wcsaxes.core import WCSAxes
from astropy import units as u
from astropy.units import spectral
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
from regions import PixCoord, EllipsePixelRegion
from multiprocessing.dummy import dict

# ––––––––––––––––––
# Plotting Functions
# ––––––––––––––––––
def imshow(datas, ax, idx=None, vmin=None, vmax=None, norm=None,
           percentile=[3,99.5], origin='lower', cmap='turbo',
           plot_boolean=False, aspect=None, **kwargs):
    '''
    Display 2D image data with optional overlays and customization.
    Parameters
    ––––––––––
    datas : np.ndarray or list of np.ndarray
        Image array or list of image arrays to plot. Each array should
        be 2D (Ny, Nx) or 3D (Nx, Ny, Nz) if using 'idx' to slice a cube.
    ax : matplotlib.axes.Axes or WCSAxes
        Matplotlib axis on which to plot the image(s).
    idx : int or list of int, optional
        Index for slicing along the first axis if 'datas'
        contains a cube.

        - i -> returns cube[i]
        - [i] -> returns cube[i]
        - [i, j] -> returns the sum of cube[i:j+1] along axis 0

        If 'datas' is a list of cubes, you may also pass a list of
        indeces.
        ex: passing indeces for 2 cubes-> [[i,j], k].
    vmin, vmax : float, optional
        Lower and upper limits for colormap scaling. If not provided,
        values are determined from 'percentile'.
    norm : str, optional
        Normalization algorithm for colormap scaling.
        - 'asinh' -> AsinhStretch using 'ImageNormalize'
        - 'log' -> logarithmic scaling using 'LogNorm'
        - 'none' or None -> no normalization applied

    percentile : list of float, default [3, 99.5]
        Default percentile range used to determine 'vmin' and 'vmax'.
    origin : str, {'upper', 'lower'}, default 'lower'
        Pixel origin convention for imshow.
    cmap : str or list of str, default 'turbo'
        Matplotlib colormap name or list of colormaps, cycled across images.
        ex: ['turbo', 'RdPu_r']
    plot_boolean : bool, default False
        If True, assumes 'datas' contain boolean arrays and fixes
        colormap scaling between 0 and 1, with norm = None.
    aspect : str, {'auto', 'equal'} or float, optional
        Aspect ratio passed to imshow. By default is None.

    Kwargs
    ––––––
    invert_xaxis : bool, default False
        Invert the x-axis if True.
    invert_yaxis : bool, default False
        Invert the y-axis if True.
    text_loc : list of float, default [0.03, 0.03]
        Relative axes coordinates for text placement when plotting interactive ellipses.
    text_color : str, default 'k'
        Color of the ellipse annotation text.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    colorbar : bool, default True
        Add colobar if True.
    clabel : str, optional
        Colorbar label.
    cbar_width : float, default 0.03
        Width of the colorbar.
    cbar_pad : float, default 0.015
        Padding between plot and colorbar.
    rotate_tick_axis : str, {'ra', 'dec'}, optional
        Coordinate axis name whose tick labels should be rotated
        by 90 degrees. Only applies if 'ax' is a WCSAxes.
    circles : list, optional
        List of circle objects (e.g., matplotlib.patches.Circle)
        to overplot on the axes.
    points : array-like, shape (2,) or (N, 2), optional
        Coordinates of points to overplot. Can be a single point '[x, y]'
        or a list/array of points '[[x1, y1], [x2, y2], ...]'.
        Points are plotted as red stars by default.
    ellipses : list, optional
        List of Ellipse objects (e.g., matplotlib.patches.Ellipse) to
        overplot on the axes. Single ellipses can also be passed directly.
    plot_ellipse : bool, default False
        If True, plot an interactive ellipse overlay.
        Ensure you are using an interactive backend such as
        use_interactive() for this to work.
    center : list of float, default [X, Y]
        Center of the default interactive ellipse (x, y).
    w : float, default X//5
        Width of the default interactive ellipse.
    h : float, default Y//5
        Height of the default interactive ellipse.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The last image plotted (if multiple images are given, the final one).
    '''
    # –––– KWARGS ––––
    # figure params
    invert_xaxis = kwargs.get('invert_xaxis', False)
    invert_yaxis = kwargs.get('invert_yaxis', False)
    # labels
    text_loc = kwargs.get('text_loc', [0.03,0.03])
    text_color = kwargs.get('text_color', 'k')
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    colorbar = kwargs.get('colorbar', True)
    clabel = kwargs.get('clabel', None)
    cbar_width = kwargs.get('cbar_width', 0.03)
    cbar_pad = kwargs.get('cbar_pad', 0.015)
    rotate_tick_axis = kwargs.get('rotate_tick_axis', None)
    # plot objects
    circles = kwargs.get('circles', None)
    points = kwargs.get('points', None)
    # plot ellipse
    ellipses = kwargs.get('ellipses', None)
    plot_ellipse = kwargs.get('plot_ellipse', False)
    # default ellipse parameters
    X, Y = datas[0].shape if isinstance(datas, list) else datas.shape
    center = kwargs.get('center', [X//2, Y//2])
    w = kwargs.get('w', X//5)
    h = kwargs.get('h', Y//5)
    # settings to plot boolean array
    if plot_boolean:
        vmin = 0
        vmax = 1
        norm = None
    # ensure inputs are iterable or conform to standard
    datas = datas if isinstance(datas, list) else [datas]
    cmap = cmap if isinstance(cmap, list) else [cmap]
    idx = idx if isinstance(cmap, list) else [idx]

    # loop over data list
    for i, data in enumerate(datas):
        # ensure data is an array
        data = check_is_array(data)
        # slice data with index if provided
        if idx is not None:
            data = return_cube_slice(data, idx[i%len(idx)])
        # set imshow norm and scaling if not plotting boolean
        if not plot_boolean:
            vmin, vmax = set_vmin_vmax(data, percentile, vmin, vmax)
            img_norm = return_imshow_norm(vmin, vmax, norm)

        # imshow image
        if norm is None:
            im = ax.imshow(data, origin=origin, vmin=vmin, vmax=vmax,
                           cmap=cmap[i%len(cmap)], aspect=aspect)
        else:
            im = ax.imshow(data, origin=origin, norm=img_norm,
                           cmap=cmap[i%len(cmap)], aspect=aspect)

    # overplot
    plot_circles(circles, ax)
    plot_points(points, ax)
    plot_ellipses(ellipses, ax)
    if plot_ellipse:
        plot_interactive_ellipse(center, w, h, ax, text_loc, text_color)

    # rotate tick labels
    if isinstance(ax, WCSAxes) and (rotate_tick_axis is not None):
        ax.coords[rotate_tick_axis].set_ticklabel(rotation=90)
    # set axes labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # add colorbar
    if colorbar:
        add_colorbar(im, ax, cbar_width, cbar_pad, clabel)
    # invert axes
    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()

def plot_histogram(data, bins='auto', style='astro', xlog=False, ylog=False,
                   colors=None, labels=None, savefig=False, dpi=600):
    data = check_is_array(data)
    colors, _ = set_plot_colors(colors)
    if data.ndim == 2:
        data = data.flatten()
    style = return_stylename(style)
    with plt.style.context(style):
        plt.figure(figsize=(5,5))
        plt.hist(data, bins=bins, color=colors[0], histtype='step')

        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')

        if labels is not None and len(labels) >= 2:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        else:
            plt.xlabel('')
            plt.ylabel('')
        if savefig:
                save_figure_2_disk(dpi)
        plt.show()

def plot_timeseries(time, data, normalize=False, xlabel=None, ylabel=None, style='astro', colors=None, figsize=(6,6)):
    if isinstance(data, np.ndarray) and data.ndim == 1:
        data = [data]
    colors, _ = set_plot_colors(colors)
    style = return_stylename(style)
    with plt.style.context(style):
        plt.figure(figsize=figsize)
        for i in range(len(data)):
            y = data[i]
            if normalize:
                y=y/np.max(y)
            plt.scatter(time, y, s=1, c=colors[i%len(colors)])

        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.show()
# ––––––––––––––
# Plotting Utils
# ––––––––––––––
def check_is_array(cube):
    '''
    Ensure array input is np.ndarray.
    Parameters
    ––––––––––
    cube : np.ndarray or dict
    Returns
    –––––––
    cube : np.ndarray
    '''
    if isinstance(cube, dict):
        cube = np.asarray(cube['data'])
    else:
        cube = np.asarray(cube)

    return cube

def shift_by_radial_vel(spectral_axis, radial_vel):
    '''
    Shift spectral axis by a radial velocity.
    Parameters
    ––––––––––
    spectral_axis : astropy.units.Quantity
        The spectral axis to shift. Can be in frequency or wavelength units.
    radial_vel : float or None
        Radial velocity in km/s (astropy units are not needed). Positive values
        correspond to a redshift (moving away). If None, no shift is applied.
    Returns
    –––––––
    shifted_axis : astropy.units.Quantity
        The spectral axis shifted according to the given radial velocity.
        If the input is in frequency units, the relativistic Doppler
        formula for frequency is applied; otherwise, the formula for
        wavelength is applied.
    '''
    # speed of light in km/s in vacuum
    c = 299792.458 # [km/s]
    if radial_vel is not None:
        # if spectral axis in units of frequency
        if spectral_axis.unit.is_equivalent(u.Hz):
            spectral_axis /= (1 - radial_vel / c)
        # if spectral axis in units of wavelength
        else:
            spectral_axis /= (1 + radial_vel / c)

    return spectral_axis

def return_stylename(style):
    '''
    Returns the path to a visualastro mpl stylesheet for
    consistent plotting parameters. Matplotlib styles are
    also available (ex: 'solaris').

    To add custom user defined mpl sheets, add files in:
    VisualAstro/visualastro/stylelib/
    Ensure the stylesheet follows the naming convention:
        mystylesheet.mplstyle

    Parameters
    ––––––––––
    style : str
        name of the mpl stylesheet without the extension
        ex: 'astro'
    Returns
    –––––––
    style_path : str
        path to matplotlib stylesheet
    '''
    # if style is a default matplotlib stylesheet
    if style in mpl.style.available:
        return style
    # if style is a visualastro stylesheet
    else:
        style = style + '.mplstyle'
        dir_path = os.path.dirname(os.path.realpath(__file__))
        style_path = os.path.join(dir_path, 'stylelib', style)
        return style_path

def set_vmin_vmax(data, percentile, vmin, vmax):
    '''
    Compute vmin and vmax for image display, optionally using percentiles.
    Parameters
    ––––––––––
    data : array-like
        Input data array (e.g., 2D image) for which to compute vmin and vmax.
    percentile : list or tuple of two floats, optional
        Percentile range '[pmin, pmax]' to compute vmin and vmax.
        If None, sets vmin and vmax to None.
    vmin : float or None
        If provided, overrides the computed vmin.
    vmax : float or None
        If provided, overrides the computed vmax.
    Returns
    –––––––
    vmin : float or None
        Minimum value for image scaling.
    vmax : float or None
        Maximum value for image scaling.
    '''
    # by default use percentile range. if vmin or vmax is provided
    # overide and use those instead
    if percentile is not None:
        vmin = np.nanpercentile(data, percentile[0]) if vmin is None else vmin
        vmax = np.nanpercentile(data, percentile[1]) if vmax is None else vmax
    # if percentile is None return None for vmin and vmax
    else:
        vmin = None
        vmax = None

    return vmin, vmax

def return_imshow_norm(vmin, vmax, norm):
    '''
    Return a matplotlib or astropy normalization object for image display.
    Parameters
    ––––––––––
    vmin : float or None
        Minimum value for normalization.
    vmax : float or None
        Maximum value for normalization.
    norm : str or None
        Normalization algorithm for colormap scaling.
        - 'asinh' -> AsinhStretch using 'ImageNormalize'
        - 'log' -> logarithmic scaling using 'LogNorm'
        - 'none' or None -> no normalization applied
    Returns
    –––––––
    norm_obj : None or matplotlib.colors.Normalize or astropy.visualization.ImageNormalize
        Normalization object to pass to `imshow`. None if `norm` is 'none'.
    '''
    # ensure norm is a string
    norm = 'none' if norm is None else norm
    # ensure case insensitivity
    norm = norm.lower()
    # dict containing possible stretch algorithms
    norm_map = {
        'asinh': ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch()),
        'log': LogNorm(vmin=vmin, vmax=vmax),
        'none': None
    }
    if norm not in norm_map:
        raise ValueError(f"ERROR: unsupported norm: {norm}")

    return norm_map[norm]

def return_cube_slice(cube, idx):
    '''
    Return a slice of a data cube along the first axis.
    Parameters
    ––––––––––
    cube : np.ndarray
        Input data cube, typically with shape (T, N, ...) where T is the first axis.
    idx : int or list of int
        Index or indices specifying the slice along the first axis:
        - i -> returns 'cube[i]'
        - [i] -> returns 'cube[i[0]]'
        - [i, j] -> returns 'cube[i:j+1].sum(axis=0)'
    Returns
    –––––––
    cube : np.ndarray
        Sliced cube with shape (N, ...).
    '''
    # if index is integer
    if isinstance(idx, int):
        return cube[idx]
    # if index is list of integers
    elif isinstance(idx, list):
        # list of len 1
        if len(idx) == 1:
            return cube[idx[0]]
        # list of len 2
        elif len(idx) == 2:
            start, end = idx
            return cube[start:end+1].sum(axis=0)
    raise ValueError("'idx' must be an int or a list of one or two integers")

def set_spectral_axis(cube, unit=None):
    axis = cube.spectral_axis
    if unit is None:
        return axis
    try:
        return axis.to(unit, equivalencies=spectral())
    except u.UnitConversionError:
        raise ValueError(f"Cannot convert spectral axis from {axis.unit} to {unit}")

def return_spectral_axis_idx(spectral_axis, idx):
    if isinstance(idx, int):
        return spectral_axis[idx].value
    elif isinstance(idx, list):
        if len(idx) == 1:
            return spectral_axis[idx[0]].value
        elif len(idx) == 2:
            start, end = idx
            return (spectral_axis[idx[0]].value + spectral_axis[idx[1]+1].value)/2
    raise ValueError("'idx' must be an int or a list of one or two integers")

def save_figure_2_disk(dpi=600):
    '''
    Saves current figure to disk as a pdf, png, or svg,
    and prompts user for a filename and format.
    Parameters
    ----------
    dpi: float
        resolution in dots per inch
    '''
    allowed_formats = {'pdf', 'png', 'svg'}
    # prompt user for filename, and extract extension
    filename = input('Input filename for image (ex: myimage.pdf): ').strip()
    basename, *extension = filename.rsplit('.', 1)
    # if extension exists, and is allowed, extract extension from list
    if extension and extension[0].lower() in allowed_formats:
        extension = extension[0]
    # else prompt user to input a valid extension
    else:
        extension = ''
        while extension not in allowed_formats:
            extension = input(f'Please choose a format from ({", ".join(allowed_formats)}): ').strip().lower()
    # construct complete filename
    filename = f'{basename}.{extension}'

    # save figure
    plt.savefig(filename, format=extension, bbox_inches='tight', dpi=dpi)

def set_plot_colors(user_colors=None):
    default_color_map = 'ibm_contrast'
    color_map = {
        #                  dsb        mvr      ibmblue      gold     mossgreen
        'visualastro': ['#483D8B', '#DC267F', '#648FFF', '#FFB000', '#26DCBA'],
        #                ibmblue       mvr     ibmpurple  mossgreen    gold      traffico
        'ibm_contrast': ['#648FFF', '#DC267F', '#785EF0', '#26DCBA', '#FFB000', '#FE6100'],
        #          bbblue     ibmblue   ibmpurple     mvr      traffico    gold      pondwater  mossgreen
        'astro': ['#9FB7FF', '#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', '#CFE23C', '#26DCBA'],
        #          mossgreen    pondwater    gold      traffico    mvr      ibmpurple   ibmblue    bbblue
        'astro_r': ['#26DCBA', '#CFE23C', '#FFB000', '#FE6100', '#DC267F', '#785EF0', '#648FFF', '#9FB7FF'],
        #          dsb        mvr       lilac    mossgreen  slateblue
        'MSG': ['#483D8B', '#DC267F', '#DBB0FF', '#26DCBA', '#7D7FF3'],
        #        ibmblue   ibmpurple     mvr      traffico    gold
        'ibm': ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
    }
    model_colors = ['r', 'purple', 'magenta']
    if user_colors is not None:
        if isinstance(user_colors, str) and user_colors in color_map:
            colors = color_map[user_colors]
        else:
            if isinstance(user_colors, str):
                if user_colors == 'mvr':
                    user_colors = ['#DC267F']
                else:
                    user_colors = [user_colors]
            colors = user_colors
    else:
        colors = color_map[default_color_map]

    return colors, model_colors

def set_unit_labels(unit):
    unit_label = {
        'MJy / sr': r'\mathrm{MJy\ sr^{-1}}',
        'Jy / beam': r'\mathrm{Jy\ beam^{-1}}',
        'micron': r'\mathrm{\mu m}',
        'um': r'\mathrm{\mu m}',
        'nm': 'nm',
        'nanometer': 'nm',
        'Angstrom': r'\mathrm{\AA}',
        'm': 'm',
        'meter': 'm',
        'Hz': 'Hz',
        'kHz': 'kHz',
        'MHz': 'MHz',
        'GHz': 'GHz',
        'electron': r'\mathrm{e^{-}}',
        'km / s': r'\mathrm{km\ s^{-1}}',
    }.get(str(unit), unit)

    return unit_label

def set_axis_labels(X, Y, ax, xlabel=None, ylabel=None, use_brackets=False):
    spectral_type = {
        'frequency': 'Frequency',
        'length': 'Wavelength',
        'speed/velocity': 'Velocity',
    }.get(str(X.unit.physical_type), 'Spectral Axis')

    brackets = [r'[$',r'$]'] if use_brackets else [r'($',r'$)']

    if xlabel is None:
        x_unit = str(getattr(X, 'spectral_unit', getattr(X, 'unit', None)))
        x_unit_label = fr'{brackets[0]}{set_unit_labels(x_unit)}{brackets[1]}'
        xlabel = fr'{spectral_type} {x_unit_label}' if x_unit_label else spectral_type

    if ylabel is None:
        y_unit = str(getattr(Y, 'spectral_unit', getattr(Y, 'unit', None)))
        y_unit_label = fr'{brackets[0]}{set_unit_labels(y_unit)}{brackets[1]}'
        ylabel = fr'Flux {y_unit_label}' if y_unit_label else 'Flux'

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def add_colorbar(im, ax, cbar_width, cbar_pad, clabel):
    fig = ax.figure
    cax = fig.add_axes([ax.get_position().x1+cbar_pad, ax.get_position().y0,
                        cbar_width, ax.get_position().height])
    cbar = fig.colorbar(im, cax=cax, pad=0.04)
    cbar.ax.tick_params(which='both', direction='out')
    if clabel is not None:
        cbar.set_label(fr'{clabel}')

def plot_circles(circles, ax):
    circle_colors = ['r', 'mediumvioletred', 'magenta']

    if circles is not None:
        circles = np.asarray(circles)
        if circles.ndim == 1 and circles.shape[0] == 3:
            circles = circles[np.newaxis, :]
        elif circles.ndim != 2 or circles.shape[1] != 3:
            error = 'Circles must be either [x, y, r] or [[x1, y1, r1], [x2, y2, r2], ...]'

            raise ValueError(error)

        for i, circle in enumerate(circles):
            x, y, r = circle
            circle_patch = Circle((x, y), radius=r, fill=False, linewidth=2,
                                  color=circle_colors[i%len(circle_colors)])
            ax.add_patch(circle_patch)

def plot_ellipses(ellipses, ax):
    if ellipses is not None:
        ellipses = ellipses if isinstance(ellipses, list) else [ellipses]
        for ellipse in ellipses:
            ax.add_patch(copy_ellipse(ellipse))

def copy_ellipse(ellipse):
    return Ellipse(
        xy=ellipse.center,
        width=ellipse.width,
        height=ellipse.height,
        angle=ellipse.angle,
        edgecolor=ellipse.get_edgecolor(),
        facecolor=ellipse.get_facecolor(),
        lw=ellipse.get_linewidth(),
        ls=ellipse.get_linestyle(),
        alpha=ellipse.get_alpha()
    )

def plot_points(points, ax):
    if points is not None:
        points = np.asarray(points)
        if points.ndim == 1 and points.shape[0] == 2:
            points = points[np.newaxis, :]
        elif points.ndim != 2 or points.shape[1] != 2:
            error = 'Points must be either [x, y] or [[x1, y1], [x2, y2], ...]'

            raise ValueError(error)

        for point in points:
            ax.scatter(point[0], point[1], s=20, marker='*', c='r')

def plot_interactive_ellipse(center, w, h, ax, text_loc=[0.03,0.03], text_color='k'):
    text = ax.text(text_loc[0], text_loc[1], '', transform=ax.transAxes, size='small', color=text_color)
    ellipse_region = EllipsePixelRegion(center=PixCoord(x=center[0], y=center[1]), width=w, height=h)
    selector = ellipse_region.as_mpl_selector(ax, callback=partial(update_region, text=text))
    ax._ellipse_selector = selector

def update_region(region, text):
    x_center = region.center.x
    y_center = region.center.y
    width = region.width
    height = region.height
    major = max(width, height)
    minor = min(width, height)

    text.set_text(
        f'Center: [{x_center:.1f}, {y_center:.1f}]\n'
        f'Major: {major:.1f}\n'
        f'Minor: {minor:.1f}\n'
    )

# ––––––––––––––
# Notebook Utils
# ––––––––––––––
def use_inline():
    '''
    Start an inline IPython backend session.
    Allows for inline plots in IPython sessions
    like Jupyter Notebook.
    '''
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic("matplotlib", "inline")
        else:
            print("Not in an IPython environment.")
    except ImportError:
        print("IPython is not installed. Install it to use this feature.")

def use_interactive():
    '''
    Start an interactive IPython backend session.
    Allows for interactive plots in IPython sessions
    like Jupyter Notebook.
    '''
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic("matplotlib", "ipympl")
        else:
            print("Not in an IPython environment.")
    except ImportError:
        print("IPython is not installed. Install it to use this feature.")

def plt_close():
    '''
    Close all interactive plots in session.
    '''
    plt.close('all')
