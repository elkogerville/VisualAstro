import os
from functools import partial
from astropy.visualization import AsinhStretch, ImageNormalize
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib import colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
import numpy as np
from regions import PixCoord, EllipsePixelRegion
from .data_cube_utils import return_cube_slice
from .numerical_utils import check_is_array, return_array_values, return_cube_data

# Plot Style and Color Functions
# ––––––––––––––––––––––––––––––
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
        Name of the mpl stylesheet without the extension
        ex: 'astro'
    Returns
    –––––––
    style_path : str
        Path to matplotlib stylesheet
    '''
    # if style is a default matplotlib stylesheet
    if style in mplstyle.available:
        return style
    # if style is a visualastro stylesheet
    else:
        style = style + '.mplstyle'
        dir_path = os.path.dirname(os.path.realpath(__file__))
        style_path = os.path.join(dir_path, 'stylelib', style)
        return style_path

def lighten_color(color, mix=0.5):
    '''
    Lightens the given matplotlib color by mixing it with white.
    Parameters
    ––––––––––
    color : matplotlib color, str
        Matplotlib named color, hex color, html color or rgb tuple.
    mix : float or int
        Ratio of color to white in mix.
        mix=0 returns the original color,
        mix=1 returns pure white.
    '''
    # convert to rgb
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    # mix color with white
    mixed = (1 - mix) * rgb + mix * white

    return mcolors.to_hex(mixed)

def sample_cmap(N, cmap='turbo', return_hex=False):
    '''
    Sample N distinct colors from a given matplotlib colormap
    returned as RGBA tuples in an array of shape (N,4).
    Parameters
    ––––––––––
    N : int
        Number of colors to sample.
    cmap : str or Colormap, optional, default='turbo'
        Name of the matplotlib colormap.
    return_hex : bool, optional, default=False
        If True, return colors as hex strings.
    Returns
    –––––––
    list of tuple
        A list of RGBA colors sampled evenly from the colormap.
    '''
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, N))
    if return_hex:
        colors = np.array([mcolors.to_hex(c) for c in colors])

    return colors

def set_plot_colors(user_colors=None, cmap='turbo'):
    '''
    Returns plot and model colors based on predefined palettes or user input.
    Parameters
    ––––––––––
    user_colors : None, str, or list, optional, default=None
        - None: returns the default palette ('ibm_contrast').
        - str:
            * If the string matches a palette name, returns that palette.
            * If the string ends with '_r', returns the reversed version of the palette.
            * If the string is a single color (hex or matplotlib color name), returns
              that color and a lighter version for the model.
        - list:
            * A list of colors (hex or matplotlib color names). Returns the list
              for plotting and lighter versions for models.
        - int:
            * An integer specifying how many colors to sample from a matplolib cmap
              using sample_cmap(). By default uses 'turbo'.
    cmap : str or list of str, default='turbo'
        Matplotlib colormap name.
    Returns
    –––––––
    plot_colors : list of str
        Colors for plotting the data.
    model_colors : list of str
        Colors for plotting the model (contrasting or lighter versions).
    '''
    # default visualastro color palettes
    palettes = {
        'visualastro': {
            'plot':  ['#483D8B', '#DC267F', '#648FFF', '#FFB000', '#26DCBA'],
            'model': ['#D62728', '#1F77B4', '#E45756', '#17BECF', '#9467BD']
        },
        'ibm_contrast': {
            'plot':  ['#648FFF', '#DC267F', '#785EF0', '#26DCBA', '#FFB000', '#FE6100'],
            'model': ['#D62728', '#2CA02C', '#9467BD', '#17BECF', '#1F77B4', '#8C564B']
        },
        'astro': {
            'plot':  ['#9FB7FF', '#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', '#CFE23C', '#26DCBA'],
            'model': ['#D62728', '#1F77B4', '#9467BD', '#2CA02C', '#E45756', '#17BECF', '#8C564B', '#FFD700']
        },
        'MSG': {
            'plot':  ['#483D8B', '#DC267F', '#DBB0FF', '#26DCBA', '#7D7FF3'],
            'model': ['#D62728', '#1F77B4', '#2CA02C', '#9467BD', '#17BECF']
        },
        'ibm': {
            'plot':  ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'],
            'model': ['#D62728', '#2CA02C', '#9467BD', '#17BECF', '#E45756']
        }
    }

    default_palette = 'ibm_contrast'
    # default case
    if user_colors is None:
        palette = palettes[default_palette]
        return palette['plot'], palette['model']
    # if user passes a color string
    if isinstance(user_colors, str):
        # if palette in visualastro palettes
        # return a reversed palette if palette
        # ends with '_r'
        if user_colors.rstrip('_r') in palettes:
            base_name = user_colors.rstrip('_r')
            palette = palettes[base_name]
            plot_colors = palette['plot']
            model_colors = palette['model']
            # if '_r', reverse palette
            if user_colors.endswith('_r'):
                plot_colors = plot_colors[::-1]
                model_colors = model_colors[::-1]
            return plot_colors, model_colors
        else:
            return [user_colors], [lighten_color(user_colors)]
    # if user passes a list or array of colors
    if isinstance(user_colors, (list, np.ndarray)):
        return user_colors, [lighten_color(c) for c in user_colors]
    # if user passes an integer N, sample a cmap for N colors
    if isinstance(user_colors, int):
        colors = sample_cmap(user_colors, cmap=cmap)
        return colors, [lighten_color(c) for c in colors]
    raise ValueError(
        'user_colors must be None, a str palette name, a str color, a list of colors, or an integer'
    )

# Imshow Stretch Functions
# ––––––––––––––––––––––––
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
        'asinh': ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch()), # type: ignore
        'log': LogNorm(vmin=vmin, vmax=vmax),
        'none': None
    }
    if norm not in norm_map:
        raise ValueError(f"ERROR: unsupported norm: {norm}")
    # use linear stretch if plotting boolean array
    if vmin==0 and vmax==1:
        return None

    return norm_map[norm]

def set_vmin_vmax(data, percentile=[1,99], vmin=None, vmax=None):
    '''
    Compute vmin and vmax for image display. By default uses the
    percentile range [1,99], but optionally vmin and/or vmax can
    be set by the user. Setting percentile to None results in no
    stretch. Passing in a boolean array uses vmin=0, vmax=1. This
    function is used internally by plotting functions.
    Parameters
    ––––––––––
    data : array-like
        Input data array (e.g., 2D image) for which to compute vmin and vmax.
    percentile : list or tuple of two floats, default=[1,99]
        Percentile range '[pmin, pmax]' to compute vmin and vmax.
        If None, sets vmin and vmax to None.
    vmin : float or None, default=None
        If provided, overrides the computed vmin.
    vmax : float or None, default=None
        If provided, overrides the computed vmax.
    Returns
    –––––––
    vmin : float or None
        Minimum value for image scaling.
    vmax : float or None
        Maximum value for image scaling.
    '''
    # check if data is an array
    data = check_is_array(data)
    # check if data is boolean
    if data.dtype == bool:  # special case for boolean arrays
        return 0, 1
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

def compute_cube_percentile(cube, slice_idx, vmin, vmax):
    '''
    Compute percentile-based intensity limits from a data cube slice.
    This function is intended to be used to compute an image scaling.
    Parameters
    ––––––––––
    cube : ndarray, SpectralCube, or DataCube
        Input data cube of shape (N_frames, N, M).
    slice_idx : int or list of int, optional
        Index or indices specifying the slice along the first axis:
        - i -> returns 'cube[i]'
        - [i] -> returns 'cube[i]'
        - [i, j] -> returns 'cube[i:j+1].sum(axis=0)'
    vmin : float
        Lower percentile (0–100) for intensity scaling.
    vmax : float
        Upper percentile (0–100) for intensity scaling.
    Returns
    –––––––
    vmin : float
        Computed lower intensity value corresponding to the
        specified 'vmin' percentile.
    vmax : float
        Computed upper intensity value corresponding to the
        specified 'vmax' percentile.
    '''
    # ensure cube is stripped of metadata
    cube = return_cube_data(cube)
    # slice cube
    data = return_cube_slice(cube, slice_idx)
    # compute vmin and vmax
    vmin = np.nanpercentile(data.value, vmin)
    vmax = np.nanpercentile(data.value, vmax)

    return vmin, vmax

# Axes Labels, Format, and Styling
# ––––––––––––––––––––––––––––––––
def add_colorbar(im, ax, cbar_width=0.03, cbar_pad=0.015, clabel=None):
    '''
    Add a colorbar next to an Axes.
    Parameters
    ––––––––––
    im : matplotlib.cm.ScalarMappable
        The image, contour set, or mappable object returned by
        a plotting function (e.g., 'imshow', 'scatter', etc...).
    ax : matplotlib.axes.Axes
        The axes to which the colorbar will be attached.
    cbar_width : float, optional, default=0.03
        Width of the colorbar in figure coordinates.
    cbar_pad : float, optional, default=0.015
        Padding between the main axes and the colorbar
        in figure coordinates.
    clabel : str, optional
        Label for the colorbar. If None, no label is set.
    '''
    # extract figure from axes
    fig = ax.figure
    # add colorbar axes
    cax = fig.add_axes([ax.get_position().x1+cbar_pad, ax.get_position().y0,
                        cbar_width, ax.get_position().height])
    # add colorbar
    cbar = fig.colorbar(im, cax=cax, pad=0.04)
    # formatting and label
    cbar.ax.tick_params(which='both', direction='out')
    if clabel is not None:
        cbar.set_label(fr'{clabel}')

def set_axis_limits(xdata, ydata, ax, xlim=None, ylim=None):
    '''
    Set axis limits based on concatenated data or user-provided limits.
    Parameters
    ––––––––––
    xdata : list/tuple of arrays or array
        X-axis data from multiple datasets.
    ydata : list/tuple of arrays or array
        Y-axis data from multiple datasets.
    ax : matplotlib axis
        The matplotlib axes object on which to set the axis limits.
    xlim : tuple/list, optional
        User-defined x-axis limits.
    ylim : tuple/list, optional
        User-defined y-axis limits.
    '''
    if xdata is not None:
        # concatenate list of data into single array
        if isinstance(xdata, (list, tuple)):
            xdata = np.concatenate(xdata)
        else:
            xdata = np.asarray(xdata)
        # min and max values across data sets
        xmin = return_array_values(np.nanmin(xdata))
        xmax = return_array_values(np.nanmax(xdata))
        # use computed limits unless user overides
        xlim = xlim if xlim is not None else [xmin, xmax]

    if ydata is not None:
        # concatenate list of data into single array
        if isinstance(ydata, (list, tuple)):
            ydata = np.concatenate(ydata)
        else:
            ydata = np.asarray(ydata)
        # min and max values across data sets
        ymin = return_array_values(np.nanmin(ydata))
        ymax = return_array_values(np.nanmax(ydata))
        # use computed limits unless user overides
        ylim = ylim if ylim is not None else [ymin, ymax]

    # set x and y limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def set_axis_labels(X, Y, ax, xlabel=None, ylabel=None, use_brackets=False):
    '''
    Automatically format labels including units for any plot involving intensity as
    a function of spectral type.
    Parameters
    ––––––––––
    X : '~astropy.units.Quantity' or object with 'unit' or 'spectral_unit' attribute
        The data for the x-axis, typically a spectral axis (frequency, wavelength, or velocity).
    Y : '~astropy.units.Quantity' or object with 'unit' or 'spectral_unit' attribute
        The data for the y-axis, typically flux or intensity.
    ax : 'matplotlib.axes.Axes'
        The matplotlib axes object on which to set the labels.
    xlabel : str or None, optional, default=None
        Custom label for the x-axis. If None, the label is inferred from 'X'.
    ylabel : str or None, optional, default=None
        Custom label for the y-axis. If None, the label is inferred from 'Y'.
    use_brackets : bool, optional, default=False
        If True, wrap units in square brackets '[ ]'. If False, use parentheses '( )'.
    Notes
    –––––
    - Units are formatted using 'set_unit_labels', which provides LaTeX-friendly labels.
    - If units are not recognized, only the axis type (e.g., 'Spectral Axis', 'Intensity')
      is displayed without units.
    '''
    # determine spectral type of data (frequency, length, or velocity)
    spectral_type = {
        'frequency': 'Frequency',
        'length': 'Wavelength',
        'speed/velocity': 'Velocity',
    }.get(str(getattr(getattr(X, 'unit', None), 'physical_type', None)), 'Spectral Axis')
    # determine intensity type of data (counts, flux)
    flux_type = {
        'count': 'Counts',
        'flux density': 'Flux',
        'surface brightness': 'Flux'
    }.get(str(getattr(getattr(Y, 'unit', None), 'physical_type', None)), 'Intensity')
    # unit bracket type [] or ()
    brackets = [r'[$',r'$]'] if use_brackets else [r'($',r'$)']
    # if xlabel is not overidden by user
    if xlabel is None:
        # determine unit from data
        x_unit = str(getattr(X, 'spectral_unit', getattr(X, 'unit', None)))
        x_unit_label = set_unit_labels(x_unit)
        # format unit label if valid unit is found
        if x_unit_label:
            xlabel = fr'{spectral_type} {brackets[0]}{x_unit_label}{brackets[1]}'
        else:
            xlabel = spectral_type
    # if ylabel is not overidden by user
    if ylabel is None:
        # determine unit from data
        y_unit = str(getattr(Y, 'spectral_unit', getattr(Y, 'unit', None)))
        y_unit_label = set_unit_labels(y_unit)
        # format unit label if valid unit is found
        if y_unit_label:
            ylabel = fr'{flux_type} {brackets[0]}{y_unit_label}{brackets[1]}'
        else:
            ylabel = flux_type
    # set plot labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def set_unit_labels(unit):
    '''
    Convert an astropy unit string into a LaTeX-formatted label for plotting.
    Parameters
    ––––––––––
    unit : str
        The unit string to convert. Common astropy units such as 'um',
        'Angstrom', 'km / s', etc. are mapped to LaTeX-friendly forms.
    Returns
    –––––––
    str or None
        A LaTeX-formatted unit label if the input is recognized.
        Returns None if the unit is not in the predefined mapping.
    '''
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
    }.get(str(unit))

    return unit_label

# Plot Matplotlib Patches and Shapes
# ––––––––––––––––––––––––––––––––––
def plot_circles(circles, ax, colors=None, linewidth=2, fill=False, cmap='turbo'):
    '''
    Plot one or more circles on a Matplotlib axis with customizable style.
    Parameters
    ––––––––––
    circles : array-like or None
        Circle coordinates and radii. Can be a single circle `[x, y, r]`
        or a list/array of circles `[[x1, y1, r1], [x2, y2, r2], ...]`.
        If None, no circles are plotted.
    ax : matplotlib.axes.Axes
        The Matplotlib axis on which to plot the circles.
    colors : list of str or None, optional, default=None
        List of colors to cycle through for each circle. Defaults to
        ['r', 'mediumvioletred', 'magenta']. A single color can also
        be passed. If there are more circles than colors, colors are
        sampled from a colormap using sample_cmap().
    linewidth : float, optional, default=2
        Width of the circle edge lines.
    fill : bool, optional, default=False
        Whether the circles are filled.
    cmap : str, optional, default='turbo'
        matplolib cmap used to sample default circle colors.
    '''
    if circles is not None:
        # ensure circles is list [x,y,r] or list of list [[x,y,r],[x,y,r]...]
        circles = np.atleast_2d(circles)
        if circles.shape[1] != 3:
            raise ValueError(
                'Circles must be either [x, y, r] or [[x1, y1, r1], [x2, y2, r2], ...]'
            )
        # number of circles to plot
        N = circles.shape[0]
        # set circle colors
        if colors is None:
            colors = ['r', 'mediumvioletred', 'magenta'] if N<=3 else sample_cmap(N)
        if isinstance(colors, str):
            colors = [colors]

        # plot each cirlce
        for i, circle in enumerate(circles):
            x, y, r = circle
            color = colors[i%len(colors)]
            circle_patch = Circle((x, y), radius=r, fill=fill, linewidth=linewidth, color=color)
            ax.add_patch(circle_patch)

def copy_ellipse(ellipse):
    '''
    Returns a copy of an Ellipse object.
    Parameters
    ––––––––––
    ellipse : matplotlib.patches.Ellipse
        The Ellipse object to copy.
    Returns
    ––––––––––
    matplotlib.patches.Ellipse
        A new Ellipse object with the same properties as the input.
    '''
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

def plot_ellipses(ellipses, ax):
    '''
    Plots an ellipse or list of ellipses to an axes.
    Parameters
    ––––––––––
    ellipses : matplotlib.patches.Ellipse or list
        The Ellipse or list of Ellipses to plot.
    ax : matplotlib.axes.Axes
        Matplotlib axis on which to plot the ellipses(s).
    '''
    if ellipses is not None:
        # ensure ellipses is iterable
        ellipses = ellipses if isinstance(ellipses, list) else [ellipses]
        # plot each ellipse
        for ellipse in ellipses:
            ax.add_patch(copy_ellipse(ellipse))

def plot_interactive_ellipse(center, w, h, ax,
                             text_loc=[0.03,0.03],
                             text_color='k',
                             highlight=True):
    '''
    Create an interactive ellipse selector on an Axes
    along with an interactive text window displaying
    the current ellipse center, width, and height.
    Parameters
    ––––––––––
    center : tuple of float
        (x, y) coordinates of the ellipse center in data units.
    w : float
        Width of the ellipse.
    h : float
        Height of the ellipse.
    ax : matplotlib.axes.Axes
        The Axes on which to draw the ellipse selector.
    text_loc : list of float, optional, default=[0.03,0.03]
        Position of the text label in Axes coordinates, given as [x, y].
    text_color : str, optional, default='k'
        Color of the annotation text.
    highlight : bool, optional, default=True
        If True, adds a bbox to highlight the text.
    Notes
    –––––
    Ensure an interactive backend is active. This can be
    activated with use_interactive().
    '''
    # define text for ellipse data display
    facecolor = 'k' if text_color == 'w' else 'w'
    bbox = dict(facecolor=facecolor, alpha=0.6, edgecolor="none") if highlight else None
    text = ax.text(text_loc[0], text_loc[1], '',
                   transform=ax.transAxes,
                   size='small', color=text_color,
                   bbox=bbox)
    # define ellipse
    ellipse_region = EllipsePixelRegion(center=PixCoord(x=center[0], y=center[1]),
                                        width=w, height=h)
    # define interactive ellipse
    selector = ellipse_region.as_mpl_selector(ax, callback=partial(update_ellipse_region, text=text))
    # bind ellipse to axes
    ax._ellipse_selector = selector

def update_ellipse_region(region, text):
    '''
    Update ellipse information text when the
    interactive region is modified.
    Parameters
    ––––––––––
    region : regions.EllipsePixelRegion
        The ellipse region being updated.
    text : matplotlib.text.Text
        The text object used to display ellipse parameters.
    '''
    # extract properties from ellipse object
    x_center = region.center.x
    y_center = region.center.y
    width = region.width
    height = region.height
    major = max(width, height)
    minor = min(width, height)
    # display properties
    text.set_text(
        f'Center: [{x_center:.1f}, {y_center:.1f}]\n'
        f'Major: {major:.1f}\n'
        f'Minor: {minor:.1f}\n'
    )

def return_ellipse_region(center, w, h, angle=0, fill=False):
    '''
    Create a matplotlib.patches.Ellipse object.
    Parameters
    ––––––––––
    center : tuple of float
        (x, y) coordinates of the ellipse center.
    w : float
        Width of the ellipse (along x-axis before rotation).
    h : float
        Height of the ellipse (along y-axis before rotation).
    angle : float, default=0
        Rotation angle of the ellipse in degrees (counterclockwise).
    fill : bool, default=False
        Whether the ellipse should be filled (True) or only outlined (False).
    Returns
    –––––––
    matplotlib.patches.Ellipse
        An Ellipse patch that can be added to a matplotlib Axes.
    '''
    ellipse = Ellipse(xy=(center[0], center[1]), width=w, height=h, angle=angle, fill=fill)

    return ellipse

def plot_points(points, ax, color='r', size=20, marker='*'):
    '''
    Plot points on a given Matplotlib axis with customizable style.
    Parameters
    ––––––––––
    points : array-like or None
        Coordinates of points to plot. Can be a single point `[x, y]`
        or a list/array of points `[[x1, y1], [x2, y2], ...]`.
        If None, no points are plotted.
    ax : matplotlib.axes.Axes
        The Matplotlib axis on which to plot the points.
    color : str or list or int, optional, default='r'
        Color of the points. If an integer, will draw colors
        from sample_cmap().
    size : float, optional, default=20
        Marker size.
    marker : str, optional, default='*'
        Matplotlib marker style.
    '''
    if points is not None:
        points = np.asarray(points)
        # ensure points is list [x,y] or list of list [[x,y],[x,y]...]
        if points.ndim == 1 and points.shape[0] == 2:
            points = points[np.newaxis, :]
        elif points.ndim != 2 or points.shape[1] != 2:
            error = 'Points must be either [x, y] or [[x1, y1], [x2, y2], ...]'
            raise ValueError(error)
        if isinstance(color, int):
            color = sample_cmap(color)
        color = color if isinstance(color, list) else [color]
        # loop through each set of points in points and plot
        for i, point in enumerate(points):
            ax.scatter(point[0], point[1], s=size, marker=marker, c=color[i%len(color)])

# Notebook Utils
# ––––––––––––––
def use_inline():
    '''
    Start an inline IPython backend session.
    Allows for inline plots in IPython sessions
    like Jupyter Notebook.
    '''
    try:
        from IPython.core.getipython import get_ipython
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
        from IPython.core.getipython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic("matplotlib", "ipympl")
        else:
            print("Not in an IPython environment.")
    except ImportError:
        print("IPython is not installed. Install it to use this feature.")

def plt_close():
    '''
    Closes all interactive plots in session.
    '''
    plt.close('all')
