import os
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy import units as u
from astropy.units import spectral
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

# ––––––––––––––––––
# Plotting Functions
# ––––––––––––––––––
def imshow(data, idx=None, vmin=None, vmax=None, norm=None, percentile=[3,99.5],
           cmap='turbo', style='astro', points=None, circles=None, plot_boolean=False,
           transpose=True, savefig=False, dpi=600):
    data = check_is_array(data)
    if idx is not None:
        data = return_cube_slice(data, idx)
    if plot_boolean:
        vmin = 0
        vmax = 1
        norm = None
    else:
        vmin, vmax = set_vmin_vmax(data, percentile, vmin, vmax)
        norm = return_imshow_norm(vmin, vmax, norm)
    style = return_stylename(style)
    with plt.style.context(style):
        data = data.T if transpose else data
        fig, ax = plt.subplots(figsize=(6,6))
        if norm is None:
            ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            ax.imshow(data, origin='lower', norm=norm, cmap=cmap)
        if circles is not None:
            circle_colors = ['r', 'mediumvioletred', 'magenta']
            for i, circle in enumerate(circles):
                x, y, r = circle
                circle = Circle((x, y), radius=r, fill=False, linewidth=2,
                                color=circle_colors[i%len(circle_colors)])
                ax.add_patch(circle)
        if points is not None:
            for point in points:
                plt.scatter(point[0], point[1], s=20, marker='*', c='r')
        plt.xlabel('X [pixels]')
        plt.ylabel('Y [pixels]')
        if savefig:
                save_figure_2_disk(dpi)
        plt.show()

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
    if isinstance(cube, dict):
        cube = np.asarray(cube['data'])
    else:
        cube = np.asarray(cube)

    return cube

def return_stylename(style):
    if style in mpl.style.available:
        return style
    else:
        style = style + '.mplstyle'
        dir_path = os.path.dirname(os.path.realpath(__file__))
        style_path = os.path.join(dir_path, 'stylelib', style)
        return style_path

def set_vmin_vmax(data, percentile, vmin, vmax):
    if percentile is not None:
        vmin = np.nanpercentile(data, percentile[0]) if vmin is None else vmin
        vmax = np.nanpercentile(data, percentile[1]) if vmax is None else vmax
    else:
        vmin = None
        vmax = None

    return vmin, vmax

def return_imshow_norm(vmin, vmax, norm):
    norm = 'none' if norm is None else norm
    norm = norm.lower()
    norm_map = {
        'asinh': ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch()),
        'log': LogNorm(vmin=vmin, vmax=vmax),
        'none': None
    }
    if norm not in norm_map:
        raise ValueError(f"ERROR: unsupported norm: {norm}")

    return norm_map[norm]

def return_cube_slice(cube, idx):
    if isinstance(idx, int):
        return cube[idx]
    elif isinstance(idx, list):
        if len(idx) == 1:
            return cube[idx[0]]
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

def save_figure_2_disk(dpi):
    '''
    Saves current figure to disk, and prompts user for a filename and format
    Parameters
    ----------
    dpi: float
        resolution in dots per inch
    '''
    file_name = input('input filename for image (ex: myimage.pdf): ')
    plot_format = input('please enter format (png, pdf, or svg): ')
    while plot_format not in {'png', 'pdf', 'svg'}:
        plot_format = input('please enter format (png, pdf, or svg): ')
    plt.savefig(file_name, format=plot_format, bbox_inches='tight', dpi=dpi)

def set_plot_colors(user_colors=None):
    default_color_map = 'ibm_contrast'
    color_map = {
        #                  dsb        mvr      ibmblue      gold
        'visualastro': ['#483D8B', '#DC267F', '#648FFF', '#FFB000', '#26DCBA'],
        'ibm_contrast': ['#648FFF', '#DC267F', '#785EF0', '#26DCBA', '#FFB000', '#FE6100'],
        #          bbblue    ibmblue   ibmpurple     mvr      traffico    gold      pondwater  mossgreen
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
    unit_map = {
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
        'km / s': r'\mathrm{km\ s^{-1}}',
    }

    return unit_map.get(str(unit), unit)

def set_axis_labels(X, Y, x_unit, y_unit, use_brackets=False):
    if x_unit is None:
        x_unit = str(getattr(X, 'spectral_unit', getattr(X, 'unit', None)))
    if y_unit is None:
        y_unit = str(getattr(Y, 'spectral_unit', getattr(Y, 'unit', None)))

    # Format for display (including LaTeX)
    x_unit_label = set_unit_labels(x_unit)
    y_unit_label = set_unit_labels(y_unit)
    if use_brackets:
        x_unit_label = r'[$' + x_unit_label + r'$]'
        y_unit_label = r'[$' + y_unit_label + r'$]'
    else:
        x_unit_label = r'($' + x_unit_label + r'$)'
        y_unit_label = r'($' + y_unit_label + r'$)'

    spectral_type_map = {
        'frequency': 'Frequency',
        'length': 'Wavelength',
        'speed/velocity': 'Velocity',
    }

    spectral_type = spectral_type_map.get(str(X.unit.physical_type), 'Spectral Axis')
    xlabel = fr'{spectral_type} {x_unit_label}' if x_unit_label else spectral_type
    ylabel = fr'Flux {y_unit_label}' if y_unit_label else 'Flux'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# ––––––––––––––
# Notebook Utils
# ––––––––––––––
def use_inline():
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
    plt.close('all')
