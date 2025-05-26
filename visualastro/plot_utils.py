import os
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

# ––––––––––––––––––
# Plotting Functions
# ––––––––––––––––––
def imshow(data, idx=None, cmap='turbo', style='astro', vmin=None, vmax=None, norm=None,
           percentile=[3,99.5], circles=None, plot_boolean=False, transpose=True, savefig=False, dpi=600):
    data = check_is_array(data)
    if idx is not None:
        data = return_cube_slice(data, idx)
    if plot_boolean:
        vmin = 0
        vmax = 1
    else:
        vmin = np.nanpercentile(data, percentile[0]) if vmin is None else vmin
        vmax = np.nanpercentile(data, percentile[1]) if vmax is None else vmax
    style = return_stylename(style)
    with plt.style.context(style):
        data = data.T if transpose else data
        fig, ax = plt.subplots(figsize=(6,6))
        if norm is None:
            ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            norm = return_imshow_norm(vmin, vmax, norm)
            ax.imshow(data, origin='lower', cmap=cmap, norm=norm)
        if circles is not None:
            circle_colors = ['r', 'mediumvioletred', 'magenta']
            for i, circle in enumerate(circles):
                x, y, r = circle
                circle = Circle((x, y), radius=r, fill=False, linewidth=2,
                                color=circle_colors[i%len(circle_colors)])
                ax.add_patch(circle)
        if savefig:
                save_figure_2_disk(dpi)
        plt.show()

def plot_histogram(data, bins='auto', style='astro', xlog=False,
                   ylog=False, labels=None, savefig=False, dpi=600):
    data = check_is_array(data)
    if data.ndim == 2:
        data = data.flatten()
    style = return_stylename(style)
    with plt.style.context(style):
        plt.figure(figsize=(5,5))
        plt.hist(data, bins=bins)

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

def return_imshow_norm(vmin, vmax, norm):
    norm = norm.lower()
    norm_map = {
        'log': LogNorm(vmin=vmin, vmax=vmax),
        'asinh': ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
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

def save_figure_2_disk(dpi):
    '''
    Saves current figure to disk, and prompts user for a filename and format
    Parameters
    ----------
    dpi: float
        resolution in dots per inch
    '''
    file_name = input('input filename for image (ex: myimage.pdf): ')
    plot_format = input('please enter format: png or pdf')
    while plot_format not in {'png', 'pdf', 'svg'}:
        plot_format = input('please enter format (png, pdf, or svg): ')
    plt.savefig(file_name, format=plot_format, bbox_inches='tight', dpi=dpi)

def set_plot_colors(user_colors=None):
    default_color_map = 'ibm_contrast'
    color_map = {
        'ibm_contrast': ['#648FFF', '#DC267F', '#785EF0', '#26DCBA', '#FFB000', '#FE6100'],
        #        mossgreen     bbblue    ibmblue   ibmpurple     pvr      traffico    gold
        'astro': ['#26DCBA', '#9FB7FF', '#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'],
        #          dsb        pvr       lilac    mossgreen  slateblue
        'MSG': ['#483D8B', '#DC267F', '#DBB0FF', '#26DCBA', '#7D7FF3'],
        #        ibmblue   ibmpurple     pvr      traffico    gold
        'ibm': ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
    }
    model_colors = ['r', 'purple', 'magenta']
    if user_colors is not None:
        if user_colors in color_map:
            colors = color_map[user_colors]
        else:
            if isinstance(user_colors, str):
                user_colors = [user_colors]
            colors = user_colors
    else:
        colors = color_map[default_color_map]

    return colors, model_colors

def set_unit_labels(unit):
    unit_map = {
        'MJy / sr': r'MJy sr$^{-1}$',
        'micron': r'$\mu$m',
        'um': r'$\mu$m',
    }
    return unit_map.get(unit, unit) if unit else None

def set_axis_labels(X, Y, x_unit, y_unit, use_brackets=False):
    if x_unit is None:
        x_unit = str(getattr(X, 'spectral_unit', getattr(X, 'unit', None)))
    if y_unit is None:
        y_unit = str(getattr(Y, 'spectral_unit', getattr(Y, 'unit', None)))

    # Format for display (including LaTeX)
    x_unit_label = set_unit_labels(x_unit)
    y_unit_label = set_unit_labels(y_unit)
    if use_brackets:
        x_unit_label = '[' + x_unit_label + ']'
        y_unit_label = '[' + y_unit_label + ']'
    else:
        x_unit_label = '(' + x_unit_label + ')'
        y_unit_label = '(' + y_unit_label + ')'
    xlabel = fr'Wavelength {x_unit_label}' if x_unit_label else 'Wavelength'
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
