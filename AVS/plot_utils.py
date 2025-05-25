import os
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

def imshow(data, idx=None, cmap='turbo', style='astro', vmin=None, vmax=None, norm=None,
           percentile=[3,99.5], circles=None, plot_boolean=False, transpose=True):
    data = check_is_array(data)
    if idx is not None and isinstance(idx, int):
        data = data[idx]
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

        plt.show()

def plot_histogram(data, bins='auto', style='astro', xlog=False, ylog=False, labels=None):
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
        plt.show()

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
        this_dir = os.path.dirname(os.path.abspath(__file__))
        style_path = os.path.join(this_dir, 'stylelib', style)
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
