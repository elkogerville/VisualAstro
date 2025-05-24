import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def imshow(data, cmap='turbo', style='astro', vmin=None, vmax=None,
           percentile=[3,99.5], circles=None, plot_boolean=False, transpose=True):
    data = check_is_array(data)
    if plot_boolean:
        vmin = 0
        vmax = 1
    else:
        vmin = np.nanpercentile(data, percentile[0]) if vmin is None else vmin
        vmax = np.nanpercentile(data, percentile[1]) if vmax is None else vmax
    with plt.style.context(style + '.mplstyle'):
        data = data.T if transpose else data
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(data.T, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
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
    with plt.style.context(style + '.mplstyle'):
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
