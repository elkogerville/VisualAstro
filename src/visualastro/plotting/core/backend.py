"""
Author: Elko Gerville-Reache
Date Created: 2025-05-24
Date Modified: 2026-10-25
Description:
    Matplotlib backend utility functions.
"""

import matplotlib.pyplot as plt


def inline():
    """
    Start an inline IPython backend session.
    Allows for inline plots in IPython sessions
    like Jupyter Notebook.
    """
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        raise ImportError(
            'IPython is not installed. Install it to use this feature'
        )

    ipython = get_ipython()
    if ipython is None:
        print('Not inside an IPython environment')
        return None

    try:
        close()
        ipython.run_line_magic('matplotlib', 'inline')
    except Exception as e:
        print(f'Unable to set inline backend: {e}')


def interactive():
    """
    Start an interactive IPython backend session.
    Allows for interactive plots in IPython sessions
    like Jupyter Notebook.

    Ensure ipympl is installed:
    >>> $ conda install -c conda-forge ipympl
    """
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        raise ImportError(
            'IPython is not installed. Install it to use this feature'
        )

    ipython = get_ipython()
    if ipython is None:
        print('Not inside an IPython environment')
        return None

    try:
        ipython.run_line_magic('matplotlib', 'ipympl')
    except Exception as e:
        print(
            f'ipympl backend unavailable: {e}. Please install with:\n'
            f'$ conda install -c conda-forge ipympl'
        )


def close():
    """
    Closes all interactive plots in session.
    """
    plt.close('all')
