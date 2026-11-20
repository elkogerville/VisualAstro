legend
======

.. currentmodule:: visualastro

.. autofunction:: legend


Overview
--------

Thin wrapper around :meth:`matplotlib.axes.Axes.legend`. Accepts all Matplotlib
arguments and adds two features: extra location keywords and a different default
frame style. Labels are taken from the ``label`` keyword of the plotting calls on
``ax``, exactly as in Matplotlib.


Default Style
^^^^^^^^^^^^^

``frameon`` defaults to ``False``. Pass ``frameon=True`` to draw the frame. Frame
appearance is controlled by the usual Matplotlib keywords:

- ``ec`` sets the edge color.
- ``fancybox=True`` rounds the frame corners.
- ``alpha`` sets the frame transparency.


.. plot:: ../examples/plotting/legends/plot_legend_default.py
    :include-source: True


Frame
^^^^^

With ``frameon=True`` the frame is drawn with square corners and the edge color
given by ``ec``.

- ``frameon=True`` enables the frame.
- ``ec='k'`` sets a black edge.
- Corners are square unless ``fancybox=True`` or the active style sets ``legend.fancybox: True``.


.. plot:: ../examples/plotting/legends/plot_legend_frameon.py
    :include-source: True


New Locations
^^^^^^^^^^^^^

``'top'`` and ``'bottom'`` place the legend outside the axes, above or below the
plotting area. Each may be combined with a horizontal keyword:

- ``'top'``, ``'top left'``, ``'top center'``, ``'top right'``
- ``'bottom'``, ``'bottom left'``, ``'bottom center'``, ``'bottom right'``

All standard Matplotlib ``loc`` values (``'upper right'``, ``'best'``, ...) remain valid.


.. plot:: ../examples/plotting/legends/plot_legend_loc.py
    :include-source: True


Setting the Line Width
^^^^^^^^^^^^^^^^^^^^^^

You can also set the ``linewidth`` of the legend box directly.


.. plot:: ../examples/plotting/legends/plot_legend_linewidth.py
    :include-source: True


See Also
^^^^^^^^

- :func:`hist`
- :func:`step`
