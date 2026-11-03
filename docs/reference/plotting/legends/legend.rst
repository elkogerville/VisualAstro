set_limits
==========

.. currentmodule:: visualastro

.. autofunction:: legend


Examples
--------

``legend`` is a thin wrapper around :meth:`matplotlib.axes.Axes.legend`.
It accepts all of Matplotlib's arguments and adds the following:

- **Additional locations:** ``'top'`` and ``'bottom'`` place the legend
  above or below the axes. Each may be combined with ``'left'``,
  ``'center'`` or ``'right'`` (e.g. ``'top left'``, ``'bottom center'``).
- **Default style:** ``frameon=False``. Pass ``frameon=True`` to draw the
  frame.


.. literalinclude:: /../examples/plotting/legends/plot_legend.py
    :language: python


.. image:: /_static/images/reference/plotting/legend_hist.png
    :scale: 50 %
    :alt: different legend styles
    :align: center
