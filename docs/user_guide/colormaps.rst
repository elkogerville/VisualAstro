Colormaps
=========

.. currentmodule:: visualastro

VisualAstro provides modules for working with colormaps. VisualAstro registers its colormaps with matplotlib
at import time

The examples on this page assume:

.. code-block:: python

   import visualastro as va


Accessing Colormaps
-------------------

Retrieve a colormap with :func:`~visualastro.get_cmap`. It is a thin wrapper
around :func:`~matplotlib.pyplot.get_cmap` with additional options:

- ``cmap_range``: ``(min, max)`` fraction of the colormap to keep
- ``bad_color``: color for masked or NaN values
- ``under_color``: color for values below the normalization minimum
- ``over_color``: color for values above the normalization maximum

The example below truncates ``turbo`` to the range 0.1 to 0.9 and sets
custom colors for masked, under-range, and over-range values.


.. plot:: ../examples/plotting/colors/plot_get_cmap.py
    :include-source: True


Creating Colormaps
------------------

Create a continuous or discrete colormap with :func:`~visualastro.create_cmap`. You can create the
colormap out of a sequence of colors or an existing VisualAstro colormap.


.. plot:: ../examples/plotting/colors/plot_create_cmap.py
    :include-source: True


Discrete colormap:


.. plot:: ../examples/plotting/colors/plot_create_cmap_discrete.py
    :include-source: True


Plot All Colormaps
------------------

Plot every colormap currently registered with matplotlib with :meth:`~visualastro.help.cmap`.

.. plot:: ../examples/plotting/colors/plot_all_cmaps.py
    :include-source: True


Plot Colormap Lightness
-----------------------

.. warning::

   This requires the optional package ``colorspacious``. See :doc:`installation`.


Plot the lightness :math:`L^*` values of a colormap, similar to the plots by
`matplotlib <https://matplotlib.org/stable/users/explain/colors/colormaps.html#lightness-of-matplotlib-colormaps>`__.


.. plot:: ../examples/plotting/colors/plot_cmap_L.py
    :include-source: True
