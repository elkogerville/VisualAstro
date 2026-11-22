.. _color-guide:

Named Colors
============

.. currentmodule:: visualastro

VisualAstro adds many colors to matplotlib when you import the package!
The colors are registered with matplotlib so they work natively.
To preview and query the colorsets, use the ``va.colors`` interface.
See :class:`~visualastro.Colors` for the full api.

The examples on this page assume:

.. code-block:: python

   import visualastro as va


Listing Named Colors
--------------------

List the names of all available named colors:

.. code-block:: python

    va.colors.available

Alias names are listed separately:

.. code-block:: python

    va.colors.aliases


Accessing Named Colors
----------------------

Retrieve a color with the :meth:`~Colors.get` method or with
indexing (``va.colors['name']``):

.. code-block:: python

    va.colors.get('pondwater')
    va.colors['dark verdigris']


Plotting Colors Tables
----------------------

VisualAstro organizes the recognized colors in colortables. These are collections of curated colors,
like the VisualAstro named colors, the default matplotlib colors, or xkcd colors. You can view a
colortable with :meth:`~Colorsets.plot`. Possible inputs are

- ``'named_colors'`` or ``None``: VisualAstro and Matplotlib named colors
- `'visualastro'` or `'va'`: VisualAstro named colors
- ``'matplotlib'`` or ``'mpl'`` or ``'css4'``: Matplotlib named colors
- `'xkcd'`: XKCD named colors
- `'base'`: Matplotlib base colors
- `'tableau'`: Matplotlib tableau colors
- `'all'` or `'all_colors'`: All of the above

If no input is passed in, all VisualAstro and matplotlib colors are plotted. By default,
the colors are sorted by hue. Turn off this behavior with ``sort_colors=False``.


VisualAstro Named Colors
^^^^^^^^^^^^^^^^^^^^^^^^

VisualAstro comes with many custom colors!

.. plot:: ../examples/plotting/colors/plot_colors_visualastro.py
    :include-source: True


Color Transforms
^^^^^^^^^^^^^^^^

You can set one of the supported color transforms to change all the colorsets automatically.
Supported transforms include ``'lighten'``, ``'darken'``, ``'saturate'``, and ``'desaturate'``.
The intensity is controlled by ``factor``.

.. plot:: ../examples/plotting/colors/plot_colors_transform.py
    :include-source: True


CVD Simulations
^^^^^^^^^^^^^^^

.. warning::

    This requires the optional package ``colorspacious``. See :doc:`installation`.

You can preview a colorset as it appears under altered color vision by
passing ``cvd_type`` as ``'deuteranomaly'``, ``'protanomaly'``,
``'tritanomaly'``, or ``'all'`` (``'d'``, ``'p'``, and ``'t'`` are also
accepted). The ``colorspacious`` package simulates how a viewer with that
color vision type would perceive the colorset. Use ``severity`` to set the
strength of the effect (0 to 100, default 100).


.. plot:: ../examples/plotting/colors/plot_colors_cvd.py
    :include-source: True


Matplotlib Named Colors
^^^^^^^^^^^^^^^^^^^^^^^

You can see the default matplotlib colors as well:

.. plot:: ../examples/plotting/colors/plot_colors_matplotlib.py
    :include-source: True


XKCD Named Colors
^^^^^^^^^^^^^^^^^

VisualAstro registers xkcd colors under their bare names (e.g. ``'sky blue'``), so the
``'xkcd:'`` prefix is optional. Names that collide with existing Matplotlib or VisualAstro
colors keep the prefix (e.g. ``'xkcd:cyan'``).

.. plot:: ../examples/plotting/colors/plot_colors_xkcd.py
    :include-source: True
