.. _colorset guide:

Colorsets
=========

.. currentmodule:: visualastro

VisualAstro provides ``va.colorsets``, an interface for viewing, querying, and
plotting the registered colorsets. See :class:`~visualastro.Colorsets` for the
full API.

The examples on this page assume:

.. code-block:: python

   import visualastro as va


Listing Colorsets
-----------------

List the names of all available colorsets:

.. code-block:: python

    va.colorsets.available

Alias names are listed separately:

.. code-block:: python

    va.colorsets.aliases


Accessing Colorsets
-------------------

Retrieve a colorset with the :meth:`~Colorsets.get` method or with
indexing (``va.colorsets['name']``):

.. code-block:: python

    va.colorsets.get('supernatural')
    va.colorsets['celestial']


Querying Colorsets
------------------

You can refine your search criteria when selecting a colorset with :meth:`~Colorsets.query`.
This allows you to set minimum or maximum bounds to the number of colors with ``min_length``
and ``max_length``, or set an exact length with ``length``. You can further sort by ``'length'``
or ``'name'`` with ``sort_by``.

.. code-block:: python

    # all colorsets with 7 colors
    va.colorsets.query(length=7)

    # colorsets with more than 3 colors but less than 8 sorted by length
    va.colorsets.query(min_length=3, max_length=8, sort_by='length')


Plotting Colorsets
------------------

You can also view the colorsets with :meth:`~Colorsets.plot`. This method works just like
:meth:`~Colorsets.query` but plots the colorset instead as a row of colors. If no input is
passed in, all colorsets are plotted.


Setting Minimum and Maximum Lengths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plot only the colorsets in the range ``[min_length, max_length]``.


.. plot:: ../examples/plotting/colors/plot_colorset_min_length.py
    :include-source: True


Setting a Set Length
^^^^^^^^^^^^^^^^^^^^


.. plot:: ../examples/plotting/colors/plot_colorset_length.py
    :include-source: True


Setting a Transform
^^^^^^^^^^^^^^^^^^^

You can set one of the supported color transforms to change all the colorsets automatically.


.. plot:: ../examples/plotting/colors/plot_colorset_transform.py
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


.. plot:: ../examples/plotting/colors/plot_colorset_cvd.py
    :include-source: True


Plotting All Colorsets
^^^^^^^^^^^^^^^^^^^^^^


.. plot:: ../examples/plotting/colors/plot_colorset_all.py
    :include-source: True
