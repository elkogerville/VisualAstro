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


.. literalinclude:: /../examples/plotting/colors/plot_colorset_min_length.py
    :language: python


.. image:: /_static/images/reference/plotting/colorset_min_length.png
    :scale: 50 %
    :alt: spectral_lines on spectral data
    :align: center
