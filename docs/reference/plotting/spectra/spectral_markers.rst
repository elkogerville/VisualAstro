spectral_markers
================

.. currentmodule:: visualastro

.. autofunction:: spectral_markers


Overview
--------

Allows for automated labeling of known spectral lines. Pass the line positions
and labels as parallel sequences, use ``y`` to control the position of the labels
and ``h`` to control the height of the prong markers.


Marker Geometry
^^^^^^^^^^^^^^^

``y`` and ``h`` are required. ``y`` is the data coordinate of the prong tips and ``h`` is the
signed marker height: the connector sits at ``y + h``, above the tips for ``h > 0`` and below
for ``h < 0``. ``h`` must be nonzero. Both accept a scalar, which is applied to every line, or a
sequence with one value per line, e.g. fitted peak amplitudes plus an offset. ``Quantity`` values
are used as their bare values and are never converted, so ``y`` and ``h`` must share units.


.. plot:: ../examples/plotting/spectra/plot_spectral_markers_basic.py
    :include-source: True


The connector and the label leader are controlled by:

- ``connector=False`` draws only the prongs. With a scalar ``x`` this gives a single prong
  with a label.
- ``connector_extend`` extends the connector beyond the outermost prongs, in x data units.
  A scalar extends both sides and a tuple is ``(left, right)``. It is ignored if
  ``connector=False``. The connector is drawn only if its length is positive, so a single line
  has no connector unless it is extended.
- ``stem`` adds a vertical segment from the connector to the label, in y data units. It points
  away from the tips (the direction of ``sign(h)``) and is drawn at the label anchor, so it
  follows ``label_loc``. It applies to labeled markers only and is drawn only where a connector
  is drawn, so a single line needs a positive connector_extend


.. plot:: ../examples/plotting/spectra/plot_spectral_markers_connector.py
    :include-source: True


Disabling Grouped Markers
^^^^^^^^^^^^^^^^^^^^^^^^^

``group=False`` disables grouping, so every line is drawn as its own marker and is labeled
individually, even when labels repeat.

This is useful if you have individual peak heights of the spectra from fitting. With ``group=False``:

- Each line gets its own prong, label and, if ``connector_extend`` is given, its own connector
  bar. Pass a sequence to configure each marker independently.
- Repeated labels are drawn once per line.
- ``stem`` has no effect, because it applies only to markers with more than one line.
  Increase ``h`` to raise the label instead.
- ``color``, ``linewidth``, ``linestyle`` and ``alpha`` apply per line. VisualAstro
  :ref:`colorsets <colorset-guide>` are supported.


.. plot:: ../examples/plotting/spectra/plot_spectral_markers_ungrouped.py
    :include-source: True


Label Placement
^^^^^^^^^^^^^^^

Label text placement is further controlled by:

- ``ha`` and ``va`` select which point of the text bounding box is pinned to the anchor.
  For example, ``va='top'`` makes the text hang below the anchor.
- ``rotation`` sets the label angle in degrees (counter-clockwise).
- ``rotation_mode`` sets the order of rotation and alignment.
  With ``'default'``, the text is rotated first and the rotated bounding box is then
  aligned via ``ha`` and ``va``. With ``'anchor'``, the text is aligned first and then
  rotated about the pinned point, so the pin stays fixed at any angle.
- ``label_offset_points`` is a signed ``(dx, dy)`` shift in points, applied after anchoring.
  Positive ``dx`` moves the label right and positive ``dy`` moves it up, regardless of ``ha``,
  ``va`` or ``rotation``.


See Also
^^^^^^^^

- :func:`plot_spectrum`
- :func:`spectral_elbow_markers`
- :func:`spectral_vlines`
