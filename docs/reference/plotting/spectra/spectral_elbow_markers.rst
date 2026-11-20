spectral_elbow_markers
======================

.. currentmodule:: visualastro

.. autofunction:: spectral_elbow_markers


Overview
--------

Labels known spectral lines with one elbow marker per line. Pass line positions
and labels as parallel sequences. Each marker is a path from the tip at ``y`` to
the label anchor at ``y + h``, with the label displaced in x by ``label_shift``.
Use it to separate crowded labels that cannot share a connector.


Marker Geometry
^^^^^^^^^^^^^^^

``y`` and ``h`` are required. ``y`` is the data coordinate of the tips and ``h`` is the
signed marker height: the label anchor sits at ``y + h``, above the tips for ``h > 0`` and
below for ``h < 0``. ``h`` must be nonzero. Both accept a scalar, applied to every line, or a
sequence with one value per line. ``Quantity`` values are used as their bare values and are
never converted, so ``y`` and ``h`` must share units.

The path is controlled by:

- ``label_shift`` is the signed x displacement of the label anchor, in x data units. It sets
  the length of the horizontal run. ``None`` means no shift, giving a straight vertical line.
- ``elbow_frac`` is the fraction of ``h`` at which the horizontal bar sits, in [0, 1].
  Scalar or one value per line.
- ``elbow_frac=1`` gives a half-elbow (L-shaped) marker: the bar sits at ``y + h`` and the
  label is placed beside its end instead of above it.
- Increase ``|h|`` across neighboring lines to stagger labels.


.. plot:: ../examples/plotting/spectra/plot_spectral_elbow_markers_basic.py
    :include-source: True


Half-Elbow Markers
^^^^^^^^^^^^^^^^^^

With ``elbow_frac=1`` and nonzero ``label_shift``, label defaults change:

- ``ha`` is ``'left'`` for positive shift and ``'right'`` for negative.
- ``va`` is ``'center'``.
- ``label_offset_points`` is ``(sign(label_shift) * half_elbow_gap, 0)``.
- ``half_elbow_gap`` sets the gap in points between the run and the label. Magnitude only.
- Explicit ``ha``, ``va`` and ``label_offset_points`` take precedence.
- Full and half elbows can be mixed in one call by passing ``elbow_frac`` as a sequence.


.. plot:: ../examples/plotting/spectra/plot_spectral_elbow_markers_types.py
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
- :func:`spectral_markers`
- :func:`spectral_vlines`
