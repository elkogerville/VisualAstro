spectral_vlines
===============

.. currentmodule:: visualastro

.. autofunction:: spectral_vlines


Overview
--------

Plot a spectrum and mark known emission lines with labeled vertical lines.
Pass the line positions and labels as parallel sequences, and use ``y`` to anchor the
labels at a data coordinate just above the spectrum.


Label Placement
^^^^^^^^^^^^^^^

Pass the line positions and labels as parallel sequences. By default (``y=None``) the
labels are anchored at the top of the axes, independent of the data limits.


.. plot:: ../examples/plotting/spectra/plot_spectral_vlines_basic.py
    :include-source: True


Pass ``y`` to anchor the labels at data coordinates instead. A scalar is applied to every line,
and a sequence gives one anchor per line, e.g. fitted peak amplitudes. A ``None`` entry in the
sequence falls back to the top of the axes for that line.

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


.. plot:: ../examples/plotting/spectra/plot_spectral_vlines_labels.py
    :include-source: True


Changing the Color and Style
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``color`` accepts a single color, a VisualAstro colorset, or a ``list`` of colors.
A colorset, or a ``list`` whose length differs from the number of lines, is a palette
assigned to the unique labels in order of first appearance. It is cycled or cropped as
needed, so lines with identical labels always share a color. A ``list`` with exactly one
color per line is used as given, without grouping. The label text inherits the color of
its line.

Unlabeled lines (``None``, ``''`` or ``'none'``) are drawn in ``unlabeled_color`` when a
colorset or palette is used.

In this example we plot 8 lines with 4 unique labels and one blank label.


.. plot:: ../examples/plotting/spectra/plot_spectral_vlines_colors.py
    :include-source: True

Other line properties such as ``linewidth``, ``linestyle`` and ``alpha`` accept a single
value or a ``list`` with one value per line.


See Also
^^^^^^^^

- :func:`plot_spectrum`
- :func:`spectral_elbow_markers`
- :func:`spectral_markers`
