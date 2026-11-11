set_limits
==========

.. currentmodule:: visualastro

.. autofunction:: set_limits


Examples
--------

Window Clipping
^^^^^^^^^^^^^^^

If ``recompute_other_axis=True``, and ``xlim`` is passed without ``ylim`` (or vice-versa),
VisualAstro will apply 1D box clipping onto the other axis. This rescales the unset
axis by only taking into account the portion of the data visible in the window set by
limit passed in. The effects are illustrated in the plot below.


.. literalinclude:: /../examples/plotting/axes/plot_set_limits_recompute.py
    :language: python


.. image:: /_static/images/reference/plotting/recompute_limits.png
    :scale: 50 %
    :alt: effect of recompute_other_axis in set_limits
    :align: center


Symmetric Limits
^^^^^^^^^^^^^^^^

In addition to ``xlim`` and ``ylim``, shared limits for both axes can be set with ``limits``.
The window is always interpreted relative to ``center`` (the origin if ``center`` is ``None``).

- Use a ``tuple`` ``(lo, hi)`` to set ``xlim=(center + lo, center + hi)`` and
  ``ylim=(center + lo, center + hi)``. With the default center this is
  ``xlim=ylim=(lo, hi)``. If ``lo > hi`` both axes are inverted.
- Use a single ``float`` ``value`` to set a symmetric window
  ``xlim=ylim=(center - |value|, center + |value|)``.

``limits`` cannot be combined with ``xlim`` or ``ylim``, and a zero-width window raises a ``ValueError``.
Symmetric limits require linear axes, since a shifted window is not meaningful on a log scale.
Axes that are already inverted (e.g. RA or magnitude) stay inverted unless the window itself is inverted.


.. plot:: ../examples/plotting/axes/plot_set_limits_symmetric.py
    :include-source: True


Centering the Window
********************

The window can be moved with ``center``, which requires ``limits``.
A scalar centers both axes at the same value, and a ``tuple`` ``(cx, cy)`` sets each axis independently.

- ``center='origin'`` (the default) centers the window at ``(0, 0)``.
- ``center='com'`` centers the window on the center of mass of the plotted points, assuming unit masses unless ``weights`` are given.
- ``center='median'`` centers on the coordinate-wise median, which is more robust to outliers. ``weights`` are ignored.
- A callable ``fn(points, weights) -> (cx, cy)`` can be passed for any custom center, such as the densest point or the most massive particle.

Data-dependent modes read their points from ``points`` if provided, and otherwise from the
scatter and marker-only artists already drawn on the axes. Lines, patches, and filled regions
are excluded, since their vertex count would bias the result. Call ``set_limits`` after all
plotting calls, or pass ``points`` explicitly.


.. plot:: ../examples/plotting/axes/plot_set_limits_recenter_window.py
    :include-source: True


The ``points`` argument can be used to center around a subset of the data plotted.
Here we center around the center of mass of only the second group.


.. plot:: ../examples/plotting/axes/plot_set_limits_recenter_window_subset.py
    :include-source: True


Square Limits
^^^^^^^^^^^^^

Set ``square=True`` to make the numeric ranges of both axes equal.
The narrower axis is expanded about its own center, so no data is cut off, and inverted axes stay inverted.
This is ignored if ``limits`` is not ``None``.

Squaring is applied last, after ``xlim``, ``ylim``, padding, and any recomputed limits, so it composes with every other option.
If both ``xlim`` and ``ylim`` are given with different ranges, the narrower one is expanded and the
requested limits are not preserved exactly.

.. note::

    Equal ranges do not imply equal scaling on screen. On a non-square axes box, use
    ``ax.set_aspect('equal')`` for geometrically correct proportions.


.. plot:: ../examples/plotting/axes/plot_set_limits_square.py
    :include-source: True
