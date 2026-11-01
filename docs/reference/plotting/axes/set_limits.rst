set_limits
==========

.. currentmodule:: visualastro

.. autofunction:: set_limits


Examples
--------

Symmetric limits
^^^^^^^^^^^^^^^^

In addition to ``xlim`` and ``ylim``, shared limits for both axes can be set with ``limits``.

- Use a ``tuple`` ``(min, max)`` to set ``xlim=(min, max)`` and ``ylim=(min, max)``.
- Use a single ``float`` ``value`` to set symmetric limits ``xlim=ylim=(-value, value)``.


.. literalinclude:: /../examples/plotting/axes/plot_set_limits_symmetric.py
    :language: python


.. image:: /_static/images/reference/plotting/set_symmetric_limits.png
    :scale: 50 %
    :alt: effect of recompute_other_axis in set_limits
    :align: center


Window Clipping
^^^^^^^^^^^^^^^

If `recompute_other_axis=True`, and `xlim` is passed without `ylim` (or vice-versa),
VisualAstro will apply 1D box clipping onto the other axis. This rescales the unset
axis by only taking into account the portion of the data visible in the window set by
limit passed in. The effects are illustrated in the plot below.


.. literalinclude:: /../examples/plotting/axes/plot_set_limits_recompute.py
    :language: python


.. image:: /_static/images/reference/plotting/recompute_limits.png
    :scale: 50 %
    :alt: effect of recompute_other_axis in set_limits
    :align: center
