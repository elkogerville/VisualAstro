plot_spectrum
=============

.. currentmodule:: visualastro

.. autofunction:: plot_spectrum


Overview
--------

- Plots a :class:`specutils.Spectrum` (spectral axis vs. flux) on a matplotlib axis.
- Accepts 1D or 2D flux. For 2D flux, each row is drawn as a separate line.
- Returns the list of :class:`matplotlib.lines.Line2D` artists created.


.. warning::

    ``This requires the optional package specutils``. See :doc:`/user_guide/installation`.


Behavior
--------

Use ``plot_mode='step'`` for a step plot and ``plot_mode='plot'`` for a standard line plot.
By default, ``xpad=0`` to remove margin padding for the spectral axis (x axis). All remaining
``**kwargs`` are forwarded to the underlying plotting function.


Examples
--------

Step plot of a 1D spectrum:


.. plot:: ../examples/plotting/spectra/plot_spectrum.py
    :include-source: True
