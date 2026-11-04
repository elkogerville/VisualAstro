.. _req-dep:

Requirements and Dependencies
============================

This page lists the requirements and dependencies of VisualAstro. For instructions on how to install VisualAstro, see :doc:`installation`.

Requirements
------------

- Python >= 3.11.18
- Use a virtual environment (``conda``, ``venv``, ``uv``, etc.)


Required Dependencies
---------------------

In order to install and use VisualAstro, you must have the following packages installed:

- `astropy <https://docs.astropy.org/en/stable/>`__: Library for astronomy.
- `Matplotlib <https://matplotlib.org/stable/>`__: Library for visualization.
- `NumPy <https://numpy.org/doc/stable/>`__: Scientific computation support.
- `SciPy <https://docs.scipy.org/doc/scipy/>`__: Mathematic and scientific algorithms.


Optional Dependencies
---------------------

VisualAstro supports several optional add-on packages.

Colors and Styles
~~~~~~~~~~~~~~~~~

VisualAstro registers these optional packages with Matplotlib when imported so that
you can use the additional colors and styles without importing them.

- `cmasher <https://github.com/1313e/CMasher>`__: Additional colormaps.
- `colorspacious <https://github.com/njsmith/colorspacious>`__: Color blindness simulation and color space utilities.
- `scienceplots <https://github.com/garrettj403/SciencePlots>`__: Scientific matplotlib stylesheets. Requires a LaTeX installation.
- `tol-colors <https://github.com/Descanonge/tol_colors>`__: Paul Tol colorsets and colormaps.

Science Utilities
~~~~~~~~~~~~~~~~~

- `dust_extinction <https://github.com/karllark/dust_extinction>`__: Dust extinction models.
- `regions <https://github.com/astropy/regions>`__: Astropy Regions support.
- `reproject <https://github.com/astropy/reproject>`__: Astropy Reproject support.
- `spectral-cube <https://github.com/radio-astro-tools/spectral-cube>`__: Spectral cube support.
- `specutils <https://github.com/astropy/specutils>`__: Spectra analysis support.
- `tqdm <https://github.com/tqdm/tqdm>`__: Progress bars.
