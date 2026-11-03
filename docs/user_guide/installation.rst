.. _installation:

Installation
============

Requirements
------------

- Python >= 3.10
- Use a virtual environment (``conda`` or ``venv``).


Required Dependencies
---------------------

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install visualastro

   .. tab-item:: conda

      .. code-block:: bash

         conda install -c conda-forge astropy matplotlib numpy scipy
         pip install --no-deps visualastro

   .. tab-item:: uv

      .. code-block:: bash

         uv pip install visualastro


Optional Dependencies
---------------------

VisualAstro supports several optional add-on packages.


.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install cmasher colorspacious dust_extinction regions reproject scienceplots spectral-cube specutils tol-colors tqdm

   .. tab-item:: conda

      .. code-block:: bash

         conda install -c conda-forge cmasher colorspacious dust_extinction regions reproject scienceplots spectral-cube specutils tqdm

   .. tab-item:: uv

      .. code-block:: bash

         uv pip install cmasher colorspacious dust_extinction regions reproject scienceplots spectral-cube specutils tol-colors tqdm

.. note::

    ``tol-colors`` is not available on conda-forge.


Colors and Styles
~~~~~~~~~~~~~~~~~

VisualAstro registers these optional packages with matplotlib when imported so they
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
