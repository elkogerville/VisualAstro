Installation
============

Requirements
------------

- Python >= 3.10
- Use a virtual environment (``conda`` or ``venv``).

Required Dependencies
---------------------

**conda**

.. code-block:: bash

    conda install -c conda-forge astropy matplotlib numpy scipy
    pip install --no-deps visualastro

**pip** (dependencies are installed automatically)

.. code-block:: bash

    pip install visualastro

Optional Dependencies
---------------------

VisualAstro supports several optional add-on packages.

**conda**

.. code-block:: bash

    conda install -c conda-forge cmasher colorspacious dust_extinction regions reproject scienceplots spectral-cube specutils tqdm
    pip install tol-colors

**pip**

.. code-block:: bash

    pip install cmasher colorspacious dust_extinction regions reproject scienceplots spectral-cube specutils tol-colors tqdm

.. note::

    ``tol-colors`` is not available on conda-forge.

Color Utilities
~~~~~~~~~~~~~~~

- ``cmasher``: Additional colormaps.
- ``colorspacious``: Color blindness simulation and color space utilities.
- ``tol-colors``: Paul Tol colorsets and colormaps.

Science Utilities
~~~~~~~~~~~~~~~~~

- ``dust_extinction``: Dust extinction models.
- ``regions``: Astropy Regions support.
- ``reproject``: Astropy Reproject support.
- ``spectral-cube``: Spectral cube support.
- ``specutils``: Spectra analysis support.
- ``tqdm``: Progress bars.

Styles
~~~~~~

- ``scienceplots``: Scientific matplotlib stylesheets. Requires a LaTeX installation.
