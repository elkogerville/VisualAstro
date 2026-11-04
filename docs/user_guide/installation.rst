.. _installation:

Installation Guide
============

You should always use a virtual environment or package manager when working with Python packages.
Here, we provide detailed instructions on how to install VisualAstro using three popular tools: Conda, venv, and uv.

For more information on the requirements and dependent packages of VisualAstro, see :doc:`req-dep`.


Install with Conda
------------

VisualAstro is not yet available on a conda channel yet, so we have to install it with pip.
We should install as many packages as possible with conda first, and only install VisualAstro as the last step.
Once we install packages with pip in a conda environment, it is generally a bad idea to go back to
installing packages with conda.

If you do not have conda installed, we recommend installing `miniforge <https://conda-forge.org/download/>`__.

First, create an environment with conda and activate it::
   conda create -n name_of_your_env -c conda-forge python
   conda activate name_of_your_env

Then, install the required dependencies::
   conda install -c conda-forge astropy matplotlib numpy scipy

Optionally, we can also install optional dependencies. The command below installs all optional dependencies::
   conda install -c conda-forge cmasher colorspacious dust_extinction regions reproject scienceplots spectral-cube specutils tol-colors tqdm

Next, we should also install other package we might use together with VisualAstro.
For example, if we want to use Jupyter and dynesty::
   conda install -c conda-forge jupyterlab dynesty

Finally, install VisualAstro::
   pip install --no-deps visualastro


.. _venv:

Install in a venv
------------

venv is Python's built-in, lightweight virtual environment solution.
It already comes to Python, so you don't need to download anything extra.
For more information on how to use venv, see the `venv documentation <https://docs.python.org/3/library/venv.html>`__.

After we create a venv and activate it, we can directly install VisualAstro.
Unlike installing with Conda, we don't have to install VisualAstro as the last step, since everything is installed with pip.
To install VisualAstro with only the required dependencies::
   pip install visualastro

If we want to install VisualAstro with the optional dependencies, run the following command instead::
   pip install visualastro[<group>]
and replace ``<group>`` with a group of optional dependencies you want to install. The options are:
- ``images``: Includes ``regions`` and ``reproject``.
- ``spectralcube``: Includes ``spectral_cube``, ``regions`` and ``reproject``.
- ``spectra``: Includes ``specutils`` and ``dust_extinction``.
- ``styles``: Includes ``scienceplots``.
- ``utils``: Includes ``cmasher``, ``colorspacious``, ``tol-colors``, and ``tqdm``.

You can also use ``all`` to install all optional dependencies, or install multiple groups by separating with commas.


Install with uv
------------

uv is a fast Python package and project manager.
For more information, including how to install it, please refer to the `uv documentation <https://docs.astral.sh/uv/>`__.

Once we have set up our project with uv, we can simply run::
   uv add visualastro
and VisualAstro will be available for this project.

Alternatively, we can also use uv without a project.
This would be very similar to using venv, except we can create a venv with::
   uv venv
and follow the instructions above for :ref:`venv`, but replace ``pip`` with ``uv pip`` in all commands.


Install from Source
------------

Instead of ``pip install visualastro``, run::
   pip install git+https://github.com/elkogerville/VisualAstro.git
