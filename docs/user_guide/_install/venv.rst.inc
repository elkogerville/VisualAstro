venv is Python's built-in, lightweight virtual environment solution.
It already comes with Python, so you don't need to download anything extra.
For more information on how to use venv, see the `venv documentation <https://docs.python.org/3/library/venv.html>`__.

After we create a venv and activate it, we can directly install VisualAstro.
Unlike installing with Conda, we don't have to install VisualAstro as the last step, since everything is installed with pip.

To install VisualAstro with only the required dependencies:

.. code-block:: bash

   pip install visualastro

If we want to install VisualAstro with the optional dependencies, run the following command instead:

.. code-block:: bash

   pip install visualastro[<group>]

and replace ``<group>`` with a group of optional dependencies you want to install. The options are:

- ``images``: Includes ``regions`` and ``reproject``.

- ``spectralcube``: Includes ``spectral_cube``, ``regions`` and ``reproject``.

- ``spectra``: Includes ``specutils`` and ``dust_extinction``.

- ``styles``: Includes ``scienceplots``.

- ``utils``: Includes ``cmasher``, ``colorspacious``, ``tol-colors``, and ``tqdm``.

You can also use ``all`` to install all optional dependencies, or install multiple groups by separating with commas.
