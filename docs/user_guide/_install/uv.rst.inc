uv is a fast Python package and project manager.
For more information, including how to install it, please refer to the `uv documentation <https://docs.astral.sh/uv/>`__.

Once we have set up our project with uv, we can simply run:

.. code-block:: bash

   uv add visualastro

and VisualAstro will be available for this project.

Alternatively, we can also use uv without a project.
This would be very similar to using venv, except we can create a venv with:

.. code-block:: bash

   uv venv

To install VisualAstro with only the required dependencies:

.. code-block:: bash

    uv pip install visualastro

If we want to install VisualAstro with the optional dependencies, run the following command instead:

.. code-block:: bash

    uv pip install visualastro[<group>]

and replace ``<group>`` with a group of optional dependencies you want to install. The options are:
- ``images``: Includes ``regions`` and ``reproject``.
- ``spectralcube``: Includes ``spectral_cube``, ``regions`` and ``reproject``.
- ``spectra``: Includes ``specutils`` and ``dust_extinction``.
- ``styles``: Includes ``scienceplots``.
- ``utils``: Includes ``cmasher``, ``colorspacious``, ``tol-colors``, and ``tqdm``.

You can also use ``all`` to install all optional dependencies, or install multiple groups by separating with commas.
