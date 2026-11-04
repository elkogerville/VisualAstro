
VisualAstro is not yet available on a conda channel yet, so we have to install it with pip.
We should install as many packages as possible with conda first, and only install VisualAstro as the last step.
Once we install packages with pip in a conda environment, it is generally a bad idea to go back to
installing packages with conda.

If you do not have conda installed, we recommend installing `miniforge <https://conda-forge.org/download/>`__.

First, create an environment with conda and activate it:

.. code-block:: bash

   conda create -n name_of_your_env -c conda-forge python
   conda activate name_of_your_env

Then, install the required dependencies:

.. code-block:: bash

   conda install -c conda-forge astropy matplotlib numpy scipy

Optionally, we can also install optional dependencies. The command below installs all optional dependencies:

.. code-block:: bash

   conda install -c conda-forge cmasher colorspacious dust_extinction regions reproject scienceplots spectral-cube specutils tol-colors tqdm

Next, we should also install other package we might use together with VisualAstro. For example, if we want to use Jupyter and dynesty:

.. code-block:: bash

    conda install -c conda-forge jupyterlab dynesty

Finally, install VisualAstro:

.. code-block:: bash

   pip install --no-deps visualastro
