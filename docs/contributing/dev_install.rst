.. _dev-install:

Development Install
===================

Contributing requires `git <https://git-scm.com/downloads>`__ and, to open
pull requests, a `GitHub <https://github.com>`__ account.


Fork and Clone
--------------

First, `fork <https://github.com/elkogerville/VisualAstro/fork>`__ the repository
on GitHub, then clone your fork and add the original repository as ``upstream``:

.. code-block:: bash

   git clone https://github.com/<your-username>/VisualAstro.git
   cd VisualAstro
   git remote add upstream https://github.com/elkogerville/VisualAstro.git


Editable Install
----------------

Install VisualAstro in editable mode from the package root:

.. code-block:: bash

    cd visualastro/
    pip install -e .

This installs only the core dependencies and is sufficient for editing the
source code.


Running Tests and Building the Docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optional dependencies are not installed by default, but if you would
like to run the tests or build the docs locally, you should instead run one of the
following commands:

.. code-block:: bash

    pip install -e ".[tests]"
    pip install -e ".[docs]"
    pip install -e ".[dev]"

``dev`` installs all optional dependencies, as well as ``tests`` and ``docs``.
All extras are defined in ``pyproject.toml``.
