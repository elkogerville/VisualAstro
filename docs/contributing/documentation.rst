Writing Documentation
=====================

.. warning::

    This page assumes you have installed visualastro with an
    editable install along with ``docs`` dependencies. If not,
    See :ref:`dev installation <dev-install>`


Building Locally
----------------

To create a clean build from scratch:

.. code-block:: bash

   cd docs
   make clean html
   open _build/html/index.html

A clean build is required after any of the following: adding a page, renaming
or removing objects, modifying a plot, or editing ``conf.py`` or the templates.

For docstring or file content edits only, it is sufficeint to run ``make html``.


Documenting a New Function or Class
-----------------------------------

1. Export it in ``src/visualastro/__init__.py``.
2. Write a NumPy-style docstring (see :doc:`docstrings`).
3. Create ``docs/reference/<topic>/<name>.rst``:

.. code-block:: rst

    set_limits
    ==========

    .. currentmodule:: visualastro

    .. autofunction:: set_limits


4. List the page, without extension, in the matching ``index.rst`` toctree.
5. Rebuild and fix all warnings.


Adding a Reference Page
-----------------------

1. Create ``docs/reference/<topic>.rst`` with a title, a
   ``.. currentmodule:: visualastro`` line and one or more ``autosummary``
   blocks grouped by purpose.
2. Add ``<topic>`` to the toctree in ``docs/reference/index.rst``.
3. Run a clean build.


Adding a Guide or Tutorial
--------------------------

1. Create the ``.rst`` file in ``docs/user_guide/``
2. List it, without extension, in the nearest ``toctree``.


Adding Plots to Documentation
-----------------------------

Plots are rendered dynamically at build time by the documentation.
Simply create a ``.py`` file in ``visualastro/examples/`` and write a script
that generates the figure. Make sure the script has all of its dependencies
imported and can run standalone. Then add the plot to the documentation page:

.. code-block:: rst

    .. plot:: ../examples/plotting/legends/<your new file>.py
        :include-source: True


Then run ``make clean html`` to rebuild the docs.
