Documenting VisualAstro
=======================

Requirements
------------

.. code-block:: bash

   pip install -e ".[docs]"

Requires Python 3.11 and the packages listed under the ``docs`` extra in ``pyproject.toml``.

Build locally
-------------

To create a clean build for scratch:

.. code-block:: bash

   cd docs
   rm -rf _build reference/generated
   make html
   open _build/html/index.html

- For docstring or file content edits only: ``make html``.
- For new pages, renamed objects, ``conf.py`` or template changes: clean build as above.
- The build must finish with 0 warnings.

Document a new function or class
--------------------------------

1. Export it in ``src/visualastro/__init__.py``.
2. Write a NumPy-style docstring (see :ref:`docstring-rules`).
3. Add its bare name to an ``autosummary`` block in the matching
   ``docs/reference/<topic>.rst``:

   .. code-block:: rst

      .. currentmodule:: visualastro

      .. autosummary::
         :nosignatures:

         load_fits

4. List each object in exactly one page. Duplicates raise warnings.
5. Rebuild and fix all warnings.


Add a reference page
--------------------

1. Create ``docs/reference/<topic>.rst`` with a title, a
   ``.. currentmodule:: visualastro`` line and one or more ``autosummary``
   blocks grouped by purpose.
2. Add ``<topic>`` to the toctree in ``docs/reference/index.rst``.
3. Run a clean build.


Add a narrative page (guide or tutorial)
----------------------------------------

1. Create the ``.rst`` file in ``docs/`` or ``docs/tutorials/``.
2. List it, without extension, in the nearest ``toctree``.
3. Store images in ``docs/_static/images/`` and reference them as
   ``/_static/images/<name>.png`` with an ``:alt:`` entry.

.. _docstring-rules:

Docstring rules
---------------

Template for a function:

.. code-block:: python

   def func(
       arg1: dtype1 | dtype2,
       arg2: dtype1 | None = None,
       *args: tuple,
       **kwargs
   ):
       r"""
       One-line summary.

       Extended description (optional).

       Parameters
       ----------
       arg1 : dtype1 | dtype2
           Description.
       arg2 : dtype, optional, default=None
           Description.
       \*args : tuple
           Description. If referenced, should be written
           as ``*args`` to avoid an unterminated `*`.
       \*\*kwargs : dict
           Description.

       Returns
       -------
       dtype
           Description.

       Raises
       ------
       ErrorType
           Condition.

       See Also
       --------
       other_func : Short description.

       Notes
       -----
       Caveats.

       Examples
       --------
       >>> import visualastro as va
       """

- NumPy convention. Multiple dtypes separated by ``|``:
  ``var : dtype1 | dtype2, optional, default=value``.
- Blank line before every section header
  (``Parameters``, ``Returns``, ``See Also``, ``Notes``, ``Examples``).
- Blank line before and after every list. Wrapped list lines align with the
  text after the bullet.
- Do not add ``Methods`` or ``Attributes`` sections to classes. Members are
  documented from their own docstrings.
- Literal code in double backticks: ````reshape(*shape)````.
- Escape variadic parameter headers and use a raw docstring:
  ``\*args : tuple``, ``\*\*kwargs : dict``, with ``r"""`` as the opening
  delimiter.
- Cross-references: ``:func:`~visualastro.load_fits```,
  ``:class:`~visualastro.DataCube```,
  ``:class:`~astropy.units.Quantity``` (intersphinx).
- Examples use synthetic data so they run without local files.

Pre-commit checklist
--------------------

- Clean local build with 0 warnings.
- ``docs/_build/`` and ``docs/reference/generated/`` not committed.
- New files listed in a toctree.
- Images committed to ``docs/_static/``.
