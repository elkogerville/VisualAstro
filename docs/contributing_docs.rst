Documenting VisualAstro
=======================

Requirements
------------

.. code-block:: bash

   pip install -e ".[docs]"

Requires Python 3.11 and the packages listed under the ``docs`` extra in ``pyproject.toml``.

Build Locally
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

Documenting a New Function or Class
-----------------------------------

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


Adding a Reference Page
-----------------------

1. Create ``docs/reference/<topic>.rst`` with a title, a
   ``.. currentmodule:: visualastro`` line and one or more ``autosummary``
   blocks grouped by purpose.
2. Add ``<topic>`` to the toctree in ``docs/reference/index.rst``.
3. Run a clean build.


Adding a Guide or Tutorial
--------------------------

1. Create the ``.rst`` file in ``docs/`` or ``docs/tutorials/``.
2. List it, without extension, in the nearest ``toctree``.
3. Store images in ``docs/_static/images/`` and reference them as
   ``/_static/images/<name>.png`` with an ``:alt:`` entry.


.. _docstring-rules:

Docstring Rules
---------------

- Use Numpy convention. See template docstring below for an example.
- Use double backticks for literal text (i.e. ````var````)
- Use single backticks for cross-references (i.e. ``:func:`~visualastro.spectral_line_marker```)
- Escape variadic inputs (i.e. ``\*data`` or ``\*\*kwargs``)
- Put a blank line before and after every list, directive, and code block. Omit only at the start or end of a docstring.


Template Docstring
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def func(
        *data: NDArray | u.Quantity,
        axis: int = 0,
        *args: tuple,
        unit: str | u.UnitBase | None = None,
        **kwargs
    ):
        r"""
        One-line summary.

        Extended description. Cross-link objects inline:
        :func:`~visualastro.load_fits`, :class:`~visualastro.DataCube`
        :meth:`~visualastro.DataCube.to`
        :class:`~astropy.units.Quantity`

        Inline math: :math:`\sigma = \mathrm{FWHM} / 2.355`.

        Block math:

        .. math::

           F = \int_{\lambda_1}^{\lambda_2} f_\lambda \, d\lambda

        .. note::
           Short callout for the reader.

        .. warning::
           Behavior that can silently produce wrong results.

        Parameters
        ----------
        data : ndarray | astropy.units.Quantity
            Input array. Can be either

            - np.ndarray
            - astropy.units.Quantity
        axis : int, optional, default=0
            Axis to operate onto ``data``.
        \*args : tuple
            Passed to ````reshape(*args)````.
        unit : str | astropy.units.Unit | None, optional, default=None
            Output unit.
        \*\*kwargs : dict
            Forwarded to ``matplotlib.axes.Axes.plot``.

        Returns
        -------
        Quantity
            Description.

        Raises
        ------
        ValueError
            If ``axis`` is out of range.

        Warns
        -----
        UserWarning
            If ``data`` contains NaN.

        See Also
        --------
        load_fits : Load an image or cube.
        visualastro.DataCube : Cube container.

        Notes
        -----
        - Implementation details and caveats.

        Examples
        --------
        >>> import visualastro as va
        >>> out = va.func(data)
        """


Classes
*******

Use the above rules for class docstrings. Do not add ``Methods`` or ``Attributes`` sections to classes.
Members are documented from their own docstrings.


Pre-Commit Checklist
--------------------

- Clean local build with 0 warnings.
- ``docs/_build/`` and ``docs/reference/generated/`` not committed.
- New files listed in a toctree.
- Images committed to ``docs/_static/``.
