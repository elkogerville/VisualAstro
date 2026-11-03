Documenting VisualAstro
=======================


Requirements
------------

Make sure VisualAstro is installed via a developer install by navigating into the
package root:

.. code-block:: bash

    cd visualastro/
    pip install -e ".[docs]"

Requires Python 3.11 and the packages listed under the ``docs`` extra in ``pyproject.toml``.


Building Locally
----------------

To create a clean build from scratch:

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
2. Write a NumPy-style docstring (see :ref:`docstring rules <docstring rules>`).
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


.. _docstring rules:

Docstring Rules
---------------

- Use numpydoc convention. See :ref:`template docstring <template docstring>`.

- For literal text, use double backticks, e.g. ````var````.

- Cross-references: use explicit roles with the target in single backticks, e.g. :func:`~visualastro.load_fits`.

  - Roles: ``:func:``, ``:class:``, ``:meth:``, ``:attr:``, ``:mod:``.
  - Use fully qualified names for external objects (``astropy.units.Quantity``, ``numpy.ndarray``).

- Variadics:

  - In ``Parameters``: write ``*args`` and ``**kwargs``.
  - In prose: use double backticks (e.g. ````reshape(*args)````). Bare ``*args`` is parsed as emphasis.

- Types: ``dtype1 | dtype2, optional, default=value``.

  - Use resolvable names.
  - ``optional`` means the parameter has a default value.

- Blank lines: one before and after every list, directive, and code block, including between a list in a parameter description and the next parameter. Omit only at the start or end of the docstring.

- Directives (``.. note::``, ``.. warning::``, ``.. math::``): blank line after the directive line, body indented.

- Use ``r"""`` only if the docstring contains backslashes (e.g. ``:math:``). Otherwise use ``"""``.

- Sections: use only napoleon section titles, in this order: summary, extended description, ``Parameters``, ``Returns``, ``Raises``, ``Warns``, ``See Also``, ``Notes``, ``Examples``.

- Classes:

  - The class docstring covers the class as a whole: summary, extended description,
    ``Parameters`` (arguments of ``__init__``, which gets no docstring of its own),
    ``Attributes``, ``Raises``, ``See Also``, ``Notes``, ``Examples``. Do not add
    document anything that already has a docstring.
  - ``Attributes``: only plain instance attributes that autodoc cannot see (set in
    ``__init__``, no docstring of their own).
  - Members (methods, properties, class-level attributes) document themselves.
    Do not list or describe them in the class docstring; this causes duplicate-object warnings and drift.

- Public members: every public method and property needs at least a one-line docstring.

- Private functions (``_name``): not rendered by autodoc, so the rules are advisory.

  - At minimum, write a one line sentence for standalone private functions.
  - Keep the numpy section layout and type format

.. _template docstring:

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

        - :func:`~visualastro.set_limits`
        - :class:`~visualastro.Colorsets`
        - :meth:`~visualastro.Colorsets.plot`
        - :class:`~astropy.units.Quantity`

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
        *data : numpy.ndarray | astropy.units.Quantity
            One or more input arrays. Each can be either:

            - ``numpy.ndarray``
            - ``astropy.units.Quantity``

        axis : int, optional, default=0
            Axis to operate on ``data``.
        unit : str | astropy.units.UnitBase | None, optional, default=None
            Output unit.
        **kwargs : dict
            Forwarded to :meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        astropy.units.Quantity
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
        astropy.units.Quantity : Array with attached units.

        Notes
        -----
        - Implementation details and caveats.

        Examples
        --------
        >>> import visualastro as va
        >>> out = va.func(data)
        """


Pre-Commit Checklist
--------------------

- Clean local build with 0 warnings.
- ``docs/_build/`` and ``docs/reference/generated/`` not committed.
- New files listed in a toctree.
- Images committed to ``docs/_static/``.
