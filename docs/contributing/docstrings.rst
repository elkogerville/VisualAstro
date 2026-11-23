.. _docstring-rules:

Docstring Rules
===============

- Use numpydoc convention. See :ref:`template docstring <template-docstring>`.

- For literal text, use double backticks, e.g. ````var````.

- Cross-references: use explicit roles with the target in single backticks, e.g. :func:`~visualastro.load_fits`.

  - Roles: ``:func:``, ``:class:``, ``:meth:``, ``:attr:``, ``:mod:``.
  - Use fully qualified names for external objects (``astropy.units.Quantity``, ``numpy.ndarray``).
  - Use ``~`` to omit the full path name.

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
    ``Attributes``, ``Raises``, ``See Also``, ``Notes``, ``Examples``. Do not
    document anything that already has its own docstring.
  - ``Attributes``: only plain instance attributes that autodoc cannot see (set in
    ``__init__``, no docstring of their own).
  - Members (methods, properties, class-level attributes) document themselves.
    Do not list or describe them in the class docstring; this causes duplicate-object warnings and drift.

- Public members: every public method and property needs at least a one-line docstring.

- Private functions (``_name``): not rendered by autodoc, so the rules are advisory.

  - At minimum, write a one line sentence for standalone private functions.
  - Keep the Numpy section layout and type format.


.. _template-docstring:

Template Docstring
------------------

.. include:: _template_docstring.rst.inc
