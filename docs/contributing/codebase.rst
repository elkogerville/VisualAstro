Contributing to the Codebase
============================


Coding Style
------------

Quote conventions:

.. code-block:: python

   name = 'spectrum'             # regular string
   label = f"{name}: {value}"    # f-string

   def func():
       """Docstring."""          # docstring

Docstring style:

- Use a numpy docstring for all public functions (see :doc:`docstrings`).
  You can also see a template docstring :ref:`here <template-docstring>`.

We also lint our code using `ruff <https://docs.astral.sh/ruff/>`__.


New Functions and Classes
-------------------------

Every new public function or class requires all of the following:

1. Export it in ``src/visualastro/__init__.py``.
2. Write a NumPy-style docstring (see :doc:`docstrings`).
3. Add tests (see :doc:`tests`).
4. Register it in the reference docs (see :doc:`documentation`) at the minimum.
   It would also nice to add some examples! If you would like to add documentation,
   but don't know how, see (see :doc:`documentation`)! Or, reach out and the VisualAstro
   team will be happy to help you write the docs.


Submitting Changes
------------------

See :doc:`pull_requests` for the branch and pull request workflow.
