Writing Tests
=============

Tests use `pytest <https://docs.pytest.org>`__. Install the test dependencies
with:

.. code-block:: bash

   pip install -e ".[test]"

Run the full suite from the repository root:

.. code-block:: bash

   pytest -vs


Conventions
-----------

- Group the tests of one function or class in a ``Test<Name>`` class with a
  one-line docstring.
- Write each test as a method with a descriptive name following this format: ``test_<behavior>``.

Example:

.. code-block:: python

   class TestGetImshowNorm:
       """Tests for `get_imshow_norm`."""

       def test_none_norm(self):
           assert get_imshow_norm(None, 0.0, 2.0) is None
           assert get_imshow_norm(None, None, None) is None

       def test_supported_norms(self):
           assert isinstance(get_imshow_norm('log', 0.1, 9), LogNorm)
           assert isinstance(get_imshow_norm('power', 0.1, 9), PowerNorm)
