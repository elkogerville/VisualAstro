spectral_region
===============

.. currentmodule:: visualastro

.. autofunction:: spectral_region


Examples
--------

Create a :class:`~specutils.SpectralRegion` from several input forms.


Single region
^^^^^^^^^^^^^

From a ``Quantity`` tuple of (lower, upper) bounds:

.. code-block:: python

    import astropy.units as u
    import visualastro as va

    va.spectral_region((8.3, 9.8) * u.um)

From a tuple of ``Quantity`` bounds:

.. code-block:: python

    import astropy.units as u
    import visualastro as va

    va.spectral_region((8.3*u.um, 9.8*u.um))


Multiple regions
^^^^^^^^^^^^^^^^

From a list of (lower, upper) tuples sharing one unit:

.. code-block:: python

    import astropy.units as u
    import visualastro as va

    va.spectral_region([(8.3, 9.8), (10.2, 11.3)] * u.um)

From a list of tuples of ``Quantity`` bounds:

.. code-block:: python

    import astropy.units as u
    import visualastro as va

    va.spectral_region([(8.3*u.um, 9.8*u.um), (10.2*u.um, 11.3*u.um)])
