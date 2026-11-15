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

    >>> import astropy.units as u
    >>> import visualastro as va
    >>> region = va.spectral_region((8.3, 9.8) * u.um)
    >>> region
    Spectral Region, 1 sub-regions:
      (8.3 um, 9.8 um)

From a tuple of ``Quantity`` bounds:

.. code-block:: python

    >>> import astropy.units as u
    >>> import visualastro as va
    >>> region = va.spectral_region((8.3*u.um, 9.8*u.um))
    >>> region
    Spectral Region, 1 sub-regions:
      (8.3 um, 9.8 um)


Multiple regions
^^^^^^^^^^^^^^^^

From a list of (lower, upper) tuples sharing one unit:

.. code-block:: python

    >>> import astropy.units as u
    >>> import visualastro as va
    >>> region = va.spectral_region([(8.3, 9.8), (10.2, 11.3)] * u.um)
    >>> region
    Spectral Region, 2 sub-regions:
      (8.3 um, 9.8 um)     (10.2 um, 11.3 um)

From a list of tuples of ``Quantity`` bounds:

.. code-block:: python

    >>> import astropy.units as u
    >>> import visualastro as va
    >>> region = va.spectral_region([(8.3*u.um, 9.8*u.um), (10.2*u.um, 11.3*u.um)])
    >>> region
    Spectral Region, 2 sub-regions:
      (8.3 um, 9.8 um)     (10.2 um, 11.3 um)


Converting the Region Unit
^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass in a astropy unit to convert the region unit.

.. code-block:: python

    >>> import astropy.units as u
    >>> import visualastro as va
    >>> region = va.spectral_region([(8.3, 9.8), (10.2, 11.3)] * u.um, unit=u.AA)
    >>> region
    Spectral Region, 2 sub-regions:
      (83000.0 Angstrom, 98000.0 Angstrom)
      (102000.0 Angstrom, 113000.0 Angstrom)
