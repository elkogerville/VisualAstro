.. doctest-skip-all::


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
