set_limits
==========

.. currentmodule:: visualastro

.. autofunction:: set_limits


Examples
--------

.. code-block:: python

    import astropy.units as u
    import matplotlib.pyplot as plt
    import visualastro as va

    plt.style.use(['minimal', 'stix'])

    file = 'spec1d_DE.20110603.45055-n7006a_DEIMOS_2011Jun03T123053.021.fits'
    with fits.open(file) as hdul:

        SPAT135 = hdul[135]
        SPAT135w = SPAT135.data['OPT_WAVE']
        SPAT135f = SPAT135.data['OPT_COUNTS']

    fig, ax = plt.subplots(1, 2, figsize=(13,4))

    va.config.fontsizes.size = 15
    va.config.show_type_label = True

    va.plot(
        SPAT135w*u.AA, SPAT135f*u.erg*u.cm**-2*u.s**-1,
        ax=ax[0],
        color='darkslateblue',
        label='NGC 7006',
        ylabel='Flux',
        recompute_other_axis=False,
        title='recompute_other_axis: False',

    )

    va.set_limits(ax=ax[0], xlim=(8200, 8400), recompute_other_axis=False)

    va.plot(
        SPAT135w*u.AA, SPAT135f*u.erg*u.cm**-2*u.s**-1,
        ax=ax[1],
        color='darkslateblue',
        label='NGC 7006',
        ylabel='Flux',
        recompute_other_axis=True,
        title='recompute_other_axis: True',
        xlim=(8200, 8400)
    )

    va.set_limits(ax=ax[1], xlim=(8200, 8400), recompute_other_axis=True)

    plt.show()


.. image:: /_static/images/reference/plotting/recompute_limits.png
    :scale: 50 %
    :alt: effect of recompute_other_axis in set_limits
    :align: center
