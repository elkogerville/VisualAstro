from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import visualastro as va


plt.style.use(['full', 'cm'])

file = 'spec1d_DE.20110603.45055-n7006a_DEIMOS_2011Jun03T123053.021.fits'
with fits.open(file) as hdul:

    SPAT135 = hdul[135]
    SPAT135w = SPAT135.data['OPT_WAVE']
    SPAT135f = SPAT135.data['OPT_COUNTS']

va.plot(
    SPAT135w*u.AA, SPAT135f*u.erg*u.cm**-2*u.s**-1,
    color='darkslateblue',
    figsize=(12,4),
    xlim=(8300, 8800),
    label='NGC 7006',
    ylabel='Flux',
    show_physical_type=True
)


va.spectral_lines(
    [8318, 8323, 8378, 8488.6, 8532.44, 8652.52, 8679], y=0.4e4,
    labels=[None, None, None, 'Ca II', 'Ca II', 'Ca II', None],
    label_anchor='left', label_offset_points=[6, 0],
    color='k',
    unlabeled_color='grey',
    lw=1
)
plt.show()
