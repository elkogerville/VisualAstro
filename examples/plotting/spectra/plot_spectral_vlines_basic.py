import matplotlib.pyplot as plt

import visualastro as va
from visualastro.core.io import example_spectrum

plt.style.use(['full', 'cm'])
va.config.show_type_label = True
va.config.show_unit_label = True
va.set_plot_cycle(colors=['darkslateblue', 'mediumvioletred'])

SN1987A_N = example_spectrum('SN1987A_north.fits')
SN1987A_S = example_spectrum('SN1987A_south.fits')
emission_labels = [
    '[Ar II]', '[Ne VI]', '[Ar III]', '[S IV]',
    '[Ne II]', '[Ne V]', '[Ne III]', '[S III]'
]
line_locs = [
    6.98524305,  7.65248305,  8.99228373, 10.51038726,
    12.81370507, 14.32125558, 15.55381827, 18.71226999
]

fig, ax = va.subplots(figsize=(13,5))
va.plot_spectra(SN1987A_N, alpha=0.8, label='NOR')
va.plot_spectra(SN1987A_S, alpha=0.8, label='SOR')

va.spectral_vlines(line_locs, emission_labels, y=0.013, alpha=0.5)

va.set_limits(ylim=(-0.003, 0.014))

plt.show()
