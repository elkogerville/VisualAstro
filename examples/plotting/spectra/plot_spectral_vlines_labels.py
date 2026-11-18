import matplotlib.pyplot as plt

import visualastro as va
from visualastro.core.io import example_spectrum

SN1987A_N = example_spectrum()
emission_labels = ['[Ne II]', '[Ne V]', '[Ne III]', '[S III]']
line_locs = [12.81370507, 14.32125558, 15.55381827, 18.71226999]
amplitudes = [0.005332791, 0.00200568, None, None]

plt.style.use(['full', 'cm'])
va.config.show_type_label = True
va.config.show_unit_label = True

fig, ax = va.subplots(figsize=(13,5))
va.plot_spectra(
    SN1987A_N, alpha=0.8, color='darkslateblue',
    label='NOR', xlim=(12,19), ylim=(-0.001, 0.012)
)

va.spectral_vlines(
    line_locs, emission_labels, y=amplitudes,
    alpha=0.5,
    ls=':',
    va='top',
    ha='right',
    label_offset_points=(-5,-10),
    rotation=0
)


plt.show()
