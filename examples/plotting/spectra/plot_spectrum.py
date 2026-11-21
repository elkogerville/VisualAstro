import matplotlib.pyplot as plt

import visualastro as va
from visualastro.core.io import example_spectrum

SN1987A_N = example_spectrum()

plt.style.use(['full', 'cm'])
va.config.show_type_label = True
va.config.show_unit_label = True

fig, ax = va.subplots(figsize=(13,5))
va.plot_spectrum(SN1987A_N, alpha=0.8, color='darkslateblue', label='NOR')

plt.show()
