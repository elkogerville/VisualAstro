import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models
from specutils import Spectrum

import visualastro as va

np.random.seed(42)
amplitudes = [1.1, 1.4, 2.0, 1.5, 1.5, 1.2, 0.97]
line_waves = [4629.339, 4634.140, 4640.640, 4641.850, 4647.420, 4651.470, 4658.050]
labels = ['Fe II', 'N III', 'N III', 'N III', 'C III', 'C III', '[Fe III]']
y_vals = [2.33, 2.73, 3.15, 3.15, 2.6, 2.6, 2.2]
gaussians = [models.Gaussian1D(A, mu, 0.2) for (A, mu) in zip(amplitudes, line_waves)]

x = np.linspace(4625, 4665, 500)
y = (1.0 + 2e-4 * (x - 4500) + sum(g(x) for g in gaussians) + np.random.normal(0.0, 0.1, x.shape))
spectrum = Spectrum(flux=y * u.Jy, spectral_axis=x * u.AA)

# create figure

plt.style.use(['full', 'stix'])
fig, ax = va.subplots(figsize=(12,6))
va.plot_spectrum(spectrum, color='7 train', xlabel=r'Wavelength', ylabel=r'Flux')

# single prong with a symmetric connector extension
va.spectral_markers(
    line_waves[0], labels[0], y=y_vals[0], h=0.1,
    connector_extend=0.5,
)

# single prong with an asymmetric connector extension
va.spectral_markers(
    line_waves[1], labels[1], y=y_vals[1], h=0.1,
    connector_extend=(0, 1.5),
    ha='center',
)

# two-line group, prongs only
va.spectral_markers(
    line_waves[2:4], labels[2:4], y=y_vals[2:4], h=0.07,
    connector=False,
    stem=0.05,
)

# two-line group with a stem, plus one single line without stem
va.spectral_markers(
    line_waves[4:], labels[4:], y=y_vals[4:], h=0.07,
    stem=0.05,
)

va.set_limits(ylim=(0.5, 3.5))

plt.show()
