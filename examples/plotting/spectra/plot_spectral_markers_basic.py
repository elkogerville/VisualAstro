import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models
from specutils import Spectrum

import visualastro as va

np.random.seed(42)
amplitudes = [1.6, 2.0, 1.5, 1.5, 1.2, 1.7, -1.1]
line_waves = [4634.140, 4640.640, 4641.850, 4647.420, 4651.470, 4711.260, 4740.120]
labels = ['N III', 'N III', 'N III', 'C III', 'C III', '[Ar IV]', '[Ar IV]',]
gaussians = [models.Gaussian1D(A, mu, 0.2) for (A, mu) in zip(amplitudes, line_waves)]

x = np.linspace(4620, 4760, 500)
y = (1.0 + 2e-4 * (x - 4500) + sum(g(x) for g in gaussians) + np.random.normal(0.0, 0.1, x.shape))
spectrum = Spectrum(flux=y * u.Jy, spectral_axis=x * u.AA)

# create figure

plt.style.use(['full', 'concrete-bold'])
fig, ax = va.subplots(figsize=(12,6))
va.plot_spectrum(spectrum, color='violetred', xlabel=r'Wavelength', ylabel=r'Flux')

# 3 prong marker
va.spectral_markers(line_waves[:3], labels[:3], y=2.7, h=0.2)
# 2 prong marker with a stem
va.spectral_markers(line_waves[3:5], labels[3:5], y=2.51, h=0.2, stem=0.1)
# single marker pointing down
va.spectral_markers(line_waves[5], labels[5], y=2.6, h=0.1)
# single marker pointing up
va.spectral_markers(line_waves[6], labels[6], y=0.0, h=-0.1)

va.set_limits(ylim=(-0.5, 3.35))

plt.show()
