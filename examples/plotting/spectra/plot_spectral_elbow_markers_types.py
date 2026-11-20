import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models
from specutils import Spectrum

import visualastro as va

np.random.seed(0)
amplitudes = [1.1, 1.4, 2.0, 1.5, 1.5, 1.2, 0.97]
line_waves = [4629.339, 4634.140, 4640.640, 4641.850, 4647.420, 4651.470, 4658.050]
labels = ['Fe II', 'N III', 'N III', 'N III', 'C III', 'C III', '[Fe III]']
gaussians = [models.Gaussian1D(A, mu, 0.2) for (A, mu) in zip(amplitudes, line_waves)]

x = np.linspace(4625, 4665, 500)
y = (
    1.0 + 2e-4 * (x - 4500)
    + sum(g(x) for g in gaussians)
    + np.random.normal(0.0, 0.05, x.shape)
)
spectrum = Spectrum(flux=y * u.Jy, spectral_axis=x * u.AA)

# calculate peak height
y_vals = np.interp(line_waves*u.AA, spectrum.spectral_axis, spectrum.flux) + 0.03*u.Jy

plt.style.use(['full', 'libertinus'])
va.config.show_type_label = True
fig, ax = va.subplots(figsize=(12,5))
va.plot_spectrum(spectrum, figsize=(12,6), color='aurora green')

va.spectral_elbow_markers(
    line_waves, labels, y=y_vals,
    h=0.11,
    label_shift=[0, 0.6, -0.5, 0.4, -0.9, 1, 1.5],
    elbow_frac=[0.5, 1, 1, 0.5, 0.5, 0.1, 0.7],
)

va.set_limits(ylim=(0.75, 3.3))

plt.show()
