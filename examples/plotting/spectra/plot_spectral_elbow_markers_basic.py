import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models
from specutils import Spectrum

import visualastro as va

np.random.seed(0)
amplitudes = [1.4, 2.0, 1.8, 1.5, 1.2]
line_waves = [4634.140, 4640.640, 4641.850, 4647.420, 4651.470]
labels = ['N III', 'N III', 'N III', 'C III', 'C III']
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

plt.style.use(['full', 'stix'])
va.config.show_type_label = True
fig, ax = va.subplots(figsize=(7,7))
va.plot_spectrum(spectrum, figsize=(12,6), color='dark verdigris')

va.spectral_elbow_markers(
    line_waves, labels,
    y=y_vals,
    h=[0.13, 0.1, 0.25, 0.3, 0.1],
    label_shift=[0.2, -0.8, 2, 0.8, -1],
    elbow_frac=[0.5, 1, 0.2, 0.8, 0.5],
)

va.set_limits(ylim=(0.75, 3.3), xlim=(4630, 4655))

plt.show()
