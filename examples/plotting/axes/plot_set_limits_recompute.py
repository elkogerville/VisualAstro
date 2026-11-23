import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models
from specutils import Spectrum

import visualastro as va

np.random.seed(0)
amplitudes = [0.4, 0.5, 2.0, 1.5, 1.5, 1.2, 0.97]
line_waves = [4629.339, 4634.140, 4640.640, 4641.850, 4647.420, 4651.470, 4658.050]
gaussians = [models.Gaussian1D(A, mu, 0.2) for (A, mu) in zip(amplitudes, line_waves)]

x = np.linspace(4625, 4665, 500)
y = (
    1.0 + 2e-4 * (x - 4500)
    + sum(g(x) for g in gaussians)
    + np.random.normal(0.0, 0.05, x.shape)
)
spectrum = Spectrum(flux=y*u.erg*u.cm**-2*u.s**-1, spectral_axis=x * u.AA)

plt.style.use('concrete')
va.config.show_type_label = False
fig, ax = va.subplots(1, 2, figsize=(12,5))

va.plot_spectrum(
    spectrum, ax=ax[0],
    color='abyssal',
    title='recompute_other_axis=False'
)
va.set_limits(ax=ax[0], xlim=(4625, 4640.34), recompute_other_axis=False)

va.plot_spectrum(
    spectrum, ax=ax[1],
    color='abyssal',
    title='recompute_other_axis=True'
)
va.set_limits(ax=ax[1], xlim=(4625, 4640.34), recompute_other_axis=True)

plt.show()
