import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.modeling import models
from specutils import Spectrum

import visualastro as va

rng = np.random.default_rng(13)
amplitudes = [1.1, 1.4, 2.0, 1.5, 1.5, 1.2, 0.97]
line_waves = [4629.339, 4634.140, 4640.640, 4641.850, 4647.420, 4651.470, 4658.050]
labels = ['Fe II', 'N III', 'N III', 'N III', 'C III', 'C III', '[Fe III]']

sigmas = [0.35, 0.25, 0.30, 0.30, 0.45, 0.40, 0.35]

gaussians = [models.Gaussian1D(A, mu, s) for A, mu, s in zip(amplitudes, line_waves, sigmas)]

x = np.linspace(4625, 4665, 500)
y = (
    1.0 - 1e-2 * (x - 4645)
    + models.Gaussian1D(-0.3, 4645, 4.0)(x)
    + sum(g(x) for g in gaussians)
    + rng.normal(0.0, 0.08, x.shape)
)
spectrum = Spectrum(flux=y * u.Jy, spectral_axis=x * u.AA)

# peak height
y_vals = np.interp(line_waves * u.AA, spectrum.spectral_axis, spectrum.flux) + 0.03 * u.Jy

plt.style.use(['full', 'libertinus'])
fig, ax = va.subplots(figsize=(7,7))
va.plot_spectrum(spectrum, color='viridian')

va.spectral_elbow_markers(
    line_waves, labels, y=y_vals, h=0.11,
    label_shift=[0, -0.8, 1, 1.8, 0.0, 1, -1.5],
    elbow_frac=[0.5, 1, 0, 0.5, 0.0, 0.1, 0],
    color=['A train', 'borscht', 'ethernet green', '7 train'],
    rotation=90,
    label_offset_points=(2,3) # x, y offset
)

va.set_limits(ylim=(0.4, 3.3))

plt.show()
