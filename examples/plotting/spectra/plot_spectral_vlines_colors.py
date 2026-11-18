import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.modeling import models
from specutils import Spectrum
import visualastro as va

np.random.seed(13)
amplitudes = [2.0, 0.6, 3.0, 1.0, 4.0, 0.5, 2.0, 2.5]
line_wave = [4102.89, 4150, 4341.68, 4364.436, 4862.68, 4932.603, 4960.295, 5008.240]
labels = ['Hδ', None, 'Hγ', 'O III', 'Hβ', 'O III', 'O III', 'O III']

gaussians = [models.Gaussian1D(A, mu, 3.0) for (A, mu) in zip(amplitudes, line_wave)]

x = np.linspace(4050, 5050, 500)

y = (
    1.0 + 2e-4 * (x - 4050)
    + sum(g(x) for g in gaussians)
    + np.random.normal(0.0, 0.15, x.shape)
)
spectrum = Spectrum(flux=y * u.Jy, spectral_axis=x * u.AA)


plt.style.use(['full', 'cm'])
fig, ax = va.subplots(figsize=(12,6))
va.plot_spectra(spectrum, color='cobalt')
va.spectral_vlines(
    line_wave, labels,
    color=['4 train', '7 train', '3 train', 'C train'],
    unlabeled_color='L train',
    lw=1,
    rotation=90,
    label_offset_points=(8,-4)
)
va.set_limits(ylim=(0, 6))
