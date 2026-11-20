import matplotlib.pyplot as plt
import numpy as np

import visualastro as va

plt.style.use(['full', 'cm'])
rng = np.random.default_rng(42)

bimodal = np.concatenate([
        rng.normal(-1.5, 0.6, 30000),
        rng.normal(2.0, 1.0, 20000),
    ])

lognormal = rng.lognormal(mean=1.0, sigma=0.6, size=5000)

normals = [
    rng.normal(0, 10, 2000),
    rng.normal(3.5, 6, 2000),
    rng.normal(3.5, 4, 2000),
    rng.normal(0, 5, 2500),
    rng.normal(-3, 8, 2600),
]

fig, ax = plt.subplots(figsize=(6,6))

va.hist(
    normals,
    ax=ax,
    histtype='barstacked',
    color='evenaworm',
    labels = [
        r'$\mathcal{N}(\mu=0,\ \sigma=10)$',
        r'$\mathcal{N}(\mu=3.5,\ \sigma=6)$',
        r'$\mathcal{N}(\mu=3.5,\ \sigma=4)$',
        r'$\mathcal{N}(\mu=0,\ \sigma=5)$',
        r'$\mathcal{N}(\mu=-3,\ \sigma=8)$',
    ],
    xlim=(-45, 45),
)
va.legend(ax=ax, loc='upper left', ncols=1, linewidth=1.3, ec='k', frameon=True)

plt.show()
