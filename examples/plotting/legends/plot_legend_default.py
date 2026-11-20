import matplotlib.pyplot as plt
import numpy as np

import visualastro as va

plt.style.use(['full', 'cm'])
rng = np.random.default_rng(42)

bimodal = np.concatenate([
        rng.normal(-1.5, 0.6, 30000),
        rng.normal(2.0, 1.0, 20000),
    ])

fig, ax = plt.subplots(figsize=(6,6))

va.hist(
    bimodal,
    ax=ax,
    label='bimodal', color='k',
    vlines=[np.median(bimodal), np.mean(bimodal), np.std(bimodal)],
    vlines_color=['violetred', 'Holy Blue', 'ibmorg'],
    vlines_ls=['solid', 'dotted', 'dashed'],
    vlines_lw=1.5,
)
va.legend(ax=ax, loc='upper right')

plt.show()
