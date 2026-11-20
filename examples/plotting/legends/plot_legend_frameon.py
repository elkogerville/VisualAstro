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

fig, ax = plt.subplots(figsize=(6,6))

va.hist(
    [lognormal, bimodal],
    ax=ax,
    histtype='stepfilled',
    ylog=True,
    alpha=0.7,
    label=['lognorm', 'bimodal'],
    color=['r', 'b'],
)
va.legend(ax=ax, frameon=True, ec='k')

plt.show()
