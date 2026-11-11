import matplotlib.pyplot as plt
import numpy as np
import visualastro as va

plt.style.use(['minimal', 'stix'])

rng = np.random.default_rng(42)
n = 2000
x = rng.normal(0, 4, n)
y = rng.normal(0, 1, n)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

for ax, square in zip(axs, (False, True)):
    va.scatter(x, y, ax=ax, s=2, color='nebula', transform='saturate', factor=0.9)
    va.set_limits(ax, xlim=(-10, 10), square=square)
    va.set_title(f'square={square}', ax=ax)

plt.show()
