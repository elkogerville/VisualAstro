import matplotlib.pyplot as plt
import numpy as np
import visualastro as va


plt.style.use(['full', 'concrete'])

gal = np.load('snapshot_N6000_timestep_1500.npy')

fig, ax = plt.subplots(1, 2, figsize=(12,6))

va.scatter(
    gal, ax=ax[0], xy_indices=(0,1), limits=(-20, 15),
    s=1, color='mediumvioletred', xlabel='X', ylabel='Y',
    title='limits=(-20, 15)'
)

va.scatter(
    gal, ax=ax[1], xy_indices=(0,2), limits=-10,
    s=1, color='slateblue', xlabel='X', ylabel='Z',
    title='limits=-10'
)

plt.show()
