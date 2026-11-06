import matplotlib.pyplot as plt
import visualastro as va

plt.style.use(['full', 'cm'])
fig, ax = plt.subplots(2, 1, figsize=(14,10))
ax[0].set_title('Perceptually Uniform Colormaps')
va.plot_cmap_lightness(['viridis', 'plasma', 'inferno', 'cividis'], ncols=6, offset=.2, ax=ax[0])
va.plot_cmap_lightness(['turbo', 'jet', 'rainbow'], ncols=5, offset=.2, ax=ax[1])
plt.show()
