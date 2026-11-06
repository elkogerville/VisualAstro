import matplotlib as mpl
import matplotlib.pyplot as plt
import visualastro as va

cmap = va.get_cmap(
    'turbo', cmap_range=(.1,.9), bad_color='grey', under_color='b', over_color='r'
)

fig, ax = plt.subplots(figsize=(6, 0.8))
fig.subplots_adjust(bottom=0.5)
norm = mpl.colors.Normalize(0, 1)
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax, orientation='horizontal', extend='both'
)
