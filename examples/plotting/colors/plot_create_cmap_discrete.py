import matplotlib as mpl
import matplotlib.pyplot as plt
import visualastro as va

cmap = va.create_cmap('astro_seq', kind='discrete')

fig, ax = plt.subplots(figsize=(6, 0.8))
fig.subplots_adjust(bottom=0.5)
norm = mpl.colors.Normalize(0, 1)
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax, orientation='horizontal', extend='both'
)
