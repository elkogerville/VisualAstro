import matplotlib.pyplot as plt
import visualastro as va

plt.style.use(['full', 'concrete'])
va.config.fontsizes.tick_labels = 0.87
va.config.legend.markerscale = 15

blob1 = va.blob(5000, as_array=True)
blob2 = va.blob(5000, as_array=True)
blob1[:,0] += 5
blob2[:,1] -= 10

dens = va.number_density(blob2[:,0:2], k=50)
va.scatter(blob1[:,0], blob1[:,1], s=0.1, color='Holy Red', title='limits=10')
va.scatter(
    blob2[:,0], blob2[:,1],
    s=0.1, c=dens, grid=True,
    grid_color='blue', grid_ls=':', grid_alpha=0.5
)
va.set_limits(limits=10, center='com', points=blob2[:,0:2])

plt.show()
