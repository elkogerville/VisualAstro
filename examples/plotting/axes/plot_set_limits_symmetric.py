import matplotlib.pyplot as plt
import visualastro as va

plt.style.use(['full', 'concrete'])
va.config.fontsizes.tick_labels = 0.87

blob1 = va.blob(5000, as_array=True)
blob2 = va.blob(2500, as_array=True)
blob1[:,0] += 5
blob2[:,1] -= 10


fig, ax = va.subplots(1, 2, figsize=(11,5))

va.scatter(blob1[:,0], blob1[:,1], ax=ax[0], s=0.1, color='Holy Red', title='limits=(-2,8)')
va.set_limits(ax=ax[0], limits=(-2, 8))

va.scatter(blob1[:,0], blob1[:,1], ax=ax[1], s=0.1, color='Holy Red', title='limits=15')
va.scatter(
    blob2[:,0], blob2[:,1], ax=ax[1],
    s=0.1, color='Holy Blue', grid=True,
    grid_color='blue', grid_ls=':', grid_alpha=0.5
)
va.set_limits(ax=ax[1], limits=15)

plt.show()
