import matplotlib.pyplot as plt
import visualastro as va

plt.style.use(['full', 'concrete'])
va.config.fontsizes.tick_labels = 0.87
va.config.legend.markerscale = 15

blob1 = va.blob(5000, as_array=True)
blob2 = va.blob(2500, as_array=True)
blob1[:,0] += 5
blob2[:,1] -= 10


fig, ax = va.subplots(1, 3, figsize=(13,4))

va.scatter(blob1[:,0], blob1[:,1], ax=ax[0], s=0.1, color='Holy Red', label='Galaxy 1')
va.scatter(blob2[:,0], blob2[:,1], ax=ax[0], s=0.1, color='Holy Blue', label='Galaxy 2')
va.set_title("limits=10, center='com'", ax=ax[0])
va.set_limits(limits=10, ax=ax[0], center='com')

va.scatter(blob1[:,0], blob1[:,1], ax=ax[1], s=0.1, color='Holy Red', label='Galaxy 1')
va.scatter(blob2[:,0], blob2[:,1], ax=ax[1], s=0.1, color='Holy Blue', label='Galaxy 2')
va.scatter(-1000000, 0, ax=ax[1], s=0.1, label='Outlier')
va.set_title("limits=10, center='com'", ax=ax[1])
va.set_limits(limits=10, ax=ax[1], center='com')

va.scatter(blob1[:,0], blob1[:,1], ax=ax[2], s=0.1, color='Holy Red', label='Galaxy 1')
va.scatter(blob2[:,0], blob2[:,1], ax=ax[2], s=0.1, color='Holy Blue', label='Galaxy 2')
va.scatter(-10000, 0, ax=ax[2], s=0.1, title="center='median'", label='Outlier')
va.set_title("limits=10, center='median'", ax=ax[2])
va.set_limits(limits=10, ax=ax[2], center='median')

plt.show()
