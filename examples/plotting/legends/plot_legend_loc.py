import matplotlib.pyplot as plt
import numpy as np

import visualastro as va

plt.style.use(['full', 'cm'])
rng = np.random.default_rng(42)

nongauss = [
    rng.gamma(shape=2.0, scale=2.0, size=2000),
    rng.laplace(loc=4.0, scale=2.0, size=2000),
    rng.uniform(low=-6.0, high=10.0, size=2000),
    rng.standard_t(df=3, size=2500) * 2.0 + 2.0,
    -rng.lognormal(mean=1.0, sigma=0.5, size=2600) + 8.0,
    rng.weibull(a=1.5, size=2200) * 5.0 - 2.0,
    rng.beta(a=0.5, b=0.5, size=2400) * 16.0 - 6.0,
    rng.chisquare(df=3, size=2300) * 1.5 - 1.0,
]

fig, ax = plt.subplots(1, 2, figsize=(12,6))

va.hist(
    nongauss,
    ax=ax[0],
    histtype='stepfilled',
    color='celestial',
    alpha=0.8,
    label=[r'$\psi$', r'$\phi$', r'$\lambda$', r'$\epsilon$', r'$\nu$', r'$\rho$', r'$\chi$', r'$\mu$'],
    xlim=(-10,20),
)
va.legend(
    ax=ax[0], frameon=True, ec='k', reverse=True, ncols=4,
    markerfirst=False, fancybox=True, loc='bottom center'
)


x = np.arange(500)
y1, y2, y3 = np.cumsum(rng.normal(size=(3, x.size)), axis=1)

va.step(x, y1, color='stratosphere blue', label='A')
va.step(x, y2, color='soup of the day', label='B')
va.step(x, y3, color='aurora green', label='C')

va.legend(loc='top center', ncols=3, borderaxespad=-1, frameon=True, ec='k')
plt.show()
