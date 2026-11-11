import matplotlib.pyplot as plt
import numpy as np
import visualastro as va

plt.style.use(['full', 'cm'])
rng = np.random.default_rng(42)

# generate some data
bimodal = np.concatenate([
        rng.normal(-1.5, 0.6, 30000),
        rng.normal(2.0, 1.0, 20000),
    ])

lognormal = rng.lognormal(mean=1.0, sigma=0.6, size=5000)

normals = [
    rng.normal(0, 10, 2000),
    rng.normal(3.5, 6, 2000),
    rng.normal(3.5, 4, 2000),
    rng.normal(0, 5, 2500),
    rng.normal(-3, 8, 2600),
]

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

# plot the data
fig, ax = plt.subplots(2, 2, figsize=(12,12))

va.hist(
    bimodal,
    ax=ax[0,0],
    label='bimodal', color='k',
    vlines=[np.median(bimodal), np.mean(bimodal), np.std(bimodal)],
    vlines_color=['violetred', 'Holy Blue', 'ibmorg'],
    vlines_ls=['solid', 'dotted', 'dashed'],
    vlines_lw=1.5,
)
va.legend(ax=ax[0,0], loc='upper right')

va.hist(
    [lognormal, bimodal],
    ax=ax[0,1],
    histtype='stepfilled',
    ylog=True,
    alpha=0.7,
    label=['lognorm', 'bimodal'],
    color=['r', 'b'],
)
va.legend(ax=ax[0,1], frameon=True, ec='k')

va.hist(
    nongauss,
    ax=ax[1,0],
    histtype='stepfilled',
    color='celestial',
    alpha=0.8,
    label=[r'$\psi$', r'$\phi$', r'$\lambda$', r'$\epsilon$', r'$\nu$', r'$\rho$', r'$\chi$', r'$\mu$'],
    xlim=(-10,20),
)
va.legend(
    ax=ax[1,0], frameon=True, ec='k', reverse=True, ncols=4,
    markerfirst=False, fancybox=True, loc='bottom center'
)

va.hist(
    normals,
    ax=ax[1,1],
    histtype='barstacked',
    color='evenaworm',
    labels = [
        r'$\mathcal{N}(\mu=0,\ \sigma=10)$',
        r'$\mathcal{N}(\mu=3.5,\ \sigma=6)$',
        r'$\mathcal{N}(\mu=3.5,\ \sigma=4)$',
        r'$\mathcal{N}(\mu=0,\ \sigma=5)$',
        r'$\mathcal{N}(\mu=-3,\ \sigma=8)$',
    ],
    xlim=(-45, 45),
)
va.legend(ax=ax[1,1], loc='best', ncols=1)

plt.show()
