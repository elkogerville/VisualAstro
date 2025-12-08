```
 /$$    /$$ /$$                               /$$  /$$$$$$              /$$                        
| $$   | $$|__/                              | $$ /$$__  $$            | $$                        
| $$   | $$ /$$  /$$$$$$$ /$$   /$$  /$$$$$$ | $$| $$  \ $$  /$$$$$$$ /$$$$$$    /$$$$$$   /$$$$$$ 
|  $$ / $$/| $$ /$$_____/| $$  | $$ |____  $$| $$| $$$$$$$$ /$$_____/|_  $$_/   /$$__  $$ /$$__  $$
 \  $$ $$/ | $$|  $$$$$$ | $$  | $$  /$$$$$$$| $$| $$__  $$|  $$$$$$   | $$    | $$  \__/| $$  \ $$
  \  $$$/  | $$ \____  $$| $$  | $$ /$$__  $$| $$| $$  | $$ \____  $$  | $$ /$$| $$      | $$  | $$
   \  $/   | $$ /$$$$$$$/|  $$$$$$/|  $$$$$$$| $$| $$  | $$ /$$$$$$$/  |  $$$$/| $$      |  $$$$$$/
    \_/    |__/|_______/  \______/  \_______/|__/|__/  |__/|_______/    \___/  |__/       \______/ 
```


# VisualAstro

**visualastro** is an astrophysical visualization system with convenient functions for easy visualization of common astronomical data. The package is developed with ease of use in mind, and making publication ready plots.

## Installation
[![PyPI Version](https://img.shields.io/pypi/v/visualastro)](https://pypi.org/project/visualastro)

Currently, the most stable version of python for visualastro is version 3.11.
To install visualastro, it is advised to create a new conda environment if possible:
```
$ conda create envname -c conda-forge python=3.11
$ conda activate envname
```
Then install the dependencies with:
```
$ conda install -c conda-forge astropy dust_extinction matplotlib numpy regions reproject scipy spectral-cube specutils tqdm
```
And finally run:
```
$ pip install visualastro
```

## Compatible Data
- 2D images
- 3D spectral cubes
- 1D spectra with gaussian fitting tools

## Features

- Simple, high-level wrapper functions for common astrophysical plots
- Custom matplotlib style sheets optimized for publication-quality figures
- Full compatibility with WCS, FITS

## Documentation
The full documentation can be found on github at https://github.com/elkogerville/VisualAstro

## Dependencies

VisualAstro requires:
astropy, matplotlib, scipy, numba, regions, reproject, spectral-cube, specutils, and tqdm.


## Credits

### Fonts
VisualAstro includes Hershey-style TrueType fonts from the smplotlib project
by Jiaxuan Li, used under the MIT License. Citation:

@software{jiaxuan_li_2023_8126529,
  author       = {Jiaxuan Li},
  title        = {AstroJacobLi/smplotlib: v0.0.9},
  month        = jul,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.0.9},
  doi          = {10.5281/zenodo.8126529},
  url          = {https://doi.org/10.5281/zenodo.8126529},
}
