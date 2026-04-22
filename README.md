```
            /$$                            /$$                    /$$                        
           |__/                           | $$                   | $$                        
 /$$    /$$ /$$  /$$$$$$ /$$  /$$  /$$$$$ | $$  /$$$$$   /$$$$$ /$$$$$$   /$$$$$   /$$$$$$ 
|  $$  /$$/| $$ /$$____/| $$ | $$ |____ $$| $$ |____ $$ /$$___/|_  $$_/  /$$__ $$ /$$__  $$
 \  $$/$$/ | $$|  $$$$$ | $$ | $$  /$$$$$$| $$  /$$$$$$|  $$$$$  | $$   | $$ \__/| $$  \ $$
  \  $$$/  | $$ \___  $$| $$ | $$ /$$__ $$| $$ /$$__ $$ \___  $$ | $$ /$| $$     | $$  | $$
   \  $/   | $$ /$$$$$$/|  $$$$$/|  $$$$$$| $$|  $$$$$$ /$$$$$$/ |  $$$$| $$     |  $$$$$$/
    \_/    |__/|______/  \_____/  \______/|__/ \______/|______/   \___/ |__/      \______/ 
```


# VisualAstro
[![PyPI Version](https://img.shields.io/pypi/v/visualastro)](https://pypi.org/project/visualastro)
[![Tests](https://github.com/elkogerville/VisualAstro/actions/workflows/test.yml/badge.svg)](https://github.com/elkogerville/VisualAstro/actions/workflows/test.yml)

**visualastro** is an astrophysical visualization system with functions for easy visualization and manipulation of common astronomical data. The package is developed with ease of use in mind, modularity, and making publication-ready plots with minimal commands.

## Installation

Currently, the most stable version of python for visualastro is version >=3.10.
To install visualastro, it is advised to create a new conda environment if possible:
```
$ conda create visualastro -c conda-forge python=3.14
$ conda activate visualastro
```
Then install the dependencies with:
```
$ conda install -c conda-forge \
    astropy dust_extinction matplotlib numpy regions reproject spectral-cube specutils scipy tqdm
```
For additional interactive functionality inside of jupyter lab:
```
$ conda install -c conda-forge ipympl ipywidgets jupyterlab
```
For classic jupyter notebook users:
```
$ conda install -c conda-forge ipympl ipywidgets notebook jupyter_server notebook-shim
```
And finally run:
```
$ pip install visualastro
```

NOTE: To ensure that interactive mode works in notebooks, first activate your conda environment and then run jupyter notebook!


## Features

- Unified interface for matplotlib, astropy, numpy, spectral-cube, specutils, and other astronomy packages
- High-level wrappers of common functions and algorithms used in astrophysical research
- Custom matplotlib style sheets optimized for publication-quality figures
- Full compatibility with WCS, FITS, and astropy units

## Documentation
Visualastro is still under development! A full documentation of the package's features is coming soon.

The full documentation can be found on github at https://github.com/elkogerville/VisualAstro

## Dependencies

VisualAstro requires:

astropy, dust_extinction, matplotlib, numpy, regions, reproject, spectral-cube, specutils, scipy, tol_colors, and tqdm.


## Credits

### Fonts
VisualAstro includes Hershey-style TrueType fonts from the smplotlib project
by Jiaxuan Li, used under the MIT License. Citation:

```
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
```
