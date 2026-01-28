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
$ conda install -c conda-forge \
    astropy dust_extinction matplotlib numpy regions reproject spectral-cube specutils scipy tqdm
```
For additional interactive functionality inside of jupyter notebooks:
```
$ conda install -c conda-forge \
    ipympl ipywidgets notebook jupyterlab jupyter_server notebook-shim
```
And finally run:
```
$ pip install visualastro
```

NOTE: To ensure that interactive mode works in notebooks, first activate your conda environment and then activate jupyter notebook!

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
astropy, dust_extinction, matplotlib, regions, reproject, spectral-cube, specutils, scipy, and tqdm.


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
