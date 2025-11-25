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
Currently, the most stable version of python for visualastro is version 3.11.
To install visualastro, it is advised to create a new conda environment if possible:
```
$ conda create envname -c conda-forge python=3.11
$ conda activate envname
```
Then install the dependencies with:
```
$ conda install -c conda-forge astropy dust_extinction matplotlib numpy reproject regions scipy spectral-cube specutils tqdm
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

## Installation

You can install visualastro via pip:
```
pip install visualastro
```
## Documentation
The full documentation can be found on github at https://github.com/elkogerville/VisualAstro

## Dependencies

VisualAstro requires:
astropy, matplotlib, scipy, numba, regions, spectral-cube, specutils, and tqdm.
