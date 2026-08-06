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
[![PyPI Version](https://img.shields.io/pypi/v/visualastro?cachebust=1)](https://pypi.org/project/visualastro)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21729934.svg)](https://doi.org/10.5281/zenodo.21729934)
[![Tests](https://github.com/elkogerville/VisualAstro/actions/workflows/test.yml/badge.svg)](https://github.com/elkogerville/VisualAstro/actions/workflows/test.yml)

**VisualAstro** is an astrophysical visualization system with functions for easy visualization and manipulation of astronomical data. The package aims to make publication-ready plots with minimal commands and no LaTeX installation.

<p align="center">
  <img src="https://github.com/elkogerville/VisualAstro/blob/main/example_figures/SN1987A.png" width="30%">
  <img src="https://github.com/elkogerville/VisualAstro/blob/main/example_figures/MIRI_spectrum_bkg_subtracted.png" width="60%">
</p>

## Features

- Unified interface for Matplotlib, Astropy, NumPy, spectral-cube, specutils, and other astronomy packages
- Custom mathtext stylesheets without `LaTeX` installation as well as many other styles for publication-quality figures
- Colorblind safe alternative colorsets, and colormaps
- User-friendly interface with many convenience methods that integrate with Matplotlib and other packages

## Installation

Currently, the most stable version of python for VisualAstro is version >=3.10.
To install VisualAstro, it is advised to create a new conda environment if possible:
```
$ conda create -n visualastro -c conda-forge python=3.13
$ conda activate visualastro
```
Then install the dependencies with:
```
$ conda install -c conda-forge \
    astropy matplotlib numpy scipy
```
Optional dependencies can also be installed if you plan to use related functionalities:
```
$ conda install -c conda-forge \
    colorspacious dust_extinction regions reproject spectral-cube scienceplots specutils tqdm
```
And finally run:
```
$ pip install visualastro
```
For additional interactive functionality inside of jupyter lab:
```
$ conda install -c conda-forge ipympl ipywidgets jupyterlab
```
For classic jupyter notebook users:
```
$ conda install -c conda-forge ipympl ipywidgets notebook jupyter_server notebook-shim
```

NOTE: To ensure that interactive mode works in notebooks, first activate your conda environment and then run jupyter notebook!


## Documentation
VisualAstro is still under development! A full documentation of the package's features is coming soon.

Check the [examples](examples/) folder for notebook tutorials!

## Dependencies

VisualAstro requires:

`astropy`, `matplotlib`, `numpy`, and `scipy`.

### Optional dependencies

Optionally, some functionalities of VisualAstro require:

* Image Data: `regions`, `reproject`
* Cube Data: `spectral-cube`
* Spectra: `specutils`, `dust_extinction`
* Extra Stylesheets: `scienceplots`
* Color Utilities: `cmasher`, `colorspacious`, `tol-colors`
* Progress Bar: `tqdm`

If you try to use functionalities that require an optionally dependent package but do not have that package installed, VisualAstro will raise an ImportError and prompt you to install that package.

## Examples

Example colorsets:
<p align="center">
  <img src="https://github.com/elkogerville/VisualAstro/blob/colors/example_figures/colorsets.png" width="90%">
</p>
<!-- <p align="center">
  <img src="https://github.com/elkogerville/VisualAstro/blob/main/example_figures/astro_seq.png" width="45%">
  <img src="https://github.com/elkogerville/VisualAstro/blob/main/example_figures/debos.png" width="45%">
</p> -->

Example fontstyles:
<p align="center">
  <img src="https://github.com/elkogerville/VisualAstro/blob/main/example_figures/cm_fontstyle.png" width="45%">
  <img src="https://github.com/elkogerville/VisualAstro/blob/main/example_figures/libertinus_fontstyle.png" width="45%">
</p>

## Credits

### Fonts
VisualAstro includes multiple mathtext fonts:

Concrete Math, distributed under the SIL OPEN FONT LICENSE from a release by:
```
author = Daniel Flipo
year = 2022-2026
url = https://ctan.org/tex-archive/fonts/concmath-otf?lang=en
```

Libertinus Math, distributed under the SIL OPEN FONT LICENSE from a release by:
```
authors = Caleb Maclennan, Libertinus Project Authors
year = 2012-2024
url = https://github.com/alerque/libertinus V7.051
```

New Computer Modern, distributed under GNU GENERAL PUBLIC LICENSE from a release by:
```
author = Antonis Tsolomitis
location = Samos, Greece
year = 2019--2026
url = https://ctan.org/texarchive/fonts/newcomputermodern?lang=en
```

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
