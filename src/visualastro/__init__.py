# Core Classes
from visualastro.DataCube import DataCube
from visualastro.FitsFile import FitsFile
from visualastro.SpectrumPlus import SpectrumPlus

# Submodules
from visualastro.data_cube import *
from visualastro.data_cube_utils import *
from visualastro.io import *
from visualastro.numerical_utils import *
from visualastro.plotting import *
from visualastro.plot_utils import *
from visualastro.spectra import *
from visualastro.spectra_utils import *
from visualastro.text_utils import pretty_table, print_pretty_table
from visualastro.units import *
from visualastro.va_config import va_config
from visualastro.visual_plots import *
from visualastro.wcs_utils import crop2D, get_wcs, reproject_wcs

def _register_fonts():
    '''
    Register additional fonts into matplotlib.
    To add more fonts, simply add a folder to
    VisualAstro/src/visualastro/stylelib/Fonts
    with .ttf or .otf files.
    '''
    from pathlib import Path
    import warnings
    import matplotlib.font_manager as fm

    package_dir = Path(__file__).parent
    fonts_dir = package_dir / 'stylelib' / 'Fonts'

    if not fonts_dir.exists():
        return

    font_files = list(fonts_dir.rglob('*.ttf')) + list(fonts_dir.rglob('*.otf'))

    for font_file in font_files:
        try:
            fm.fontManager.addfont(str(font_file))
        except Exception as e:
            warnings.warn(
                f'[visualastro] Could not register font {font_file.name}: {e}'
            )

_register_fonts()
