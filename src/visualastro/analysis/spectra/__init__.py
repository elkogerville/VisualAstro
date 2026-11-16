"""
Author: Elko Gerville-Reache
Date Created: 2025-11-16
Date Modified: 2026-11-16
Description:
    Spectra analysis module imports.
"""

from .fitting import (
    GaussianFitResult,
    ExtractedPixelSpectra,
    gaussian,
    gaussian_continuum,
    gaussian_line,
    gaussian_p0,
)
from .manipulation import (
    compute_continuum,
    continuum_spectrum,
    deredden_flux,
    extract_region,
    shift_to_rest_frame,
)
from .utils import (
    estimate_spectrum_line_flux,
    propagate_flux_errors,
    sort_spectra_by_line_strength,
    spectral_idx_2_world,
    spectral_world_2_idx,
)
from .spectral_region import spectral_region, spectral_region_mask
