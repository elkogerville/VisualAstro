from .colors import (
    get_color,
    get_colors,
    get_colorset,
    get_namedcolor,
    random_colors,
    sample_cmap,
)
from .definitions import (
    COLORSETS,
    COLORSET_ALIASES,
    COLORSET_NAMES,
    VISUALASTRO_NAMED_COLORS
)
from .plots import (
    plot_color_deltaE,
    plot_colors,
    plot_colorset,
    plot_colortable,
)
from .transforms import (
    darken_colors,
    desaturate_colors,
    lighten_colors,
    saturate_colors,
    simulate_colorblindness,
)
from .utils import as_color
