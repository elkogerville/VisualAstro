"""
Author: Elko Gerville-Reache
Date Created: 2026-04-10
Date Modified: 2026-10-06
Description:
    VisualAstro color definitions.
"""

from typing import TypeAlias
import matplotlib as mpl
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.typing import ColorType


RGBTuple: TypeAlias = tuple[float, float, float]
RGBATuple: TypeAlias = tuple[float, float, float, float]


# ##########################
# VISUALASTRO COLOR PALETTES
# ##########################
COLORSETS: dict[str, list[ColorType]] = {
    'visualastro': ['#483D8B', '#DC267F', '#648FFF', '#FFB000', '#26DCBA'],
    'ditto': ['#483D8B', '#CC2B77', '#6A7DE0', '#22C9B6', '#ADCF00', '#F6A3FF', '#FF663C'],
    'supernatural': [
        '#CC436C', '#FF6CBC', '#D91D30', '#A4BF00', '#438B94',
        '#06D49C', '#00A9E8', '#6565B5', '#3434A1', '#1F2861'
    ],
    'supernova': [
        '#A10E4F', '#CF3C4F', '#C76AAE', '#EDEB00', '#A4BF00', '#08E3B2',
        '#0DB1FF', '#2E7F91', '#6768B5', '#3137A1', '#1F2861'
    ],
    'supersequential': [
        '#DC469F', '#FF4E9E', '#FE5325', '#FE901D', '#FFCA02', '#7AD12C',
        '#00AD54', '#02ACAC', '#00B1F5', '#0081CC', '#533DA1'
    ],
    'celestial': [
        '#08E3B2', '#0DB1FF', '#2E7F91', '#6565B5', '#3137A1', '#1F2861', '#A10E4F', '#CF3C4F', '#C76AAE'
    ],
    'astro_seq': [
        '#9FB7FF', '#648FFF', '#785EF0', '#DC267F',
        '#FE6100', '#FFB000', '#CFE23C', '#26DCBA'
    ],
    'astro': [
        '#785EF0', '#26DCBA', '#DC267F', '#648FFF',
        '#FFB000', '#9FB7FF', '#CFE23C', '#FE6100'
    ],
    'astro_contrast': [
        '#aed1ff', '#8f8ce7', '#5a06ef', '#dc267f', '#6c7a0e', '#cfe23c', '#26dcba'
    ],
    'turbo6': ['#35359A', '#4F65FF', '#5BFFD9', '#C4FF05', '#FF7D3C', '#AB0449'],
    'MSG': ['#483D8B', '#D81B60', '#DBB0FF', '#26DCBA', '#7D7FF3', '#CFE23C'],
    'MSGII': ['#DC267F', '#7D7FF3', '#483D8B', '#26DCBA', '#DBB0FF', '#CFE23C'],
    'MSG_seq': ['#483d8b', '#7d7ff3', '#dbb0ff', '#D81B60', '#26dcba', '#cfe23c'],
    'cardstock_dark': ['#000080', '#668035', '#187218', '#991D1B', '#992391', '#4E6767'],
    'cardstock_light': ['#9FD8FB', '#AED75B', '#BDDCBD', '#FB998E', '#E177AB', '#CDCDCD'],
    'crayons': [
        '#ED0A3F', '#FF8833', '#FBE870', '#01A368',
        '#0066FF', '#8359A3', '#AF593E', '#000000'
    ],
    'crayons_neon_seq': [
        '#00B9FB', '#00ECBD', '#66FF66', '#CCFF00', '#FFCC33',
        '#FF9966', '#FD5B78', '#FF1DCE', '#FF6EFF'
    ],
    'crayons_neon': [
        '#00B9FB', '#CCFF00' , '#FF6EFF', '#66FF66', '#FF9966',
        '#00ECBD', '#FF1DCE', '#FFCC33', '#FD5B78'
    ],
    'toad': ['#BFDBE8', '#867E09', '#93CB59', '#34E693', '#97968B'],
    'subway': [
        '#005DAD', '#F48820', '#00A66E', '#A67837', '#FFD005',
        '#929598', '#E42031', '#72B444', '#AD3F97', '#00ABCD'
    ],
    'default': list(TABLEAU_COLORS.values()),
    'ibm': ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'],
    'ibm_contrast': [
        '#648FFF', '#DC267F', '#785EF0', '#26DCBA', '#FFB000', '#FE6100'
    ],
    'temple_os': [
        '#555555', '#5555FF', '#55FF55', '#55FFFF', '#FF5555', '#FF55FF', '#FFFF55'
    ],
    'smplot': [
        'k', '#FF0000', '#0000FF', '#00FF00',
        '#00FFFF', '#FF00FF', '#FFFF00'
    ],
    'set2': mpl.color_sequences['Set2'],
    'dark2': mpl.color_sequences['Dark2'],
    'cornfield': ['#FFF662', '#CDE47D', '#F1BE95', '#70813F', '#DD716B', '#E4AC2C'],
    'tmrw_night': ['#719C95', '#9CD6CF', '#FFDA81', '#F3A169', '#8FB3D3', '#6B859C', '#435561'],
    'tmrw_night_seq': ['#8FB3D3', '#6B859C', '#719C95', '#9CD6CF', '#FFDA81', '#F3A169'],
    '2mrw_nite': ['#9EACD2', '#7C859D', '#719C95', '#9CD6CF', '#FFDA81', '#F3A169'],
    'debos': ['#3464F5', '#93BFE6', '#8FE3BC', '#F4C572', '#F56D53', '#D3153A', '#9C0569'],
    'deb': ['#3464F5', '#93BFE6', '#F4C572', '#D3153A'],
    'NGC6818': ['#5AC3BE', '#E770A2', '#4165C0', '#696969'],
    'rgb': ['#FF000F', '#007C6C', '#006B96'],
    'crayons_neon_rgb': ['#FF1DCE', '#CCFF00', '#00B9FB'],
    'evenaworm': ['#008b8b', '#98fb98', '#ff81c0', '#ceaefa', '#d81b60'],
    'ocean_seq': ['#253494', '#2c7fb8', '#41b6c4', '#a1dab4', '#ffffcc'],
    'forest_seq': ['#006837', '#31a354', '#78c679', '#c2e699', '#ffffcc'],
    'jade_seq': ['#006d2c', '#2ca25f', '#66c2a4', '#b2e2e2', '#edf8fb'],
    'violetred_seq': ['#980043', '#dd1c77', '#df65b0', '#d7b5d8', '#f1eef6'],
    'PiGn_div': ['#d01c8b', '#f1b6da', '#f7f7f7', '#b8e186', '#4dac26'],
    'BrBG_div': ['#a6611a', '#dfc27d', '#f5f5f5', '#80cdc1', '#018571'],
    'pastel5': ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0'],
    'high_vis': ['#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd'],
    'retro': ['#4165c0', '#e770a2', '#5ac3be', '#696969', '#f79a1e', '#ba7dcd'],
    'bright': ['#4477aa', '#ee6677', '#228833', '#ccbb44', '#66ccee', '#aa3377', '#bbbbbb'],
    'vibrant': ['#ee7733', '#0077bb', '#33bbee', '#ee3377', '#cc3311', '#009988', '#bbbbbb'],
    'muted': [
        '#cc6677', '#332288', '#ddcc77', '#117733', '#88ccee',
        '#882255', '#44aa99', '#999933', '#aa4499', '#dddddd'
    ],
    'light': [
        '#77aadd', '#ee8866', '#eedd88', '#ffaabb', '#99ddff',
        '#44bb99', '#bbcc33', '#aaaa00', '#dddddd'
    ],
    'dark': ['#222255', '#663333', '#225522', '#666633', '#225555', '#555555'],
    'medium_contrast': [
        '#6699cc', '#004488', '#eecc66', '#997700', '#ee99aa', '#994455', '#000000'
    ],
    'high_contrast': ['#000000', '#004488', '#bb5566', '#ddaa33', '#ffffff'],
    'land_cover': [
        '#5566aa', '#117733', '#668822', '#44aa66', '#99bb55',
        '#55aa22', '#558877', '#88bbaa', '#ddcc66', '#ffdd44',
        '#aaddcc', '#44aa88', '#ffee88', '#bb0011'
    ],
    'okabe_ito': [
        '#E69F00', '#56B4E9', '#009E73', '#F0E442',
        '#0072B2', '#D55E00', '#CC79A7', '#000000'
    ],
    'oit': ['#E69F00', '#56B4E9', '#D55E00', '#009E73', '#CC79A7'],
    'tab20b': mpl.color_sequences['tab20b'],
}

COLORSET_ALIASES = {
    'va': 'visualastro',
    'vb': 'ditto',
}

COLORSET_NAMES = [key for key in COLORSETS.keys()]


# ########################
# VISUALASTRO NAMED COLORS
# ########################
VISUALASTRO_NAMED_COLORS: dict[str, ColorType] = {
    'dsb': '#483D8B',
    'msb': '#7B68EE',
    'sb': '#6A5ACD',
    'mvr': '#C71585',
    'pvr': '#DB7093',
    'violetred': '#D81B60',
    'mam': '#66CDAA',
    'msg': '#3CB371',
    'jade': '#26DCBA',
    'nebula': '#9FB7FF',
    'unicorn': '#DBB0FF',
    'pondwater': '#CFE23C',
    'ibmpur': '#785EF0',
    'ibmpnk': '#DC267F',
    'ibmblu': '#648FFF',
    'ibmylw': '#FFB000',
    'ibmorg': '#FE6100',
    'laser lemon': '#E6FF66',
    'electric lime': '#CCFF00',
    'battery charged blue': '#00B9FB',
    'shocking pink': '#FF6EFF',
    'hot magenta': '#FF1DCE',
    'wild watermelon': '#FD5B78',
    'atomic tangerine': '#FF9966',
    'sunglow': '#FFCC33',
    'metis merlot': '#7B1242',
    'worm of the day': '#B577AC',
    'worm of the night': '#75415C',
    'soup of the day': '#848A21',
    'Holy Grey': '#555555',
    'Holy Blue': '#5555FF',
    'Holy Green': '#55FF55',
    'Holy Cyan': '#55FFFF',
    'Holy Red': '#FF5555',
    'Holy Magenta': '#FF55FF',
    'Holy Yellow': '#FFFF55',
    'subway blue': '#005DAD',
    'A train': '#0038A5',
    'C train': '#0089D0',
    'F train': '#F48820',
    '4 train': '#00A66E',
    'J train': '#A67837',
    'Q train': '#FFD005',
    'L train': '#929598',
    '3 train': '#E42031',
    'G train': '#72B444',
    '7 train': '#AD3F97',
    'T train': '#00ABCD',
}
