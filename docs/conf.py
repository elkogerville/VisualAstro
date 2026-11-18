# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from importlib.metadata import version as _pkg_version

project = 'VisualAstro'
copyright = '2026, Elko Gerville-Reache and VisualAstro contributors'
author = 'Elko Gerville-Reache, Qiushi Chris Tian, and Audrey Whitmer'
release = _pkg_version('visualastro')
version = '.'.join(release.split('.')[:3])

html_title = f'VisualAstro {version}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
    "pytest_doctestplus.sphinx.doctestplus",
]

default_role = "py:obj"

# napoleon
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# autodoc / autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "none"
add_module_names = False

# copybutton
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "specutils": ("https://specutils.readthedocs.io/en/stable/", None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'logo': {'text': f'VisualAstro {version}'},
    'secondary_sidebar_items': ['page-toc'],
    'show_toc_level': 2,
    'navigation_depth': 3,
}
html_static_path = ['_static']

plot_apply_rcparams = True
plot_rcparams = {"savefig.bbox": "tight"}
plot_formats = ["svg"]
