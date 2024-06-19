# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
sys.path.insert(0, os.path.abspath('../../'))

project = 'vtacML'
copyright = '2024, Jeremy Palmerio'
author = 'Jeremy Palmerio'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.coverage',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',
    'recommonmark'
]
# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,  # Do not include private members
    'special-members': False,  # Do not include special members
    'inherited-members': False,  # Do not include inherited members
    'show-inheritance': True,
    'exclude-members': '_split_data, _get_data, _load_data, _create_pipe, _load_config, _resample'
}
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

from recommonmark.transform import AutoStructify


def setup(app):
    app.add_config_value('recommonmark_config', {
        'auto_toc_tree_section': 'Contents',
        'enable_eval_rst': True,
    }, True)
    app.add_transform(AutoStructify)
