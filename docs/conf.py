"""Sphinx configuration."""
project = "Indypy Talk 20230508"
author = "Brian Ritz"
copyright = "2023, Brian Ritz"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
