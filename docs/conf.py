# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "PathBench"
copyright = "2025, Bence Mark Halpern"
author = "Bence Mark Halpern"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]

html_css_files = [
    "https://cdn.datatables.net/2.2.2/css/dataTables.dataTables.min.css",
    "https://cdn.datatables.net/buttons/3.2.1/css/buttons.dataTables.min.css",
    "pb_datatable.css",
]

html_js_files = [
    ("https://code.jquery.com/jquery-3.7.1.slim.min.js", {"defer": "defer"}),
    ("https://cdn.datatables.net/2.2.2/js/dataTables.min.js", {"defer": "defer"}),
    ("https://cdn.datatables.net/buttons/3.2.1/js/dataTables.buttons.min.js", {"defer": "defer"}),
    ("https://cdn.datatables.net/buttons/3.2.1/js/buttons.colVis.min.js", {"defer": "defer"}),
    ("pb_datatable.js", {"defer": "defer"}),
]

# -- autodoc settings --------------------------------------------------------
# Mock all heavy / native dependencies so Sphinx can import pathbench modules
# without requiring PyTorch, CUDA, espeak-ng, FFTW, etc.
autodoc_mock_imports = [
    "dtw",
    "jiwer",
    "librosa",
    "matplotlib",
    "numpy",
    "parselmouth",
    "phonemizer",
    "pyctcdecode",
    "scipy",
    "sklearn",
    "soundfile",
    "torch",
    "torchaudio",
    "transformers",
]
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
