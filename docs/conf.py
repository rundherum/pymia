#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# pymia documentation build configuration file, created by
# sphinx-quickstart on Tue May 30 21:31:00 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import shutil
import sys

import sphinx_rtd_theme

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, basedir)

about = {}
with open(os.path.join(basedir, 'pymia', '__version__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), about)

# -- Copy example Jupyter notebooks for documentation building
shutil.copyfile(os.path.join(basedir, 'examples', 'augmentation', 'basic.ipynb'),
                os.path.join(basedir, 'docs', 'examples.augmentation.basic.ipynb'))

shutil.copyfile(os.path.join(basedir, 'examples', 'data', 'creation.ipynb'),
                os.path.join(basedir, 'docs', 'examples.data.creation.ipynb'))

# examples.data.extraction_assembly.ipynb not copied as there exists a rst file

shutil.copyfile(os.path.join(basedir, 'examples', 'evaluation', 'basic.ipynb'),
                os.path.join(basedir, 'docs', 'examples.evaluation.basic.ipynb'))

shutil.copyfile(os.path.join(basedir, 'examples', 'evaluation', 'logging.ipynb'),
                os.path.join(basedir, 'docs', 'examples.evaluation.logging.ipynb'))

shutil.copyfile(os.path.join(basedir, 'examples', 'filtering', 'basic.ipynb'),
                os.path.join(basedir, 'docs', 'examples.filtering.basic.ipynb'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '1.3'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage',
              'sphinx.ext.githubpages',
              'sphinx.ext.imgmath',
              'sphinx.ext.todo',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'nbsphinx',
              'sphinx_copybutton']

napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = about['__title__']
copyright = about['__copyright__']
author = about['__author__']

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = about['__version__']
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# modules to be mocked
autodoc_mock_imports = ['tensorflow', 'torch']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# The output image format for rendered math images.
imgmath_image_format = 'svg'

# Enable to output the class and the __init__ method docstring
autoclass_content = 'both'

# Enable figure numbering
numfig = True

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'pymiadoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'pymia.tex', f'{project} Documentation',
     author, 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, project, f'{project} Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, project, f'{project} Documentation',
     author, project, about['__description__'],
     'Miscellaneous'),
]
