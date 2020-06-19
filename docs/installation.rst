.. _installation:

.. role:: bash(code)
   :language: bash

Installation
============

Install pymia using pip (e.g., within a `Python virtual environment <https://www.geeksforgeeks.org/python-virtual-environment/>`_):

.. code-block:: bash

    pip install pymia

Alternatively, you can download or clone the code from `GitHub <https://github.com/rundherum/pymia>`_ and install pymia by

.. code-block:: bash

    git clone https://github.com/rundherum/pymia
    cd pymia
    python setup.py install

Dependencies
------------
pymia requires Python 3.6 (or higher) and depends on the following packages:

 - `h5py <https://www.h5py.org/>`_
 - `NumPy <http://www.numpy.org/>`_
 - `SciPy <https://www.scipy.org/>`_
 - `SimpleITK <http://www.simpleitk.org/>`_

.. note::
   For the :py:mod:`data` package, not all dependencies are installed directly due to their heaviness.
   Meaning, you need to either manually install PyTorch by

       - :bash:`pip install torch`

   or TensorFlow by

       - :bash:`pip install tensorflow`

   depending on your preferred deep learning framework when using the :py:mod:`data` package.
   Upon loading a module from the :py:mod:`data` package, pymia will always check if the required dependencies are fulfilled.

Building the documentation
--------------------------

Building the documentation requires:

 - `Sphinx <http://www.sphinx-doc.org>`_
 - `Read the Docs Sphinx Theme <https://github.com/rtfd/sphinx_rtd_theme>`_

#. Download or clone the code from `GitHub <https://github.com/rundherum/pymia>`_

#. Install Sphinx and other required packages using pip:

   - :bash:`pip install sphinx`
   - :bash:`pip install sphinx-rtd-theme`
   - :bash:`pip install nbsphinx`
   - :bash:`pip install sphinx-copybutton`

.. note::
   It might further be, that you need to install `pandoc <https://pandoc.org/>`_.

#. Run Sphinx in the pymia root directory to create the documentation

   - :bash:`sphinx-build -b html ./docs ./docs/_build`
   - The documentation is now available under ``./docs/_build/index.html``

In case of the warning ``WARNING: LaTeX command 'latex' cannot be run (needed for math display), check the imgmath_latex setting``,
set the `imgmath_latex <http://www.sphinx-doc.org/en/master/usage/extensions/math.html#confval-imgmath_latex>`_ setting in the ``./docs/conf.py`` file.
