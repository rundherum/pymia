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
 - `NumPy <https://numpy.org/>`_
 - `scikit-image <https://scikit-image.org/>`_
 - `SciPy <https://www.scipy.org/>`_
 - `SimpleITK <https://simpleitk.org/>`_

.. note::
   For the :mod:`pymia.data` package, not all dependencies are installed directly due to their heaviness.
   Meaning, you need to either manually install PyTorch by

       - :bash:`pip install torch`

   or TensorFlow by

       - :bash:`pip install tensorflow`

   depending on your preferred deep learning framework when using the :mod:`pymia.data` package.
   Upon loading a module from the :mod:`pymia.data` package, pymia will always check if the required dependencies are fulfilled.

Building the documentation
--------------------------

Building the documentation requires the following packages:

 - `Sphinx <http://www.sphinx-doc.org>`_
 - `Read the Docs Sphinx Theme <https://sphinx-rtd-theme.readthedocs.io/en/stable/>`_
 - `nbsphinx <https://nbsphinx.readthedocs.io/en/latest/>`_
 - `Sphinx-copybutton <https://sphinx-copybutton.readthedocs.io/en/latest/>`_
 - `Jupyter <https://jupyterlab.readthedocs.io/en/stable/>`_

Install the required packages using pip:

.. code-block:: bash

   pip install sphinx
   pip install sphinx-rtd-theme
   pip install nbsphinx
   pip install sphinx-copybutton
   pip install jupyter

Run Sphinx in the pymia root directory to create the documentation:

   - :bash:`sphinx-build -b html ./docs ./docs/_build`
   - The documentation is now available under ``./docs/_build/index.html``

.. note::
   To build the documentation, it might be required to install `pandoc <https://pandoc.org/>`_.

   In case of the warning `WARNING: LaTeX command 'latex' cannot be run (needed for math display), check the imgmath_latex setting`, set the `imgmath_latex <http://www.sphinx-doc.org/en/master/usage/extensions/math.html#confval-imgmath_latex>`_ setting in the ``./docs/conf.py`` file.
