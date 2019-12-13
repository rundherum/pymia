.. _installation:

.. role:: bash(code)
   :language: bash

Installation
============

Install pymia using pip (e.g., within a virtualenv):

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
 - `Matplotlib <https://matplotlib.org/>`_
 - `NumPy <http://www.numpy.org/>`_
 - `SciPy <https://www.scipy.org/>`_
 - `SimpleITK <http://www.simpleitk.org/>`_
 - `PyYAML <https://pyyaml.org/>`_
 - `PyTorch <https://pytorch.org/>`_
 - `TensorFlow <https://www.tensorflow.org/>`_
 - `VTK <https://www.vtk.org/>`_

Note that not all dependencies are installed directly but only required when certain modules are used.
Upon loading a module, pymia will check if the dependencies are fulfilled.

Building the documentation
--------------------------

Building the documentation requires:

 - `Sphinx <http://www.sphinx-doc.org>`_
 - `Read the Docs Sphinx Theme <https://github.com/rtfd/sphinx_rtd_theme>`_

#. Download or clone the code from `GitHub <https://github.com/rundherum/pymia>`_

#. Install Sphinx and the RTD theme using pip:

   - :bash:`pip install sphinx`
   - :bash:`pip install sphinx-rtd-theme`

#. Run Sphinx in the pymia root directory to create the documentation

   - :bash:`sphinx-build -b html ./docs ./docs/_build`
   - The documentation is now available under ``./docs/_build/index.html``

In case of the warning ``WARNING: LaTeX command 'latex' cannot be run (needed for math display), check the imgmath_latex setting``,
set the `imgmath_latex <http://www.sphinx-doc.org/en/master/usage/extensions/math.html#confval-imgmath_latex>`_ setting in the ``./docs/conf.py`` file.
