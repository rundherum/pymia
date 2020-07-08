.. include:: ../README.rst

Main Features
=============
The main features of pymia are data handling (:mod:`pymia.data` package) and evaluation (:mod:`pymia.evaluation` package).
The intended use of pymia in the deep learning environment is depicted in :numref:`fig-overview`.
The data package is used to extract data (images, labels, demography, etc.) from a dataset in the desired format (2-D, 3-D; full- or patch-wise) for feeding to a neural network.
The output of the neural network is then assembled back to the original format before extraction, if necessary.
The evaluation package provides both evaluation routines as well as metrics to assess predictions against references.
Evaluation can be used both for stand-alone result calculation and reporting, and for monitoring of the training progress.
Further, pymia provides some basic image filtering and manipulation functionality (:mod:`pymia.filtering` package).
We recommend following our :ref:`examples <examples>`.

.. _fig-overview:
.. figure:: ./images/fig-overview.png
    :width: 800
    :alt: The pymia package in the deep learning environment.

    The pymia package in the deep learning environment. The data package allows to create a dataset from raw data. Extraction of the data from this dataset is possible in nearly every desired format (2-D, 3-D; full- or patch-wise) for feeding to a neural network. The prediction of the neural network can, if necessary, be assembled back to the format before extraction. The evaluation package allows to evaluate predictions against references using a vast amount of metrics. It can be used stand-alone (solid) or for performance monitoring during training (dashed).

Getting Started
===============

If you are new to pymia, here are a few guides to get you up to speed right away.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting started

    installation
    examples
    contribution
    history
    acknowledgment

* :doc:`installation` helps you installing pymia.

* :doc:`examples` give you an overview of pymia's intended use. Jupyter notebooks and Python scripts are available at `GitHub <https://github.com/rundherum/pymia/tree/master/examples/>`_.

* Do you want to contribute? See :doc:`contribution`.

* :doc:`history`.

* :doc:`acknowledgment`.

Citation
========
If you use pymia for your research, please acknowledge it accordingly by citing:

.. code-block:: none

    Jungo, A., Scheidegger, O., Reyes, M., & Balsiger, F. (2020). pymia: A Python package for data handling and evaluation in deep learning-based medical image analysis. ArXiv preprint.


BibTeX entry:

.. code-block:: none

    @article{Jungo2020a,
    archivePrefix = {arXiv},
    arxivId = {TBD},
    author = {Jungo, Alain and Scheidegger, Olivier and Reyes, Mauricio and Balsiger, Fabian},
    journal = {arXiv preprint},
    title = {{pymia: A Python package for data handling and evaluation in deep learning-based medical image analysis}},
    year = {2020}
    }

.. toctree::
    :maxdepth: 3
    :caption: Packages

    pymia.data
    pymia.evaluation
    pymia.filtering

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
