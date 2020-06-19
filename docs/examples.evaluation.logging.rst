.. _example-evaluation2:

Logging the training progress
=============================

This example shows how to use the :mod:`pymia.evaluation` package to log the training progress in deep learning projects.

.. note::
   To be able to run this example:

    * Get the example data by executing `./examples/example-data/pull_example_data.py`.
    * Create the dataset by executing `./examples/data/create_dataset.py`.
    * You should have a basic understanding of the :mod:`pymia.data` package, see example :ref:`TODO(fabianbalsiger): title <example-data1>`.


.. literalinclude:: ../examples/evaluation/logging_torch.py

The source code is available under `./examples/evaluation/logging_torch.py`.