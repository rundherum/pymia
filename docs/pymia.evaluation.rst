Evaluation (:mod:`pymia.evaluation` package)
============================================

This package provides an easy way to evaluate the performance of your algorithms.

We provide a large amount of metrics for image segmentation, image reconstruction, and regression in the
:mod:`pymia.evaluation.metric` package. The metrics can easily be used by the
:class:`pymia.evaluation.evaluator.Evaluator` to evaluate results. The :mod:`pymia.evaluation.writer` package provides
several writers to report the results, and statistics of the results, to CSV files and the console.

Subpackages
-----------

.. toctree::

    pymia.evaluation.metric

The evaluator module (:mod:`pymia.evaluation.evaluator`)
--------------------------------------------------------

.. automodule:: pymia.evaluation.evaluator
    :members:
    :undoc-members:
    :show-inheritance:

The writer module (:mod:`pymia.evaluation.writer`)
----------------------------------------------------------

.. automodule:: pymia.evaluation.writer
    :members:
    :undoc-members:
    :show-inheritance:
