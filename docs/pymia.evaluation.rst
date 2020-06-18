Evaluation (:mod:`pymia.evaluation` package)
============================================

The evaluation package provides metrics and evaluation functionalities.

pymia provides a large amount of metrics for image segmentation, image reconstruction, and regression in the
:mod:`pymia.evaluation.metric.metric` package. All metrics implement the
:class:`pymia.evaluation.metric.base.Metric` interface, and can be used with the :mod:`pymia.evaluation.evaluator` package
to evaluate results (e.g., with the :class:`pymia.evaluation.evaluator.SegmentationEvaluator`).
The :mod:`pymia.evaluation.writer` package provides several writers to report the results, and statistics of the results,
to CSV files (e.g., the :class:`pymia.evaluation.writer.CSVWriter` and :class:`pymia.evaluation.writer.CSVStatisticsWriter`)
and the console (e.g., the :class:`pymia.evaluation.writer.ConsoleWriter` and
:class:`pymia.evaluation.writer.ConsoleStatisticsWriter`).

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
