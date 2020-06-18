Metric (:mod:`pymia.evaluation.metric` package)
===============================================

The metric package provides metrics for evaluation of image segmentation, image reconstruction, and regression.

All metrics implement the :class:`pymia.evaluation.metric.base.IMetric` interface, and can be used with the
:mod:`pymia.evaluation.evaluator` package to evaluate results (e.g., with the :class:`pymia.evaluation.evaluator.Evaluator`).
To implement your own metric and use it with the :class:`pymia.evaluation.evaluator.Evaluator`, you need to inherit from
:class:`pymia.evaluation.metric.base.IMetric`, :class:`pymia.evaluation.metric.base.IConfusionMatrixMetric`,
:class:`pymia.evaluation.metric.base.IDistanceMetric`, :class:`pymia.evaluation.metric.base.ISimpleITKImageMetric` or
:class:`pymia.evaluation.metric.base.INumpyArrayMetric` and implement :func:`pymia.evaluation.metric.base.IMetric.calculate`.

.. note::
   The segmentation metrics are selected based on the paper by Taha and Hanbury. We recommend to refer to the paper for
   guidelines on how to select appropriate metrics, descriptions, and the math.

      Taha, A. A., & Hanbury, A. (2015). Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool.
      BMC Medical Imaging, 15. https://doi.org/10.1186/s12880-015-0068-x

.. note::
   The separation of the metrics into :mod:`pymia.evaluation.metric.segmentation` and :mod:`pymia.evaluation.metric.regression`
   is not strict. Meaning, a metric in :mod:`pymia.evaluation.metric.segmentation` could also be applied to regression.

Base (:mod:`pymia.evaluation.metric.base`) module
-------------------------------------------------

.. automodule:: pymia.evaluation.metric.base
    :members:
    :undoc-members:
    :show-inheritance:

Metric (:mod:`pymia.evaluation.metric.metric`) module
-----------------------------------------------------

.. automodule:: pymia.evaluation.metric.metric
    :members:
    :undoc-members:
    :show-inheritance:

Regression metrics (:mod:`pymia.evaluation.metric.regression`) module
---------------------------------------------------------------------

.. automodule:: pymia.evaluation.metric.regression
    :members:
    :undoc-members:
    :show-inheritance:

Segmentation metrics (:mod:`pymia.evaluation.metric.segmentation`) module
-------------------------------------------------------------------------

.. automodule:: pymia.evaluation.metric.segmentation
    :members:
    :undoc-members:
    :show-inheritance:
