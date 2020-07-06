Metric (:mod:`pymia.evaluation.metric` package)
===============================================

The metric package provides metrics for evaluation of image segmentation, image reconstruction, and regression.

All metrics implement the :class:`pymia.evaluation.metric.base.Metric` interface, and can be used with the
:mod:`pymia.evaluation.evaluator` package to evaluate results
(e.g., with the :class:`pymia.evaluation.evaluator.SegmentationEvaluator`).
To implement your own metric and use it with the :class:`pymia.evaluation.evaluator.Evaluator`, you need to inherit from
:class:`pymia.evaluation.metric.base.Metric`, :class:`pymia.evaluation.metric.base.ConfusionMatrixMetric`,
:class:`pymia.evaluation.metric.base.DistanceMetric`, :class:`pymia.evaluation.metric.base.SimpleITKImageMetric` or
:class:`pymia.evaluation.metric.base.NumpyArrayMetric` and implement :func:`pymia.evaluation.metric.base.Metric.calculate`.

.. note::
   The segmentation metrics are selected based on the paper by Taha and Hanbury. We recommend to refer to the paper for
   guidelines on how to select appropriate metrics, descriptions, and the math.

      Taha, A. A., & Hanbury, A. (2015). Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool.
      BMC Medical Imaging, 15. https://doi.org/10.1186/s12880-015-0068-x


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

Binary metrics (:mod:`pymia.evaluation.metric.binary`) module
-------------------------------------------------------------

.. automodule:: pymia.evaluation.metric.binary
    :members:
    :undoc-members:
    :show-inheritance:

Continuous metrics (:mod:`pymia.evaluation.metric.continuous`) module
---------------------------------------------------------------------

.. automodule:: pymia.evaluation.metric.continuous
    :members:
    :undoc-members:
    :show-inheritance:
