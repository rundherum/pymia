"""
======================================
Evaluation (:mod:`evaluation` package)
======================================

This package provides an easy way to evaluate the performance of your algorithms.

We provide an :class:`evaluator.Evaluator` and two writers (:class:`evaluator.CSVEvaluatorWriter`
and :class:`evaluator.ConsoleEvaluatorWriter`), which can be used with a large amount of metrics
(see :mod:`evaluation.metric`).


This package provides a number of metric measures that e.g. can be used for testing
and/or evaluation purposes on two binary masks (i.e. measuring their similarity) or
distance between histograms.

The evaluator module (:mod:`evaluation.evaluator`)
==================================================

.. autoclass:: evaluation.evaluator.Evaluator

.. autoclass:: evaluation.evaluator.CSVEvaluatorWriter

The metric module (:mod:`evaluation.metric`)
********************************************
The :mod:`metrics` module contains a set of metrics based on the paper of Taha 2016.
Refer to the paper for metrics selection, description, and math.

.. module:: evaluation.metric
.. autosummary::

    Accuracy
    AreaUnderCurve
    AverageDistance
    CohenKappaMetric
    DiceCoefficient
    Fallout
    FMeasure
    GlobalConsistencyError
    HausdorffDistance
    InterclassCorrelation
    JaccardCoefficient
    MahalanobisDistance
    MutualInformation
    Precision
    ProbabilisticDistance
    RandIndex
    Recall
    Sensitivity
    Specificity
    VariationOfInformation
    VolumeSimilarity

It is possible to implement your own metrics and use them with the :class:`evaluator.Evaluator`.
Just inherit from :class:`metric.IMetric` or :class:`metric.IConfusionMatrixMetric`
and implement :func:`metric.IMetric.calculate`.

The validation module (:mod:`evaluation.validation`)
====================================================

.. automodule:: evaluation.validation
    :members:

"""