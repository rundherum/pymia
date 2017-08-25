"""
======================================
Evaluation (:mod:`evaluation` package)
======================================

This package provides an easy way to evaluate the performance of your algorithms.

We provide an :class:`evaluator.Evaluator` and two writers (:class:`evaluator.CSVEvaluatorWriter`)
and (:class:`evaluator.ConsoleEvaluatorWriter`), which can be used with a large amount of metrics
(see :mod:`evaluation.metric`).


This package provides a number of metric measures that e.g. can be used for testing
and/or evaluation purposes on two binary masks (i.e. measuring their similarity) or
distance between histograms.

The evaluator module (:mod:`evaluation.evaluator`)
--------------------------------------------------

.. autoclass:: evaluation.evaluator.Evaluator

.. autoclass:: evaluation.evaluator.CSVEvaluatorWriter

.. autoclass:: evaluation.evaluator.ConsoleEvaluatorWriter


The metric module (:mod:`evaluation.metric`)
********************************************

.. automodule:: evaluation.metric
    :members:

The validation module (:mod:`evaluation.validation`)
----------------------------------------------------

.. automodule:: evaluation.validation
    :members:

"""