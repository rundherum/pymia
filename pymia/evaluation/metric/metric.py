"""The metric module contains a set of evaluation metrics.

The metrics are selected based on the paper of Taha 2016.
Refer to the paper for guidelines how to select appropriate metrics, in-depth description,
and the math.

It is possible to implement your own metrics and use them with the :class:`pymia.evaluation.evaluator.Evaluator`.
Just inherit from :class:`pymia.evaluation.metric.base.IMetric`,
:class:`pymia.evaluation.metric.base.IConfusionMatrixMetric`,
:class:`pymia.evaluation.metric.base.IDistanceMetric`,
:class:`pymia.evaluation.metric.base.ISimpleITKImageMetric` or
:class:`pymia.evaluation.metric.base.INumpyArrayMetric`
and implement the function :func:`pymia.evaluation.metric.base.IMetric.calculate`.
"""
from .regression import (CoefficientOfDetermination, MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError,
                         NormalizedRootMeanSquaredError)
from .segmentation import (Accuracy, AdjustedRandIndex, AreaUnderCurve, AverageDistance, CohenKappaMetric,
                           DiceCoefficient, FalseNegative, FalsePositive, Fallout, FMeasure,
                           GlobalConsistencyError, GroundTruthVolume, HausdorffDistance,
                           InterclassCorrelation, JaccardCoefficient, MahalanobisDistance, MutualInformation, Precision,
                           ProbabilisticDistance, RandIndex, Recall, SegmentationVolume,
                           Sensitivity, Specificity, TrueNegative, TruePositive,
                           VariationOfInformation, VolumeSimilarity)


def get_all_segmentation_metrics():
    """Gets a list with all segmentation metrics.

    Returns:
        list[IMetric]: A list of metrics.
    """
    return get_overlap_metrics() + get_distance_metrics() + get_distance_metrics()


def get_all_regression_metrics():
    """Gets a list with all regression metrics.

    Returns:
        list[IMetric]: A list of metrics.
    """
    return [CoefficientOfDetermination(), MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError(),
            NormalizedRootMeanSquaredError()]


def get_overlap_metrics():
    """Gets a list of overlap-based metrics.

    Returns:
        list[IMetric]: A list of metrics.
    """
    return [DiceCoefficient(),
            JaccardCoefficient(),
            AreaUnderCurve(),
            CohenKappaMetric(),
            RandIndex(),
            AdjustedRandIndex(),
            InterclassCorrelation(),
            VolumeSimilarity(),
            MutualInformation()]


def get_distance_metrics():
    """Gets a list of distance-based metrics.

    Returns:
        list[IMetric]: A list of metrics.
    """

    return [HausdorffDistance(),
            AverageDistance(),
            MahalanobisDistance(),
            VariationOfInformation(),
            GlobalConsistencyError(),
            ProbabilisticDistance()]


def get_classical_metrics():
    """Gets a list of classical metrics.

    Returns:
        list[IMetric]: A list of metrics.
    """

    return[Sensitivity(),
           Specificity(),
           Precision(),
           Recall(),
           FMeasure(),
           Accuracy(),
           Fallout(),
           TruePositive(),
           FalsePositive(),
           TrueNegative(),
           FalseNegative(),
           GroundTruthVolume(),
           SegmentationVolume()]
