"""The metric module provides a set of metrics."""
from .regression import (CoefficientOfDetermination, MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError,
                         NormalizedRootMeanSquaredError)
from .segmentation import (Accuracy, AdjustedRandIndex, AreaUnderCurve, AverageDistance, CohenKappaMetric,
                           DiceCoefficient, FalseNegative, FalsePositive, Fallout, FMeasure,
                           GlobalConsistencyError, GroundTruthVolume, HausdorffDistance,
                           InterclassCorrelation, JaccardCoefficient, MahalanobisDistance, MutualInformation, Precision,
                           ProbabilisticDistance, RandIndex, Recall, SegmentationVolume,
                           Sensitivity, Specificity, SurfaceOverlap, SurfaceDiceOverlap, TrueNegative, TruePositive,
                           VariationOfInformation, VolumeSimilarity)


def get_all_segmentation_metrics():
    """Gets a list with all segmentation metrics.

    Returns:
        list[IMetric]: A list of metrics.
    """
    return get_overlap_metrics() + get_distance_metrics() + get_classical_metrics()


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
    return [AdjustedRandIndex(),
            AreaUnderCurve(),
            CohenKappaMetric(),
            DiceCoefficient(),
            InterclassCorrelation(),
            JaccardCoefficient(),
            MutualInformation(),
            RandIndex(),
            SurfaceOverlap(),
            SurfaceDiceOverlap(),
            VolumeSimilarity()]


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
