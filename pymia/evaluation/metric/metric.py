"""The metric module provides a set of metrics."""
from .categorical import (Accuracy, AdjustedRandIndex, AreaUnderCurve, AverageDistance, CohenKappaCoefficient,
                          DiceCoefficient, FalseNegative, FalsePositive, Fallout, FalseNegativeRate, FMeasure,
                          GlobalConsistencyError, HausdorffDistance,
                          InterclassCorrelation, JaccardCoefficient, MahalanobisDistance, MutualInformation, Precision,
                          PredictionVolume, ProbabilisticDistance, RandIndex, ReferenceVolume,
                          Sensitivity, Specificity, SurfaceDiceOverlap, SurfaceOverlap, TrueNegative, TruePositive,
                          VariationOfInformation, VolumeSimilarity)
from .continuous import (CoefficientOfDetermination, MeanAbsoluteError, MeanSquaredError, NormalizedRootMeanSquaredError,
                         PeakSignalToNoiseRatio, RootMeanSquaredError, StructuralSimilarityIndexMeasure)


def get_reconstruction_metrics():
    """Gets a list with reconstruction metrics.

    Returns:
        list[Metric]: A list of metrics.
    """
    return [PeakSignalToNoiseRatio(), StructuralSimilarityIndexMeasure()]


def get_segmentation_metrics():
    """Gets a list with segmentation metrics.

    Returns:
        list[Metric]: A list of metrics.
    """
    return get_overlap_metrics() + get_distance_metrics() + get_classical_metrics()


def get_regression_metrics():
    """Gets a list with regression metrics.

    Returns:
        list[Metric]: A list of metrics.
    """
    return [CoefficientOfDetermination(), MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError(),
            NormalizedRootMeanSquaredError()]


def get_overlap_metrics():
    """Gets a list of overlap-based metrics.

    Returns:
        list[Metric]: A list of metrics.
    """
    return [AdjustedRandIndex(),
            AreaUnderCurve(),
            CohenKappaCoefficient(),
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
        list[Metric]: A list of metrics.
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
        list[Metric]: A list of metrics.
    """

    return[Sensitivity(),
           Specificity(),
           Precision(),
           FMeasure(),
           Accuracy(),
           Fallout(),
           FalseNegativeRate(),
           TruePositive(),
           FalsePositive(),
           TrueNegative(),
           FalseNegative(),
           ReferenceVolume(),
           PredictionVolume()]
