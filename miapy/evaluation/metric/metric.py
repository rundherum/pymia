"""The metric module contains a set of evaluation metrics.

The metrics are selected based on the paper of Taha 2016.
Refer to the paper for guidelines how to select appropriate metrics, in-depth description,
and the math.

It is possible to implement your own metrics and use them with the :class:`evaluator.Evaluator`.
Just inherit from :class:`metric.IMetric`, :class:`metric.IConfusionMatrixMetric` or :class:`ISimpleITKImageMetric`
and implement the function :func:`calculate`.
"""
import abc

import numpy as np

from .regression import (MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError)
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
    return [MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError()]


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


class ConfusionMatrix:
    """Represents a confusion matrix (or error matrix)."""

    def __init__(self, prediction, label):
        """Initializes a new instance of the ConfusionMatrix class."""

        # true positive (tp): we predict a label of 1 (positive), and the true label is 1
        self.tp = np.sum(np.logical_and(prediction == 1, label == 1))
        # true negative (tn): we predict a label of 0 (negative), and the true label is 0
        self.tn = np.sum(np.logical_and(prediction == 0, label == 0))
        # false positive (fp): we predict a label of 1 (positive), but the true label is 0
        self.fp = np.sum(np.logical_and(prediction == 1, label == 0))
        # false negative (fn): we predict a label of 0 (negative), but the true label is 1
        self.fn = np.sum(np.logical_and(prediction == 0, label == 1))

        self.n = prediction.size


class IMetric(metaclass=abc.ABCMeta):
    """Represents an evaluation metric."""

    def __init__(self):
        self.metric = 'IMetric'

    @abc.abstractmethod
    def calculate(self):
        """Calculates the metric."""

        raise NotImplementedError

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return '{self.metric}' \
            .format(self=self)


class IConfusionMatrixMetric(IMetric):
    """Represents an evaluation metric based on the confusion matrix."""

    def __init__(self):
        """Initializes a new instance of the IConfusionMatrixMetric class."""
        super().__init__()
        self.metric = 'IConfusionMatrixMetric'
        self.confusion_matrix = None  # ConfusionMatrix

    @abc.abstractmethod
    def calculate(self):
        """Calculates the metric."""

        raise NotImplementedError


class ISimpleITKImageMetric(IMetric):
    """Represents an evaluation metric based on SimpleITK images."""

    def __init__(self):
        """Initializes a new instance of the ISimpleITKImageMetric class."""
        super(ISimpleITKImageMetric, self).__init__()
        self.metric = 'ISimpleITKImageMetric'
        self.ground_truth = None  # SimpleITK.Image
        self.segmentation = None  # SimpleITK.Image

    @abc.abstractmethod
    def calculate(self):
        """Calculates the metric."""

        raise NotImplementedError


class INumpyArrayMetric(IMetric):
    """Represents an evaluation metric based on numpy arrays."""

    def __init__(self):
        """Initializes a new instance of the INumpyArrayMetric class."""
        super(INumpyArrayMetric, self).__init__()
        self.metric = 'INumpyArrayMetric'
        self.ground_truth = None  # np.ndarray
        self.segmentation = None  # np.ndarray

    @abc.abstractmethod
    def calculate(self):
        """Calculates the metric."""

        raise NotImplementedError
