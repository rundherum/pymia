import abc

import numpy as np


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


class Information(IMetric):
    """Represents an information.

    Can be used to add an additional column of information to an evaluator.
    """

    def __init__(self, column_name: str, value: str):
        """Initializes a new instance of the Information class."""
        super().__init__()
        self.metric = column_name
        self.value = value

    def calculate(self):
        """Outputs the value of the information."""
        return self.value
