"""The pymia.evaluation.metric.regression module provides several metrics to measure regression performance."""

import numpy as np

from .base import INumpyArrayMetric


class MeanAbsoluteError(INumpyArrayMetric):
    """Represents a mean absolute error metric."""

    def __init__(self, metric: str = 'MAE'):
        """Initializes a new instance of the MeanAbsoluteError class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the mean absolute error."""

        return np.mean(np.abs(self.ground_truth - self.segmentation))


class MeanSquaredError(INumpyArrayMetric):
    """Represents a mean squared error metric."""

    def __init__(self, metric: str = 'MSE'):
        """Initializes a new instance of the MeanSquaredError class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the mean squared error."""

        return np.mean(np.square(self.ground_truth - self.segmentation))


class RootMeanSquaredError(INumpyArrayMetric):
    """Represents a root mean squared error metric."""

    def __init__(self, metric: str = 'RMSE'):
        """Initializes a new instance of the RootMeanSquaredError class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the root mean squared error."""

        return np.sqrt(np.mean(np.square(self.ground_truth - self.segmentation)))


class NormalizedRootMeanSquaredError(INumpyArrayMetric):
    """Represents a normalized root mean squared error metric."""

    def __init__(self, metric: str = 'NRMSE'):
        """Initializes a new instance of the NormalizedRootMeanSquaredError class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the normalized root mean squared error."""

        rmse = np.sqrt(np.mean(np.square(self.ground_truth - self.segmentation)))
        return rmse / (self.ground_truth.max() - self.ground_truth.min())


class CoefficientOfDetermination(INumpyArrayMetric):
    """Represents a coefficient of determination (R^2) error metric."""

    def __init__(self, metric: str = 'R2'):
        """Initializes a new instance of the CoefficientOfDetermination class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the coefficient of determination (R^2) error.

        See Also:
            https://stackoverflow.com/a/45538060
        """
        y_true = self.ground_truth.flatten()
        y_predicted = self.segmentation.flatten()

        sse = sum((y_true - y_predicted) ** 2)
        tse = (len(y_true) - 1) * np.var(y_true, ddof=1)
        r2_score = 1 - (sse / tse)
        return r2_score
