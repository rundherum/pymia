"""The continuous module provides metrics to measure image reconstruction and regression performance."""
import warnings

import numpy as np
import skimage.metrics

from .base import (NumpyArrayMetric, NotComputableMetricWarning)


class MeanAbsoluteError(NumpyArrayMetric):

    def __init__(self, metric: str = 'MAE'):
        """Represents a mean absolute error metric.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the mean absolute error."""

        return np.mean(np.abs(self.reference - self.prediction))


class MeanSquaredError(NumpyArrayMetric):

    def __init__(self, metric: str = 'MSE'):
        """Represents a mean squared error metric.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the mean squared error."""

        return np.mean(np.square(self.reference - self.prediction))


class RootMeanSquaredError(NumpyArrayMetric):

    def __init__(self, metric: str = 'RMSE'):
        """Represents a root mean squared error metric.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the root mean squared error."""

        return np.sqrt(np.mean(np.square(self.reference - self.prediction)))


class NormalizedRootMeanSquaredError(NumpyArrayMetric):

    def __init__(self, metric: str = 'NRMSE'):
        """Represents a normalized root mean squared error metric.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the normalized root mean squared error."""

        rmse = np.sqrt(np.mean(np.square(self.reference - self.prediction)))
        return rmse / (self.reference.max() - self.reference.min())


class CoefficientOfDetermination(NumpyArrayMetric):

    def __init__(self, metric: str = 'R2'):
        """Represents a coefficient of determination (R^2) error metric.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the coefficient of determination (R^2) error.

        See Also:
            https://stackoverflow.com/a/45538060
        """
        y_true = self.reference.flatten()
        y_predicted = self.prediction.flatten()

        sse = sum((y_true - y_predicted) ** 2)
        tse = (len(y_true) - 1) * np.var(y_true, ddof=1)
        r2_score = 1 - (sse / tse)
        return r2_score


class PeakSignalToNoiseRatio(NumpyArrayMetric):

    def __init__(self, metric: str = 'PSNR'):
        """Represents a peak signal to noise ratio metric.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the peak signal to noise ratio."""
        psnr = skimage.metrics.peak_signal_noise_ratio(self.reference, self.prediction, data_range=self.reference.max())
        return psnr


class StructuralSimilarityIndexMeasure(NumpyArrayMetric):

    def __init__(self, metric: str = 'SSIM'):
        """Represents a structural similarity index measure metric.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the structural similarity index measure."""
        if self.reference.ndim == 2:
            ssim = skimage.metrics.structural_similarity(self.reference, self.prediction, data_range=self.reference.max())
        elif self.reference.ndim == 3:
            ssim = skimage.metrics.structural_similarity(self.reference, self.prediction, data_range=self.reference.max(),
                                                         multichannelbool=True)
        else:
            warnings.warn('Unable to compute StructuralSimilarityIndexMeasure for images of dimension other than 2 or 3.',
                          NotComputableMetricWarning)
            ssim = float('-inf')
        return ssim
