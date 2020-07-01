"""The reconstruction module provides metrics to measure image reconstruction performance."""
import warnings

import skimage.metrics

from .base import (NumpyArrayMetric, NotComputableMetricWarning)


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
