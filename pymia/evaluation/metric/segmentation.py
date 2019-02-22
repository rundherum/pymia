import abc
import math
import warnings

import numpy as np
import SimpleITK as sitk

from .base import (IConfusionMatrixMetric, IDistanceMetric, ISimpleITKImageMetric, INumpyArrayMetric,
                   NotComputableMetricWarning)


class AreaMetric(ISimpleITKImageMetric):
    """Represents an area metric."""

    def __init__(self, metric: str = 'AREA'):
        """Initializes a new instance of the AreaMetric class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    @abc.abstractmethod
    def calculate(self):
        """Calculates the metric."""
        raise NotImplementedError

    @staticmethod
    def _calculate_area(image: sitk.Image, slice_number: int = -1) -> float:
        """Calculates the area of a slice in a label image.

        Args:
            image (sitk.Image): The 3-D label image.
            slice_number (int): The slice number to calculate the area.
                Defaults to -1, which will calculate the area on the intermediate slice.
        """

        img_arr = sitk.GetArrayFromImage(image)

        if slice_number == -1:
            slice_number = int(img_arr.shape[0] / 2)  # use the intermediate slice

        return img_arr[slice_number, ...].sum() * image.GetSpacing()[0] * image.GetSpacing()[1]


class VolumeMetric(ISimpleITKImageMetric):
    """Represents a volume metric."""

    def __init__(self, metric: str = 'VOL'):
        """Initializes a new instance of the VolumeMetric class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    @abc.abstractmethod
    def calculate(self):
        """Calculates the metric."""
        raise NotImplementedError

    @staticmethod
    def _calculate_volume(image: sitk.Image) -> float:
        """Calculates the volume of a label image.

        Args:
            image (sitk.Image): The 3-D label image.
        """

        voxel_volume = np.prod(image.GetSpacing())
        number_of_voxels = sitk.GetArrayFromImage(image).sum()

        return number_of_voxels * voxel_volume


class Accuracy(IConfusionMatrixMetric):
    """Represents an accuracy metric."""

    def __init__(self, metric: str = 'ACURCY'):
        """Initializes a new instance of the Accuracy class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the accuracy."""

        sum_ = self.confusion_matrix.tp + self.confusion_matrix.tn + self.confusion_matrix.fp + self.confusion_matrix.fn

        if sum_ != 0:
            return (self.confusion_matrix.tp + self.confusion_matrix.tn) / sum_
        else:
            return 0


class AdjustedRandIndex(IConfusionMatrixMetric):
    """Represents an adjusted rand index metric."""

    def __init__(self, metric: str = 'ADJRIND'):
        """Initializes a new instance of the AdjustedRandIndex class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the adjusted rand index."""

        tp = self.confusion_matrix.tp
        tn = self.confusion_matrix.tn
        fp = self.confusion_matrix.fp
        fn = self.confusion_matrix.fn
        n = self.confusion_matrix.n

        fp_tn = tn + fp
        tp_fn = fn + tp
        tn_fn = tn + fn
        tp_fp = fp + tp
        nis = tn_fn * tn_fn + tp_fp * tp_fp
        njs = fp_tn * fp_tn + tp_fn * tp_fn
        sum_of_squares = tp * tp + tn * tn + fp * fp + fn * fn

        a = (tp * (tp - 1) + fp * (fp - 1) + tn * (tn - 1) + fn * (fn - 1)) / 2.
        b = (njs - sum_of_squares) / 2.
        c = (nis - sum_of_squares) / 2.
        d = (n * n + sum_of_squares - nis - njs) / 2.

        x1 = a - ((a + c) * (a + b) / (a + b + c + d))
        x2 = ((a + c) + (a + b)) / 2.
        x3 = ((a + c) * (a + b)) / (a + b + c + d)
        denominator = x2 - x3

        if denominator != 0:
            return x1 / denominator
        else:
            return 0


class AreaUnderCurve(IConfusionMatrixMetric):
    """Represents an area under the curve metric."""

    def __init__(self, metric: str = 'AUC'):
        """Initializes a new instance of the AreaUnderCurve class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the area under the curve."""

        specificity = self.confusion_matrix.tn / (self.confusion_matrix.tn + self.confusion_matrix.fp)

        false_positive_rate = 1 - specificity

        true_positive_rate = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)

        return (true_positive_rate - false_positive_rate + 1) / 2


class AverageDistance(ISimpleITKImageMetric):
    """Represents an average (Hausdorff) distance metric.

        Calculates the distance between the set of non-zero pixels of two images using the following equation:

        .. math:: AVD(A,B) = max(d(A,B), d(B,A)),

        where

        .. math:: d(A,B) = \\frac{1}{N} \\sum_{a \\in A} \\min_{b \\in B} \\lVert a - b \\rVert

        is the directed Hausdorff distance and :math:`A` and :math:`B` are the set of non-zero pixels in the images.
        """

    def __init__(self, metric: str = 'AVGDIST'):
        """Initializes a new instance of the AverageDistance class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the average (Hausdorff) distance."""

        if np.count_nonzero(sitk.GetArrayFromImage(self.ground_truth)) == 0:
            warnings.warn('Unable to compute average distance due to empty label mask, returning inf',
                          NotComputableMetricWarning)
            return float('inf')
        if np.count_nonzero(sitk.GetArrayFromImage(self.segmentation)) == 0:
            warnings.warn('Unable to compute average distance due to empty segmentation mask, returning inf',
                          NotComputableMetricWarning)
            return float('inf')

        distance_filter = sitk.HausdorffDistanceImageFilter()
        distance_filter.Execute(self.ground_truth, self.segmentation)
        return distance_filter.GetAverageHausdorffDistance()


class CohenKappaMetric(IConfusionMatrixMetric):
    """Represents a Cohen's kappa coefficient metric."""

    def __init__(self, metric: str = 'KAPPA'):
        """Initializes a new instance of the CohenKappaMetric class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the Cohen's kappa coefficient."""

        tp = self.confusion_matrix.tp
        tn = self.confusion_matrix.tn
        fp = self.confusion_matrix.fp
        fn = self.confusion_matrix.fn

        agreement = tp + tn
        chance0 = (tn + fn) * (tn + fp)
        chance1 = (fp + tp) * (fn + tp)
        sum_ = tn + fn + fp + tp
        chance = (chance0 + chance1) / sum_

        return (agreement - chance) / (sum_ - chance)


class DiceCoefficient(IConfusionMatrixMetric):
    """Represents a Dice coefficient metric."""

    def __init__(self, metric: str = 'DICE'):
        """Initializes a new instance of the DiceCoefficient class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the Dice coefficient."""

        return 2 * self.confusion_matrix.tp / \
               (2 * self.confusion_matrix.tp + self.confusion_matrix.fp + self.confusion_matrix.fn)


class FalseNegative(IConfusionMatrixMetric):
    """Represents a false negative metric."""

    def __init__(self, metric: str = 'FN'):
        """Initializes a new instance of the FalseNegative class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the false negatives."""

        return self.confusion_matrix.fn


class FalsePositive(IConfusionMatrixMetric):
    """Represents a false positive metric."""

    def __init__(self, metric: str = 'FP'):
        """Initializes a new instance of the FalsePositive class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the false positives."""

        return self.confusion_matrix.fp


class Fallout(IConfusionMatrixMetric):
    """Represents a fallout (false positive rate) metric."""

    def __init__(self, metric: str = 'FALLOUT'):
        """Initializes a new instance of the Fallout class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the fallout (false positive rate)."""

        specificity = self.confusion_matrix.tn / (self.confusion_matrix.tn + self.confusion_matrix.fp)
        return 1 - specificity


class FalseNegativeRate(IConfusionMatrixMetric):
    """Represents a false negative rate metric."""

    def __init__(self, metric: str = 'FNR'):
        """Initializes a new instance of the FalseNegativeRate class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the false negative rate."""

        sensitivity = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        return 1 - sensitivity


class FMeasure(IConfusionMatrixMetric):
    """Represents a F-measure metric."""

    def __init__(self, metric: str = 'FMEASR'):
        """Initializes a new instance of the FMeasure class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the F1 measure."""

        beta = 1  # or 0.5 or 2 can also calculate F2 or F0.5 measure

        beta_squared = beta * beta
        precision = Precision()
        precision.confusion_matrix = self.confusion_matrix
        precision = precision.calculate()
        recall = Recall()
        recall.confusion_matrix = self.confusion_matrix
        recall = recall.calculate()

        denominator = beta_squared * precision + recall

        if denominator != 0:
            return (1 + beta_squared) * ((precision * recall) / denominator)
        else:
            return 0


class GlobalConsistencyError(IConfusionMatrixMetric):
    """Represents a global consistency error metric.

    Implementation based on Martin 2001.
    """

    def __init__(self, metric: str = 'GCOERR'):
        """Initializes a new instance of the GlobalConsistencyError class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the global consistency error."""

        tp = self.confusion_matrix.tp
        tn = self.confusion_matrix.tn
        fp = self.confusion_matrix.fp
        fn = self.confusion_matrix.fn

        n = tp + tn + fp + fn
        e1 = (fn * (fn + 2 * tp) / (tp + fn) + fp * (fp + 2 * tn) / (tn + fp)) / n
        e2 = (fp * (fp + 2 * tp) / (tp + fp) + fn * (fn + 2 * tn) / (tn + fn)) / n

        return min(e1, e2)


class GroundTruthArea(AreaMetric):
    """Represents a ground truth area metric."""

    def __init__(self, slice_number: int = -1, metric: str = 'GTAREA'):
        """Initializes a new instance of the GroundTruthArea class.

        Args:
            slice_number (int): The slice number to calculate the area.
                Defaults to -1, which will calculate the area on the intermediate slice.
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)
        self.slice_number = slice_number

    def calculate(self):
        """Calculates the ground truth area on a specified slice in mm2."""

        return self._calculate_area(self.ground_truth, self.slice_number)


class GroundTruthVolume(VolumeMetric):
    """Represents a ground truth volume metric."""

    def __init__(self, metric: str = 'GTVOL'):
        """Initializes a new instance of the GroundTruthVolume class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the ground truth volume in mm3."""

        return self._calculate_volume(self.ground_truth)


class HausdorffDistance(IDistanceMetric):
    """Represents a Hausdorff distance metric.

    Calculates the distance between the set of non-zero pixels of two images using the following equation:

    .. math:: H(A,B) = max(h(A,B), h(B,A)),

    where

    .. math:: h(A,B) = \\max_{a \\in A} \\min_{b \\in B} \\lVert a - b \\rVert

    is the directed Hausdorff distance and :math:`A` and :math:`B` are the set of non-zero pixels in the images.

    See Also:
        - Nikolov et al., 2018 Deep learning to achieve clinically applicable segmentation of head and neck anatomy for
            radiotherapy. `arXiv <https://arxiv.org/abs/1809.04430>`_
        - `Original implementation <https://github.com/deepmind/surface-distance>`_
    """

    def __init__(self, percentile: float = 100.0, metric: str = 'HDRFDST'):
        """Initializes a new instance of the HausdorffDistance class.

        Args:
            percentile (float): The percentile (0, 100] to compute, i.e. 100 computes the Hausdorff distance and
                95 computes the 95th Hausdorff distance.
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)
        self.percentile = percentile

    def calculate(self):
        """Calculates the Hausdorff distance."""

        if self.distances.distances_gt_to_pred is not None and len(self.distances.distances_pred_to_gt) > 0:
            surfel_areas_cum_gt = np.cumsum(self.distances.surfel_areas_gt) / np.sum(self.distances.surfel_areas_gt)
            idx = np.searchsorted(surfel_areas_cum_gt, self.percentile / 100.0)
            perc_distance_gt_to_pred = self.distances.distances_gt_to_pred[
                min(idx, len(self.distances.distances_gt_to_pred) - 1)]
        else:
            warnings.warn('Unable to compute Hausdorff distance due to empty label mask, returning inf',
                          NotComputableMetricWarning)
            return float('inf')

        if self.distances.distances_pred_to_gt is not None and len(self.distances.distances_pred_to_gt) > 0:
            surfel_areas_cum_pred = (np.cumsum(self.distances.surfel_areas_pred) / 
                                     np.sum(self.distances.surfel_areas_pred))
            idx = np.searchsorted(surfel_areas_cum_pred, self.percentile / 100.0)
            perc_distance_pred_to_gt = self.distances.distances_pred_to_gt[
                min(idx, len(self.distances.distances_pred_to_gt) - 1)]
        else:
            warnings.warn('Unable to compute Hausdorff distance due to empty segmentation mask, returning inf',
                          NotComputableMetricWarning)
            return float('inf')

        return max(perc_distance_gt_to_pred, perc_distance_pred_to_gt)


class InterclassCorrelation(INumpyArrayMetric):
    """Represents a interclass correlation metric."""

    def __init__(self, metric: str = 'ICCORR'):
        """Initializes a new instance of the InterclassCorrelation class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the interclass correlation."""

        gt = self.ground_truth.flatten()
        seg = self.segmentation.flatten()

        n = gt.size
        mean_gt = gt.mean()
        mean_seg = seg.mean()
        mean = (mean_gt + mean_seg) / 2

        m = (gt + seg) / 2
        ssw = np.power(gt - m, 2).sum() + np.power(seg - m, 2).sum()
        ssb = np.power(m - mean, 2).sum()

        ssw /= n
        ssb = ssb / (n - 1) * 2

        return (ssb - ssw) / (ssb + ssw)


class JaccardCoefficient(IConfusionMatrixMetric):
    """Represents a Jaccard coefficient metric."""

    def __init__(self, metric: str = 'JACRD'):
        """Initializes a new instance of the JaccardCoefficient class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the Jaccard coefficient."""

        tp = self.confusion_matrix.tp
        fp = self.confusion_matrix.fp
        fn = self.confusion_matrix.fn

        return tp / (tp + fp + fn)


class MahalanobisDistance(INumpyArrayMetric):
    """Represents a Mahalanobis distance metric."""

    def __init__(self, metric: str = 'MAHLNBS'):
        """Initializes a new instance of the MahalanobisDistance class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the Mahalanobis distance."""

        gt_n = np.count_nonzero(self.ground_truth)
        gt_indices = np.flip(np.where(self.ground_truth == 1), axis=0)
        gt_mean = gt_indices.mean(axis=1)
        gt_cov = np.cov(gt_indices)

        seg_n = np.count_nonzero(self.segmentation)
        seg_indices = np.flip(np.where(self.segmentation == 1), axis=0)
        seg_mean = seg_indices.mean(axis=1)
        seg_cov = np.cov(seg_indices)

        # calculate common covariance matrix
        common_cov = (gt_n * gt_cov + seg_n * seg_cov) / (gt_n + seg_n)
        common_cov_inv = np.linalg.inv(common_cov)

        mean = np.matrix(np.array(gt_mean) - np.array(seg_mean))

        return math.sqrt(mean * np.matrix(common_cov_inv) * mean.T)


class MutualInformation(IConfusionMatrixMetric):
    """Represents a mutual information metric."""

    def __init__(self, metric: str = 'MUTINF'):
        """Initializes a new instance of the MutualInformation class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the mutual information."""

        tp = self.confusion_matrix.tp
        tn = self.confusion_matrix.tn
        fp = self.confusion_matrix.fp
        fn = self.confusion_matrix.fn
        n = self.confusion_matrix.n

        fn_tp = fn + tp
        fp_tp = fp + tp
        
        if fn_tp == 0 or fn_tp / n == 1 or fp_tp == 0 or fp_tp / n == 1:
            warnings.warn('Unable to compute mutual information due to log2 of 0, returning -inf',
                          NotComputableMetricWarning)
            return float('-inf')

        h1 = -((fn_tp / n) * math.log2(fn_tp / n) + (1 - fn_tp / n) * math.log2(1 - fn_tp / n))
        h2 = -((fp_tp / n) * math.log2(fp_tp / n) + (1 - fp_tp / n) * math.log2(1 - fp_tp / n))

        p00 = 1 if tn == 0 else (tn / n)
        p01 = 1 if fn == 0 else (fn / n)
        p10 = 1 if fp == 0 else (fp / n)
        p11 = 1 if tp == 0 else (tp / n)

        h12 = -((tn / n) * math.log2(p00) +
                (fn / n) * math.log2(p01) +
                (fp / n) * math.log2(p10) +
                (tp / n) * math.log2(p11))

        mi = h1 + h2 - h12
        return mi


class Precision(IConfusionMatrixMetric):
    """Represents a precision metric."""

    def __init__(self, metric: str = 'PRCISON'):
        """Initializes a new instance of the Precision class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the precision."""

        sum_ = self.confusion_matrix.tp + self.confusion_matrix.fp

        if sum_ != 0:
            return self.confusion_matrix.tp / sum_
        else:
            return 0


class ProbabilisticDistance(INumpyArrayMetric):
    """Represents a probabilistic distance metric."""

    def __init__(self, metric: str = 'PROBDST'):
        """Initializes a new instance of the ProbabilisticDistance class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the probabilistic distance."""

        gt = self.ground_truth.flatten().astype(np.int8)
        seg = self.segmentation.flatten().astype(np.int8)

        probability_difference = np.absolute(gt - seg).sum()
        probability_joint = (gt * seg).sum()

        if probability_joint != 0:
            return probability_difference / (2. * probability_joint)
        else:
            return -1


class RandIndex(IConfusionMatrixMetric):
    """Represents a rand index metric."""

    def __init__(self, metric: str = 'RNDIND'):
        """Initializes a new instance of the RandIndex class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the rand index."""

        tp = self.confusion_matrix.tp
        tn = self.confusion_matrix.tn
        fp = self.confusion_matrix.fp
        fn = self.confusion_matrix.fn
        n = self.confusion_matrix.n

        fp_tn = tn + fp
        tp_fn = fn + tp
        tn_fn = tn + fn
        tp_fp = fp + tp
        nis = tn_fn * tn_fn + tp_fp * tp_fp
        njs = fp_tn * fp_tn + tp_fn * tp_fn
        sum_of_squares = tp * tp + tn * tn + fp * fp + fn * fn

        a = (tp * (tp - 1) + fp * (fp - 1) + tn * (tn - 1) + fn * (fn - 1)) / 2.
        b = (njs - sum_of_squares) / 2.
        c = (nis - sum_of_squares) / 2.
        d = (n * n + sum_of_squares - nis - njs) / 2.

        return (a + d) / (a + b + c + d)


class Recall(IConfusionMatrixMetric):
    """Represents a recall metric."""

    def __init__(self, metric: str = 'RECALL'):
        """Initializes a new instance of the Recall class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the recall."""

        sum_ = self.confusion_matrix.tp + self.confusion_matrix.fn

        if sum_ != 0:
            return self.confusion_matrix.tp / sum_
        else:
            return 0


class SegmentationArea(AreaMetric):
    """Represents a segmentation area metric."""

    def __init__(self, slice_number: int = -1, metric: str = 'SEGAREA'):
        """Initializes a new instance of the SegmentationArea class.

        Args:
            slice_number (int): The slice number to calculate the area.
                Defaults to -1, which will calculate the area on the intermediate slice.
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)
        self.slice_number = slice_number

    def calculate(self):
        """Calculates the segmentation area on a specified slice in mm2."""

        return self._calculate_area(self.segmentation, self.slice_number)


class SegmentationVolume(VolumeMetric):
    """Represents a segmentation volume metric."""

    def __init__(self, metric: str = 'SEGVOL'):
        """Initializes a new instance of the SegmentationVolume class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the segmented volume in mm3."""

        return self._calculate_volume(self.segmentation)


class Sensitivity(IConfusionMatrixMetric):
    """Represents a sensitivity (true positive rate or recall) metric."""

    def __init__(self, metric: str = 'SNSVTY'):
        """Initializes a new instance of the Sensitivity class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the sensitivity (true positive rate)."""

        return self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)


class Specificity(IConfusionMatrixMetric):
    """Represents a specificity metric."""

    def __init__(self, metric: str = 'SPCFTY'):
        """Initializes a new instance of the Specificity class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the specificity."""

        return self.confusion_matrix.tn / (self.confusion_matrix.tn + self.confusion_matrix.fp)


class SurfaceDiceOverlap(IDistanceMetric):
    """Represents a surface Dice coefficient overlap metric.

    See Also:
        - Nikolov et al., 2018 Deep learning to achieve clinically applicable segmentation of head and neck anatomy for
            radiotherapy. `arXiv <https://arxiv.org/abs/1809.04430>`_
        - `Original implementation <https://github.com/deepmind/surface-distance>`_
    """

    def __init__(self, tolerance: float = 1, metric: str = 'SURFDICE'):
        """Initializes a new instance of the SurfaceDiceOverlap class.

        Args:
            tolerance (float): The tolerance of the surface distance in mm.
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)
        self.tolerance = tolerance

    def calculate(self):
        """Calculates the surface Dice coefficient overlap."""

        overlap_gt = np.sum(self.distances.surfel_areas_gt[self.distances.distances_gt_to_pred <= self.tolerance])
        overlap_pred = np.sum(self.distances.surfel_areas_pred[self.distances.distances_pred_to_gt <= self.tolerance])
        surface_dice = (overlap_gt + overlap_pred) / \
                       (np.sum(self.distances.surfel_areas_gt) + np.sum(self.distances.surfel_areas_pred))
        return float(surface_dice)


class SurfaceOverlap(IDistanceMetric):
    """Represents a surface overlap metric.

    Computes the overlap of the ground truth surface with the predicted surface and vice versa allowing a
    specified tolerance (maximum surface-to-surface distance that is regarded as overlapping).
    The overlapping fraction is computed by correctly taking the area of each surface element into account.

    See Also:
        - Nikolov et al., 2018 Deep learning to achieve clinically applicable segmentation of head and neck anatomy for
            radiotherapy. `arXiv <https://arxiv.org/abs/1809.04430>`_
        - `Original implementation <https://github.com/deepmind/surface-distance>`_
    """

    def __init__(self, tolerance: float = 1.0, prediction_to_label: bool = True, metric: str = 'SURFOVLP'):
        """Initializes a new instance of the SurfaceOverlap class.

        Args:
            tolerance (float): The tolerance of the surface distance in mm.
            prediction_to_label (bool): Computes the prediction to labels if `True`, otherwise the label to prediction.
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)
        self.tolerance = tolerance
        self.prediction_to_label = prediction_to_label

    def calculate(self):
        """Calculates the surface overlap."""

        if self.prediction_to_label:
            return float(np.sum(self.distances.surfel_areas_pred[self.distances.distances_pred_to_gt <= self.tolerance])
                         / np.sum(self.distances.surfel_areas_pred))
        else:
            return float(np.sum(self.distances.surfel_areas_gt[self.distances.distances_gt_to_pred <= self.tolerance])
                         / np.sum(self.distances.surfel_areas_gt))


class TrueNegative(IConfusionMatrixMetric):
    """Represents a true negative metric."""

    def __init__(self, metric: str = 'TN'):
        """Initializes a new instance of the TrueNegative class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the true negatives."""

        return self.confusion_matrix.tn


class TruePositive(IConfusionMatrixMetric):
    """Represents a true positive metric."""

    def __init__(self, metric: str = 'TP'):
        """Initializes a new instance of the TruePositive class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the true positives."""

        return self.confusion_matrix.tp


class VariationOfInformation(IConfusionMatrixMetric):
    """Represents a variation of information metric."""

    def __init__(self, metric: str = 'VARINFO'):
        """Initializes a new instance of the VariationOfInformation class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the variation of information."""

        tp = self.confusion_matrix.tp
        tn = self.confusion_matrix.tn
        fp = self.confusion_matrix.fp
        fn = self.confusion_matrix.fn
        n = self.confusion_matrix.n

        fn_tp = fn + tp
        fp_tp = fp + tp

        if fn_tp == 0 or fn_tp / n == 1 or fp_tp == 0 or fp_tp / n == 1:
            warnings.warn('Unable to compute variation of information due to log2 of 0, returning -inf',
                          NotComputableMetricWarning)
            return float('-inf')

        h1 = -((fn_tp / n) * math.log2(fn_tp / n) + (1 - fn_tp / n) * math.log2(1 - fn_tp / n))
        h2 = -((fp_tp / n) * math.log2(fp_tp / n) + (1 - fp_tp / n) * math.log2(1 - fp_tp / n))

        p00 = 1 if tn == 0 else (tn / n)
        p01 = 1 if fn == 0 else (fn / n)
        p10 = 1 if fp == 0 else (fp / n)
        p11 = 1 if tp == 0 else (tp / n)

        h12 = -((tn / n) * math.log2(p00) +
                (fn / n) * math.log2(p01) +
                (fp / n) * math.log2(p10) +
                (tp / n) * math.log2(p11))

        mi = h1 + h2 - h12

        vi = h1 + h2 - 2 * mi
        return vi


class VolumeSimilarity(IConfusionMatrixMetric):
    """Represents a volume similarity metric."""

    def __init__(self, metric: str = 'VOLSMTY'):
        """Initializes a new instance of the VolumeSimilarity class.

        Args:
            metric (str): The identification string of the metric.
        """
        super().__init__(metric)

    def calculate(self):
        """Calculates the volume similarity."""

        tp = self.confusion_matrix.tp
        fp = self.confusion_matrix.fp
        fn = self.confusion_matrix.fn

        return 1 - abs(fn - fp) / (2 * tp + fn + fp)
