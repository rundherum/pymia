"""The metric module contains a set of evaluation metrics.

The metrics are selected based on the paper of Taha 2016.
Refer to the paper for guidelines how to select appropriate metrics, in-depth description,
and the math.

It is possible to implement your own metrics and use them with the :class:`evaluator.Evaluator`.
Just inherit from :class:`metric.IMetric`, :class:`metric.IConfusionMatrixMetric` or :class:`ISimpleITKImageMetric`
and implement the function :func:`calculate`.
"""
from abc import ABCMeta, abstractmethod
import math

import numpy as np
import SimpleITK as sitk


def get_all_metrics():
    """Gets a list with all metrics.

    Returns:
        list[IMetric]: A list of metrics.
    """
    return get_overlap_metrics() + get_distance_metrics() + get_distance_metrics()


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
           FMeasure(),
           Accuracy(),
           Fallout(),
           TruePositive(),
           FalsePositive(),
           TrueNegative(),
           FalseNegative(),
           LabelVolume(),
           PredictionVolume()]


def _calculate_volume(image: sitk.Image):
    """Calculates the volume of a label image."""

    voxel_volume = np.prod(image.GetSpacing())
    number_of_voxels = sitk.GetArrayFromImage(image).sum()

    return number_of_voxels * voxel_volume


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


class IMetric(metaclass=ABCMeta):
    """Represents an evaluation metric."""

    def __init__(self):
        self.metric = "IMetric"

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def calculate(self):
        """Calculates the metric."""

        raise NotImplementedError


class Accuracy(IConfusionMatrixMetric):
    """Represents an accuracy metric."""

    def __init__(self):
        """Initializes a new instance of the Accuracy class."""
        super().__init__()
        self.metric = "ACURCY"

    def calculate(self):
        """Calculates the accuracy."""

        sum = self.confusion_matrix.tp + self.confusion_matrix.tn + self.confusion_matrix.fp + self.confusion_matrix.fn

        if sum != 0:
            return (self.confusion_matrix.tp + self.confusion_matrix.tn) / sum
        else:
            return 0


class AdjustedRandIndex(IConfusionMatrixMetric):
    """Represents an adjusted rand index metric."""

    def __init__(self):
        """Initializes a new instance of the AdjustedRandIndex class."""
        super().__init__()
        self.metric = "ADJRIND"

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

    def __init__(self):
        """Initializes a new instance of the AreaUnderCurve class."""
        super().__init__()
        self.metric = "AUC"

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

    def __init__(self):
        """Initializes a new instance of the AverageDistance class."""
        super().__init__()
        self.metric = "AVGDIST"

    def calculate(self):
        """Calculates the average (Hausdorff) distance."""

        distance_filter = sitk.HausdorffDistanceImageFilter()
        distance_filter.Execute(self.ground_truth, self.segmentation)
        return distance_filter.GetAverageHausdorffDistance()


class CohenKappaMetric(IConfusionMatrixMetric):
    """Represents a Cohen's kappa coefficient metric."""

    def __init__(self):
        """Initializes a new instance of the CohenKappaMetric class."""
        super().__init__()
        self.metric = "KAPPA"

    def calculate(self):
        """Calculates the Cohen's kappa coefficient."""

        tp = self.confusion_matrix.tp
        tn = self.confusion_matrix.tn
        fp = self.confusion_matrix.fp
        fn = self.confusion_matrix.fn

        agreement = tp + tn
        chance0 = (tn + fn) * (tn + fp)
        chance1 = (fp + tp) * (fn + tp)
        sum = tn + fn + fp + tp
        chance = (chance0 + chance1) / sum

        return (agreement - chance) / (sum - chance)


class DiceCoefficient(IConfusionMatrixMetric):
    """Represents a Dice coefficient metric."""

    def __init__(self):
        """Initializes a new instance of the DiceCoefficient class."""
        super().__init__()
        self.metric = "DICE"

    def calculate(self):
        """Calculates the Dice coefficient."""

        return 2 * self.confusion_matrix.tp / \
               (2 * self.confusion_matrix.tp +
                self.confusion_matrix.fp + self.confusion_matrix.fn)


class FalseNegative(IConfusionMatrixMetric):
    """Represents a false negative metric."""

    def __init__(self):
        """Initializes a new instance of the FalseNegative class."""
        super().__init__()
        self.metric = "FN"

    def calculate(self):
        """Calculates the false negatives."""

        return self.confusion_matrix.fn


class FalsePositive(IConfusionMatrixMetric):
    """Represents a false positive metric."""

    def __init__(self):
        """Initializes a new instance of the FalsePositive class."""
        super().__init__()
        self.metric = "FP"

    def calculate(self):
        """Calculates the false positives."""

        return self.confusion_matrix.fp


class Fallout(IConfusionMatrixMetric):
    """Represents a fallout (false positive rate) metric."""

    def __init__(self):
        """Initializes a new instance of the Fallout class."""
        super().__init__()
        self.metric = "FALLOUT"

    def calculate(self):
        """Calculates the fallout (false positive rate)."""

        specificity = self.confusion_matrix.tn / (self.confusion_matrix.tn + self.confusion_matrix.fp)
        return 1 - specificity


class FalseNegativeRate(IConfusionMatrixMetric):
    """Represents a false negative rate metric."""

    def __init__(self):
        super().__init__()
        self.metric = 'FNR'

    def calculate(self):
        """Calculates the false negative rate."""

        sensitivity = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        return 1 - sensitivity


class FMeasure(IConfusionMatrixMetric):
    """Represents a F-measure metric."""

    def __init__(self):
        """Initializes a new instance of the FMeasure class."""
        super().__init__()
        self.metric = "FMEASR"

    def calculate(self):
        """Calculates the F1 measure."""

        beta = 1 # or 0.5 or 2 can also calculate F2 or F0.5 measure

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

    def __init__(self):
        """Initializes a new instance of the GlobalConsistencyError class."""
        super().__init__()
        self.metric = "GCOERR"

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


class HausdorffDistance(ISimpleITKImageMetric):
    """Represents a Hausdorff distance metric.

    Calculates the distance between the set of non-zero pixels of two images using the following equation:

    .. math:: H(A,B) = max(h(A,B), h(B,A)),

    where

    .. math:: h(A,B) = \\max_{a \\in A} \\min_{b \\in B} \\lVert a - b \\rVert

    is the directed Hausdorff distance and :math:`A` and :math:`B` are the set of non-zero pixels in the images.
    """

    def __init__(self):
        """Initializes a new instance of the HausdorffDistance class."""
        super().__init__()
        self.metric = "HDRFDST"

    def calculate(self):
        """Calculates the Hausdorff distance."""

        distance_filter = sitk.HausdorffDistanceImageFilter()
        distance_filter.Execute(self.ground_truth, self.segmentation)
        return distance_filter.GetHausdorffDistance()


class InterclassCorrelation(INumpyArrayMetric):
    """Represents a interclass correlation metric."""

    def __init__(self):
        """Initializes a new instance of the InterclassCorrelation class."""
        super().__init__()
        self.metric = "ICCORR"

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

    def __init__(self):
        """Initializes a new instance of the JaccardCoefficient class."""
        super().__init__()
        self.metric = "JACRD"

    def calculate(self):
        """Calculates the Jaccard coefficient."""

        tp = self.confusion_matrix.tp
        fp = self.confusion_matrix.fp
        fn = self.confusion_matrix.fn

        return tp / (tp + fp + fn)


class LabelVolume(ISimpleITKImageMetric):
    """Represents a label (ground truth) volume metric."""

    def __init__(self):
        """Initializes a new instance of the LabelVolume class."""
        super().__init__()
        self.metric = "LBLVOL"

    def calculate(self):
        """Calculates the labeled (ground truth) volume in mm3."""

        return _calculate_volume(self.ground_truth)


class MahalanobisDistance(INumpyArrayMetric):
    """Represents a Mahalanobis distance metric."""

    def __init__(self):
        """Initializes a new instance of the MahalanobisDistance class."""
        super().__init__()
        self.metric = "MAHLNBS"

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

    def __init__(self):
        """Initializes a new instance of the MutualInformation class."""
        super().__init__()
        self.metric = "MUTINF"

    def calculate(self):
        """Calculates the mutual information."""

        tp = self.confusion_matrix.tp
        tn = self.confusion_matrix.tn
        fp = self.confusion_matrix.fp
        fn = self.confusion_matrix.fn
        n = self.confusion_matrix.n

        fn_tp = fn + tp
        fp_tp = fp + tp

        H1 = -((fn_tp / n) * math.log2(fn_tp / n) +
               (1 - fn_tp / n) * math.log2(1 - fn_tp / n))

        H2 = -((fp_tp / n) * math.log2(fp_tp / n) +
               (1 - fp_tp / n) * math.log2(1 - fp_tp / n))

        p00 = 1 if tn == 0 else (tn / n)
        p01 = 1 if fn == 0 else (fn / n)
        p10 = 1 if fp == 0 else (fp / n)
        p11 = 1 if tp == 0 else (tp / n)

        H12 = -((tn / n) * math.log2(p00) +
                (fn / n) * math.log2(p01) +
                (fp / n) * math.log2(p10) +
                (tp / n) * math.log2(p11))

        MI = H1 + H2 - H12
        return MI


class Precision(IConfusionMatrixMetric):
    """Represents a precision metric."""

    def __init__(self):
        """Initializes a new instance of the Precision class."""
        super().__init__()
        self.metric = "PRCISON"

    def calculate(self):
        """Calculates the precision."""

        sum = self.confusion_matrix.tp + self.confusion_matrix.fp

        if sum != 0:
            return self.confusion_matrix.tp / sum
        else:
            return 0


class PredictionVolume(ISimpleITKImageMetric):
    """Represents a prediction (segmentation) volume metric."""

    def __init__(self):
        """Initializes a new instance of the PredictionVolume class."""
        super().__init__()
        self.metric = "PRDVOL"

    def calculate(self):
        """Calculates the predicted (segmented) volume in mm3."""

        return _calculate_volume(self.segmentation)


class ProbabilisticDistance(INumpyArrayMetric):
    """Represents a probabilistic distance metric."""

    def __init__(self):
        """Initializes a new instance of the ProbabilisticDistance class."""
        super().__init__()
        self.metric = "PROBDST"

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

    def __init__(self):
        """Initializes a new instance of the RandIndex class."""
        super().__init__()
        self.metric = "RNDIND"

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

    def __init__(self):
        """Initializes a new instance of the Recall class."""
        super().__init__()
        self.metric = "RECALL"

    def calculate(self):
        """Calculates the recall."""

        sum = self.confusion_matrix.tp + self.confusion_matrix.fn

        if sum != 0:
            return self.confusion_matrix.tp / sum
        else:
            return 0


class Sensitivity(IConfusionMatrixMetric):
    """Represents a sensitivity (true positive rate or recall) metric."""

    def __init__(self):
        """Initializes a new instance of the Sensitivity class."""
        super().__init__()
        self.metric = "SNSVTY"

    def calculate(self):
        """Calculates the sensitivity (true positive rate)."""

        return self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)


class Specificity(IConfusionMatrixMetric):
    """Represents a specificity metric."""

    def __init__(self):
        """Initializes a new instance of the Specificity class."""
        super().__init__()
        self.metric = "SPCFTY"

    def calculate(self):
        """Calculates the specificity."""

        return self.confusion_matrix.tn / (self.confusion_matrix.tn + self.confusion_matrix.fp)


class TrueNegative(IConfusionMatrixMetric):
    """Represents a true negative metric."""

    def __init__(self):
        """Initializes a new instance of the TrueNegative class."""
        super().__init__()
        self.metric = "TN"

    def calculate(self):
        """Calculates the true negatives."""

        return self.confusion_matrix.tn


class TruePositive(IConfusionMatrixMetric):
    """Represents a true positive metric."""

    def __init__(self):
        """Initializes a new instance of the TruePositive class."""
        super().__init__()
        self.metric = "TP"

    def calculate(self):
        """Calculates the true positives."""

        return self.confusion_matrix.tp


class VariationOfInformation(IConfusionMatrixMetric):
    """Represents a variation of information metric."""

    def __init__(self):
        """Initializes a new instance of the VariationOfInformation class."""
        super().__init__()
        self.metric = "VARINFO"

    def calculate(self):
        """Calculates the variation of information."""

        tp = self.confusion_matrix.tp
        tn = self.confusion_matrix.tn
        fp = self.confusion_matrix.fp
        fn = self.confusion_matrix.fn
        n = self.confusion_matrix.n

        fn_tp = fn + tp
        fp_tp = fp + tp

        H1 = -((fn_tp / n) * math.log2(fn_tp / n) +
               (1 - fn_tp / n) * math.log2(1 - fn_tp / n))

        H2 = -((fp_tp / n) * math.log2(fp_tp / n) +
               (1 - fp_tp / n) * math.log2(1 - fp_tp / n))

        p00 = 1 if tn == 0 else (tn / n)
        p01 = 1 if fn == 0 else (fn / n)
        p10 = 1 if fp == 0 else (fp / n)
        p11 = 1 if tp == 0 else (tp / n)

        H12 = -((tn / n) * math.log2(p00) +
                (fn / n) * math.log2(p01) +
                (fp / n) * math.log2(p10) +
                (tp / n) * math.log2(p11))

        MI = H1 + H2 - H12

        VI = H1 + H2 - 2 * MI
        return VI


class VolumeSimilarity(IConfusionMatrixMetric):
    """Represents a volume similarity metric."""

    def __init__(self):
        """Initializes a new instance of the VolumeSimilarity class."""
        super().__init__()
        self.metric = "VOLSMTY"

    def calculate(self):
        """Calculates the volume similarity."""

        tp = self.confusion_matrix.tp
        fp = self.confusion_matrix.fp
        fn = self.confusion_matrix.fn

        return 1 - abs(fn - fp) / (2 * tp + fn + fp)
