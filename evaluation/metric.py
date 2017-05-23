import math
from abc import ABCMeta, abstractmethod


class IMetricParams(metaclass=ABCMeta):
    """
    
    """


class ConfusionMatrix:
    """
    Represents a confusion matrix (or error matrix).
    """

    def __init__(self):
        """
        Initializes a new instance of the ConfusionMatrix class.
        """
        self.condition_positive = 0
        self.condition_negative = 0
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        # self.


class IMetric(metaclass=ABCMeta):
    """
    Represents an evaluation metric.
    """

    def __init__(self):
        self.metric = "IMetric"

    @abstractmethod
    def calculate(self):
        """
        Calculates the metric.
        """
        raise NotImplementedError

    def __str__(self):
        """
        Gets a nicely printable string representation.
        :return: String representation.
        """
        return '{self.metric}' \
            .format(self=self)


class IConfusionMatrixMetric(IMetric):
    """
    Represents an evaluation metric based on the confusion matrix.
    """

    def __init__(self):
        """
        Initializes a new instance of the IConfusionMatrixMetric class.
        """
        super().__init__()
        self.metric = "IConfusionMatrixMetric"
        self.confusion_matrix = ConfusionMatrix()

    @abstractmethod
    def calculate(self):
        """
        Calculates the metric.
        """
        raise NotImplementedError


class Accuracy(IConfusionMatrixMetric):
    """
    Represents an accuracy metric.
    """

    def __init__(self):
        """
        Initializes a new instance of the Accuracy class.
        """
        super().__init__()
        self.metric = "ACURCY"

    def calculate(self):
        """
        Calculates the accuracy.
        """

        sum = self.confusion_matrix.true_positive + self.confusion_matrix.true_negative + \
              self.confusion_matrix.false_positive + self.confusion_matrix.false_negative

        if sum != 0:
            return (self.confusion_matrix.true_positive + self.confusion_matrix.true_negative) / sum
        else:
            return 0


class AreaUnderCurve(IConfusionMatrixMetric):
    """
    Represents an area under the curve metric.
    """

    def __init__(self):
        """
        Initializes a new instance of the AreaUnderCurve class.
        """
        super().__init__()
        self.metric = "AUC"

    def calculate(self):
        """
        Calculates the area under the curve.
        """
        specificity = self.confusion_matrix.true_negative / \
                      (self.confusion_matrix.true_negative + self.confusion_matrix.false_positive)

        false_positive_rate =  1 - specificity

        true_positive_rate = self.confusion_matrix.true_positive / \
                      (self.confusion_matrix.true_positive + self.confusion_matrix.false_negative)

        return (true_positive_rate - false_positive_rate + 1) / 2


class AverageDistance(IMetric):

    def __init__(self):
        """
        Initializes a new instance of the AverageDistance class.
        """
        super().__init__()
        self.metric = "AVGDIST"

    def calculate(self):
        """
        Calculates the metric.
        """
        raise NotImplementedError


class CohenKappaMetric(IConfusionMatrixMetric):
    """
    Represents a Cohen's kappa coefficient metric.
    """

    def __init__(self):
        """
        Initializes a new instance of the CohenKappaMetric class.
        """
        super().__init__()
        self.metric = "KAPPA"

    def calculate(self):
        """
        Calculates the Cohen's kappa coefficient.
        """
        agreement = self.confusion_matrix.true_positive + self.confusion_matrix.true_negative
        chance0 = (self.confusion_matrix.true_negative + self.confusion_matrix.false_negative) * \
                  (self.confusion_matrix.true_negative + self.confusion_matrix.false_positive)
        chance1 = (self.confusion_matrix.false_positive + self.confusion_matrix.true_positive) * \
                  (self.confusion_matrix.false_negative + self.confusion_matrix.true_positive)
        sum = self.confusion_matrix.true_positive + self.confusion_matrix.true_negative + \
              self.confusion_matrix.false_positive + self.confusion_matrix.false_negative
        chance = (chance0 + chance1) / sum

        return (agreement - chance1) / (sum - chance)


class DiceCoefficient(IConfusionMatrixMetric):
    """
    Represents a Dice coefficient metric.
    """

    def __init__(self):
        """
        Initializes a new instance of the DiceCoefficient class.
        """
        super().__init__()
        self.metric = "DICE"

    def calculate(self):
        """
        Calculates the Dice coefficient.
        """
        return 2 * self.confusion_matrix.true_positive / \
               (2 * self.confusion_matrix.true_positive +
                self.confusion_matrix.false_positive + self.confusion_matrix.false_negative)


class Fallout(IConfusionMatrixMetric):
    """
    Represents a fallout (false positive rate) metric.
    """

    def __init__(self):
        """
        Initializes a new instance of the Fallout class.
        """
        super().__init__()
        self.metric = "FALLOUT"

    def calculate(self):
        """
        Calculates the fallout (false positive rate).
        """

        specificity = self.confusion_matrix.true_negative / \
                      (self.confusion_matrix.true_negative + self.confusion_matrix.false_positive)
        return 1 - specificity


class FMeasure(IConfusionMatrixMetric):
    """
    Represents a F-measure metric.
    """

    def __init__(self):
        """
        Initializes a new instance of the FMeasure class.
        """
        super().__init__()
        self.metric = "FMEASR"

    def calculate(self):
        """
        Calculates the Dice coefficient.
        """

        raise NotImplementedError


class GlobalConsistencyError(IConfusionMatrixMetric):
    """
    Represents a global consistency error metric.
    Implementation based on Martin 2001.
    """

    def __init__(self):
        """
        Initializes a new instance of the GlobalConsistencyError class.
        """
        super().__init__()
        self.metric = "GCOERR"

    def calculate(self):
        """
        Calculates the global consistency error.
        """
        tp = self.confusion_matrix.true_positive
        tn = self.confusion_matrix.true_negative
        fp = self.confusion_matrix.false_positive
        fn = self.confusion_matrix.false_negative

        n = tp + tn + fp + fn
        e1 = (fn * (fn + 2 * tp) / (tp + fn) + fp * (fp + 2 * tn) / (tn + fp)) / n
        e2 = (fp * (fp + 2 * tp) / (tp + fp) + fn * (fn + 2 * tn) / (tn + fn)) / n

        return min(e1, e2)


class HausdorffDistance(IMetric):

    def __init__(self):
        """
        Initializes a new instance of the HausdorffDistance class.
        """
        super().__init__()
        self.metric = "HDRFDST"

    def calculate(self):
        """
        Calculates the metric.
        """
        raise NotImplementedError


class InterclassCorrelation(IMetric):

    def __init__(self):
        """
        Initializes a new instance of the InterclassCorrelation class.
        """
        super().__init__()
        self.metric = "ICCORR"

    def calculate(self):
        """
        Calculates the metric.
        """
        raise NotImplementedError


class JaccardCoefficient(IConfusionMatrixMetric):
    """
    Represents a Jaccard coefficient metric.
    """

    def __init__(self):
        """
        Initializes a new instance of the JaccardCoefficient class.
        """
        super().__init__()
        self.metric = "JACRD"

    def calculate(self):
        """
        Calculates the Jaccard coefficient.
        """
        tp = self.confusion_matrix.true_positive
        fp = self.confusion_matrix.false_positive
        fn = self.confusion_matrix.false_negative

        return tp / (tp + fp + fn)


class MahalanobisDistance(IMetric):

    def __init__(self):
        """
        Initializes a new instance of the MahalanobisDistance class.
        """
        super().__init__()
        self.metric = "MAHLNBS"

    def calculate(self):
        """
        Calculates the metric.
        """
        raise NotImplementedError


class MutualInformation(IMetric):

    def __init__(self):
        """
        Initializes a new instance of the MutualInformation class.
        """
        super().__init__()
        self.metric = "MUTINF"

    def calculate(self):
        """
        Calculates the metric.
        """
        raise NotImplementedError


class Precision(IConfusionMatrixMetric):
    """
    Represents a precision metric.
    """

    def __init__(self):
        """
        Initializes a new instance of the Precision class.
        """
        super().__init__()
        self.metric = "PRCISON"

    def calculate(self):
        """
        Calculates the precision.
        """

        sum = self.confusion_matrix.true_positive + self.confusion_matrix.false_positive

        if sum != 0:
            return self.confusion_matrix.true_positive / sum
        else:
            return 0


class ProbabilisticDistance(IMetric):

    def __init__(self):
        """
        Initializes a new instance of the ProbabilisticDistance class.
        """
        super().__init__()
        self.metric = "PROBDST"

    def calculate(self):
        """
        Calculates the metric.
        """
        raise NotImplementedError


class RandIndex(IMetric):

    def __init__(self):
        """
        Initializes a new instance of the RandIndex class.
        """
        super().__init__()
        self.metric = "RNDIND"

    def calculate(self):
        """
        Calculates the metric.
        """
        raise NotImplementedError


class Recall(IConfusionMatrixMetric):
    """
    Represents a recall metric.
    """

    def __init__(self):
        """
        Initializes a new instance of the Recall class.
        """
        super().__init__()
        self.metric = "RECALL"

    def calculate(self):
        """
        Calculates the recall.
        """

        sum = self.confusion_matrix.true_positive + self.confusion_matrix.false_negative

        if sum != 0:
            return self.confusion_matrix.true_positive / sum
        else:
            return 0


class Sensitivity(IConfusionMatrixMetric):
    """
    Represents a sensitivity (true positive rate) metric.
    """

    def __init__(self):
        """
        Initializes a new instance of the Sensitivity class.
        """
        super().__init__()
        self.metric = "SNSVTY"

    def calculate(self):
        """
        Calculates the sensitivity (true positive rate).
        """
        return self.confusion_matrix.true_positive / \
                      (self.confusion_matrix.true_positive + self.confusion_matrix.false_negative)


class Specificity(IConfusionMatrixMetric):
    """
    Represents a specificity metric.
    """

    def __init__(self):
        """
        Initializes a new instance of the Specificity class.
        """
        super().__init__()
        self.metric = "SPCFTY"

    def calculate(self):
        """
        Calculates the specificity.
        """
        return self.confusion_matrix.true_negative / \
                      (self.confusion_matrix.true_negative + self.confusion_matrix.false_positive)


class VariationOfInformation(IMetric):

    def __init__(self):
        """
        Initializes a new instance of the VariationOfInformation class.
        """
        super().__init__()
        self.metric = "VARINFO"

    def calculate(self):
        """
        Calculates the metric.
        """
        raise NotImplementedError


class VolumeSimilarity(IConfusionMatrixMetric):
    """
    Represents a volume similarity metric.
    """

    def __init__(self):
        """
        Initializes a new instance of the VolumeSimilarity class.
        """
        super().__init__()
        self.metric = "VOLSMTY"

    def calculate(self):
        """
        Calculates the volume similarity.
        """
        tp = self.confusion_matrix.true_positive
        fp = self.confusion_matrix.false_positive
        fn = self.confusion_matrix.false_negative

        return 1 - abs(fn - fp) / (2 * tp + fn + fp)
