from abc import ABCMeta, abstractmethod


class IMetricParams(metaclass=ABCMeta):
    """
    
    """



class IMetric(metaclass=ABCMeta):
    """
    Represents an evaluation metric.
    """

    @abstractmethod
    def calculate(self, params: IMetricParams=None):
        """
        Calculates the metric.
        """
        raise NotImplementedError




class DiceCoefficient(IMetric):
    """
    Represents a CSV evaluator writer.
    """

    def calculate(self):
        """
        Calculates the metric.
        """
        raise NotImplementedError

