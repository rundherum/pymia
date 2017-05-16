from abc import ABCMeta, abstractmethod


class IEvaluatorWriter(metaclass=ABCMeta):
    """
    Represents an evaluator writer interface, which enables to write or ensemble evaluation results.
    """

    @abstractmethod
    def write(self):
        """
        Writes the evaluation results.
        """
        raise NotImplementedError


class CSVEvaluatorWriter(IEvaluatorWriter):
    """
    Represents a CSV evaluator writer.
    """

    def write(self):
        pass
