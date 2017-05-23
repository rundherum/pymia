import csv
from abc import ABCMeta, abstractmethod


class IEvaluatorWriter(metaclass=ABCMeta):
    """
    Represents an evaluator writer interface, which enables to write or ensemble evaluation results.
    """

    def __init__(self):
        self.precision = 2
        self.path = None

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
        with open(self.path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["IMAGE", "DICE"])


class ConsoleEvaluatorWriter(IEvaluatorWriter):

    def write(self):
        pass