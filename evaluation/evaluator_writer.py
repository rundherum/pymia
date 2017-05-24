import csv
import os
from abc import ABCMeta, abstractmethod


class IEvaluatorWriter(metaclass=ABCMeta):
    """
    Represents an evaluator writer interface, which enables to write evaluation results.
    """

    @abstractmethod
    def write(self, data: list):
        """
        Writes the evaluation results.
        :param data: The evaluation data.
        :type data: list
        """
        raise NotImplementedError

    @abstractmethod
    def write_header(self, header: list):
        """
        Writes the evaluation header.
        :param header: The evaluation header.
        :type header: list
        """
        raise NotImplementedError


class CSVEvaluatorWriter(IEvaluatorWriter):
    """
    Represents a CSV evaluator writer.
    """

    def __init__(self, path: str):
        """
        Initializes a new instance of the CSVEvaluatorWriter class.
        :param path: The file path.
        :type path: str
        """
        super().__init__()

        self.path = path

        # check file extension
        if not self.path.lower().endswith(".csv"):
            self.path = os.path.join(self.path, ".csv")

        open(self.path, 'w', newline='')  # creates (and overrides an existing) file

    def write(self, data: list):
        """
        Writes the evaluation results.
        :param data: The evaluation data.
        :type data: list
        """

        self.write_line(data)

    def write_header(self, header: list):
        """
        Writes the evaluation header.
        :param header: The evaluation header.
        :type header: list
        """

        self.write_line(header)

    def write_line(self, data: list):
        """
        Writes a line.
        :param data: The data.
        :type data: list
        """
        with open(self.path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(data)


class ConsoleEvaluatorWriter(IEvaluatorWriter):
    """
    Represents a console evaluator writer.
    """

    def __init__(self, precision: int=3):
        """
        Initializes a new instance of the ConsoleEvaluatorWriter class.
        :param precision: The float precision.
        :type precision: int
        """
        super().__init__()

        self.header = None
        self.precision = precision

    def write(self, data: list):
        """
        Writes the evaluation results.
        :param data: The evaluation data.
        :type data: list of floats
        """

        # format all floats with given precision to str
        data = [val if isinstance(val, str) else "{0:.{1}f}".format(val, self.precision) for val in data]

        # get length for output alignment
        length = max(len(max(data, key=len)), len(max(self.header, key=len))) + 2
        print(length)
        print(",".join("{0:>{1}}".format(val, length) for val in self.header),
              "\n",
              ",".join("{0:>{1}}".format(val, length) for val in data),
              sep='')

    def write_header(self, header: list):
        """
        Writes the evaluation header.
        :param header: The evaluation header.
        :type header: list of str
        """

        self.header = header
