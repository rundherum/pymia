import abc
import csv
import logging
import os

import numpy as np


class WriterBase(abc.ABC):
    """Represents a writer interface to write evaluation results."""

    @abc.abstractmethod
    def write(self, data: list):
        """Writes the evaluation results.

        Args:
            data (list): The evaluation data.
        """
        raise NotImplementedError


class CSVWriter(WriterBase):
    """Represents a CSV (comma-separated values) evaluator writer."""

    def __init__(self, path: str, delimiter: str=';'):
        """Initializes a new instance of the CSVEvaluatorWriter class.

        Args:
            path (str): The CSV file path.
            delimiter (str): The CSV column delimiter.
        """
        super().__init__()
        self.path = path
        self.delimiter = delimiter

        # check file extension
        if not self.path.lower().endswith('.csv'):
            self.path = os.path.join(self.path, '.csv')

    def write(self, data: list):
        """Writes the evaluation results.

        Args:
            data (list): The evaluation data.
        """

        # get unique labels
        labels = []

        # get unique metrics
        metrics = []

        # get unique ids
        ids = []

        with open(self.path, 'w', newline='') as file:  # creates (and overrides an existing) file
            writer = csv.writer(file, delimiter=self.delimiter)
            # write header
            writer.writerow(['SUBJECT', 'LABEL'] + metrics)

            for id_ in ids:
                for label in labels:
                    row = [id_, label]
                    for metric in metrics:
                        row += [1., ]
                    writer.writerow(row)


class ConsoleWriter(WriterBase):
    """Represents a console evaluator writer."""

    def __init__(self, precision: int = 3, use_logging: bool = False):
        """Initializes a new instance of the ConsoleEvaluatorWriter class.

        Args:
            precision (int): The float precision.
            use_logging (bool): Indicates whether to use the Python logging module or not.
        """
        super().__init__()

        self.precision = precision
        self.use_logging = use_logging

    def write(self, data: list):
        """Writes the evaluation results.

        Args:
            data (list of list): The evaluation data,
                e.g. [['PATIENT1', 'BACKGROUND', 0.90], ['PATIENT1', 'TUMOR', '0.62']]
        """

        # format all floats with given precision to str
        out_as_string = [self.header]
        for line in data:
            out_as_string.append([val if isinstance(val, str) else "{0:.{1}f}".format(val, self.precision)
                                  for val in line])

        # determine length of each column for output alignment
        lengths = np.array([list(map(len, row)) for row in out_as_string])
        lengths = lengths.max(0)
        lengths += (len(lengths) - 1) * [2] + [0, ]  # append two spaces except for last column

        # format for output alignment
        out = [['{0:<{1}}'.format(val, lengths[idx]) for idx, val in enumerate(line)] for line in out_as_string]

        if self.use_logging:
            logging.info('\n'.join(''.join(line) for line in out))
        else:
            print('\n'.join(''.join(line) for line in out), sep='', end='\n')


class StatisticsWriter(WriterBase, abc.ABC):

    def __init__(self, functions: dict={'MEAN': np.mean, 'STD': np.std}):
        self.functions = functions

    def calculate(self):
        pass
