import abc
import csv
import logging
import os
import typing

import numpy as np

import pymia.evaluation.evaluator as eval_


class WriterBase(abc.ABC):
    """Represents an evaluation results writer interface."""

    @abc.abstractmethod
    def write(self, results: typing.List[eval_.Result]):
        """Writes the evaluation results.

        Args:
            results (list of eval_.Result): The evaluation results.
        """
        raise NotImplementedError


class CSVWriter(WriterBase):
    """Represents a CSV file evaluation results writer."""

    def __init__(self, path: str, delimiter: str = ';'):
        """Initializes a new instance of the CSVWriter class.

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

    def write(self, results: typing.List[eval_.Result]):
        """Writes the evaluation results to a CSV file.

        Args:
            results (list of eval_.Result): The evaluation results.
        """

        # get unique ids, labels, and metrics
        ids = sorted({result.id_ for result in results})
        labels = sorted({result.label for result in results})
        metrics = sorted({result.metric for result in results})

        with open(self.path, 'w', newline='') as file:  # creates (and overrides an existing) file
            writer = csv.writer(file, delimiter=self.delimiter)

            # write header
            writer.writerow(['SUBJECT', 'LABEL'] + metrics)

            for id_ in ids:
                for label in labels:
                    row = [id_, label]
                    for metric in metrics:
                        # search for result
                        value = next(
                            (r.value for r in results if r.id_ == id_ and r.label == label and r.metric == metric),
                            None)
                        row.append(value if value else 'n/a')
                    writer.writerow(row)


class ConsoleWriter(WriterBase):
    """Represents a console evaluation results writer."""

    def __init__(self, precision: int = 3, use_logging: bool = False):
        """Initializes a new instance of the ConsoleWriter class.

        Args:
            precision (int): The float precision.
            use_logging (bool): Indicates whether to use the Python logging module or not.
        """
        super().__init__()

        self.precision = precision
        self.use_logging = use_logging

    def write(self, results: typing.List[eval_.Result]):
        """Writes the evaluation results.

        Args:
            results (list of eval_.Result): The evaluation results.
        """

        # get unique ids, labels, and metrics
        ids = sorted({result.id_ for result in results})
        labels = sorted({result.label for result in results})
        metrics = sorted({result.metric for result in results})

        # header
        out_as_string = [['SUBJECT', 'LABEL'] + metrics]

        for id_ in ids:
            for label in labels:
                row = [id_, label]
                for metric in metrics:
                    # search for result
                    value = next(
                        (r.value for r in results if r.id_ == id_ and r.label == label and r.metric == metric),
                        None)
                    if value:
                        # format float with given precision
                        row.append(value if isinstance(value, str) else f'{value:.{self.precision}f}')
                    else:
                        row.append('n/a')
                out_as_string.append(row)

        # determine length of each column for output alignment
        lengths = np.array([list(map(len, row)) for row in out_as_string])
        lengths = np.max(lengths, axis=0)
        lengths += (len(lengths) - 1) * [2] + [0, ]  # append two spaces except for last column

        # format with spaces for output alignment
        out = [[f'{val:<{lengths[idx]}}' for idx, val in enumerate(line)] for line in out_as_string]

        if self.use_logging:
            logging.info('\n'.join(''.join(line) for line in out))
        else:
            print('\n'.join(''.join(line) for line in out), sep='', end='\n')


class StatisticsWriter(WriterBase, abc.ABC):

    def __init__(self, functions: dict={'MEAN': np.mean, 'STD': np.std}):
        self.functions = functions

    def calculate(self):
        pass
