"""The writer module provides classes to write evaluation results.

All writers inherit the :class:`pymia.evaluation.writer.Writer`, which writes the results when
calling :meth:`pymia.evaluation.writer.Writer.write`. Currently, pymia has CSV file
(:class:`pymia.evaluation.writer.CSVWriter` and :class:`pymia.evaluation.writer.CSVStatisticsWriter`) and
console writers (:class:`pymia.evaluation.writer.ConsoleWriter` and :class:`pymia.evaluation.writer.ConsoleStatisticsWriter`).
"""
import abc
import csv
import logging
import os
import typing

import numpy as np

import pymia.evaluation.evaluator as evaluator


class Writer(abc.ABC):
    """Represents an evaluation results writer base class."""

    @abc.abstractmethod
    def write(self, results: typing.List[evaluator.Result], **kwargs):
        """Writes the evaluation results.

        Args:
            results (list of evaluator.Result): The evaluation results.
        """
        raise NotImplementedError


class ConsoleWriterHelper:

    def __init__(self, use_logging: bool = False):
        """Represents a console writer helper.

        Args:
            use_logging (bool): Indicates whether to use the Python logging module or not.
        """
        self.use_logging = use_logging

    def format_and_write(self, lines: list):
        """Formats and writes the results.

        Args:
            lines (list of lists): The lines to write. Each line is a list of columns.
        """

        # determine length of each column for output alignment
        lengths = np.array([list(map(len, row)) for row in lines])
        lengths = np.max(lengths, axis=0)
        lengths += (len(lengths) - 1) * [2] + [0, ]  # append two spaces except for last column

        # format with spaces for output alignment
        out = [[f'{val:<{lengths[idx]}}' for idx, val in enumerate(line)] for line in lines]

        if self.use_logging:
            logging.info('\n'.join(''.join(line) for line in out))
        else:
            print('\n'.join(''.join(line) for line in out), sep='', end='\n')


class StatisticsAggregator:

    def __init__(self, functions: dict = None):
        """Represents a statistics evaluation results aggregator.

        Args:
            functions (dict): The numpy function handles to calculate the statistics.
        """
        super().__init__()

        if functions is None:
            functions = {'MEAN': np.mean, 'STD': np.std}
        self.functions = functions

    def calculate(self, results: typing.List[evaluator.Result]) -> typing.List[evaluator.Result]:
        """Calculates aggregated results (e.g., mean and standard deviation of a metric over all cases).

        Args:
            results (typing.List[evaluator.Result]): The results to aggregate.

        Returns:
            typing.List[evaluator.Result]: The aggregated results.
        """

        # get unique labels and metrics
        labels = sorted({result.label for result in results})
        metrics = sorted({result.metric for result in results})

        aggregated_results = []

        for label in labels:
            for metric in metrics:
                # search for results
                values = [r.value for r in results if r.label == label and r.metric == metric]

                for fn_id, fn in self.functions.items():
                    aggregated_results.append(evaluator.Result(
                        fn_id,
                        label,
                        metric,
                        float(fn(values))
                    ))

        return aggregated_results


class CSVWriter(Writer):

    def __init__(self, path: str, delimiter: str = ';'):
        """Represents a CSV file evaluation results writer.

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

    def write(self, results: typing.List[evaluator.Result], **kwargs):
        """Writes the evaluation results to a CSV file.

        Args:
            results (typing.List[evaluator.Result]): The evaluation results.
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
                        row.append(value if value is not None else 'n/a')
                    writer.writerow(row)


class ConsoleWriter(Writer):

    def __init__(self, precision: int = 3, use_logging: bool = False):
        """Represents a console evaluation results writer.

        Args:
            precision (int): The decimal precision.
            use_logging (bool): Indicates whether to use the Python logging module or not.
        """
        super().__init__()

        self.write_helper = ConsoleWriterHelper(use_logging)
        self.precision = precision

    def write(self, results: typing.List[evaluator.Result], **kwargs):
        """Writes the evaluation results.

        Args:
            results (typing.List[evaluator.Result]): The evaluation results.
        """

        # get unique ids, labels, and metrics
        ids = sorted({result.id_ for result in results})
        labels = sorted({result.label for result in results})
        metrics = sorted({result.metric for result in results})

        # header
        lines = [['SUBJECT', 'LABEL'] + metrics]

        for id_ in ids:
            for label in labels:
                row = [id_, label]
                for metric in metrics:
                    # search for result
                    value = next(
                        (r.value for r in results if r.id_ == id_ and r.label == label and r.metric == metric),
                        None)
                    if value is not None:
                        # format float with given precision
                        row.append(value if isinstance(value, str) else f'{value:.{self.precision}f}')
                    else:
                        row.append('n/a')
                lines.append(row)

        self.write_helper.format_and_write(lines)


class CSVStatisticsWriter(Writer):

    def __init__(self, path: str, delimiter: str = ';', functions: dict = None):
        """Represents a CSV file evaluation results statistics writer.

        Args:
            path (str): The CSV file path.
            delimiter (str): The CSV column delimiter.
            functions (dict): The functions to calculate the statistics.
        """
        super().__init__()
        self.aggregator = StatisticsAggregator(functions)
        self.path = path
        self.delimiter = delimiter

        # check file extension
        if not self.path.lower().endswith('.csv'):
            self.path = os.path.join(self.path, '.csv')

    def write(self, results: typing.List[evaluator.Result], **kwargs):
        """Writes the evaluation statistic results (e.g., mean and standard deviation of a metric over all cases).

        Args:
            results (typing.List[evaluator.Result]): The evaluation results.
        """
        aggregated_results = self.aggregator.calculate(results)

        with open(self.path, 'w', newline='') as file:  # creates (and overrides an existing) file
            writer = csv.writer(file, delimiter=self.delimiter)

            # write header
            writer.writerow(['LABEL', 'METRIC', 'STATISTIC', 'VALUE'])

            for result in aggregated_results:
                writer.writerow([result.label, result.metric, result.id_, result.value])


class ConsoleStatisticsWriter(Writer):

    def __init__(self, precision: int = 3, use_logging: bool = False,
                 functions: dict = None):
        """Represents a console evaluation results statistics writer.

        Args:
            precision (int): The float precision.
            use_logging (bool): Indicates whether to use the Python logging module or not.
            functions (dict): The function handles to calculate the statistics.
        """
        super().__init__()
        self.aggregator = StatisticsAggregator(functions)
        self.write_helper = ConsoleWriterHelper(use_logging)
        self.precision = precision

    def write(self, results: typing.List[evaluator.Result], **kwargs):
        """Writes the evaluation statistic results (e.g., mean and standard deviation of a metric over all cases).

        Args:
            results (typing.List[evaluator.Result]): The evaluation results.
        """
        aggregated_results = self.aggregator.calculate(results)

        # header
        lines = [['LABEL', 'METRIC', 'STATISTIC', 'VALUE']]

        for result in aggregated_results:
            lines.append([result.label, result.metric, result.id_,
                          result.value if isinstance(result.value, str) else f'{result.value:.{self.precision}f}'])

        self.write_helper.format_and_write(lines)
