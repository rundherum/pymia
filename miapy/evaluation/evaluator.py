"""Contains evaluation function"""
import csv
import os
from abc import ABCMeta, abstractmethod
from typing import Union
import SimpleITK as sitk
import numpy as np
from miapy.evaluation.metric import IMetric, IConfusionMatrixMetric, ConfusionMatrix


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

        for evaluation in data:
            self.write_line(evaluation)

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
        :type data: list of list, e.g. [["PATIENT1", "BACKGROUND", 0.90], ["PATIENT1", "TUMOR", "0.62"]]
        """

        # format all floats with given precision to str
        out = []
        for line in data:
            out += [val if isinstance(val, str) else "{0:.{1}f}".format(val, self.precision) for val in line]
            out += ["\n"]

        # get length for output alignment
        length = max(len(max(out, key=len)), len(max(self.header, key=len))) + 2

        print("".join("{0:>{1}}".format(val, length) for val in self.header),
              "\n",
              "".join("{0:>{1}}".format(val, length) for val in out),
              sep='', end='')

    def write_header(self, header: list):
        """
        Writes the evaluation header.

        :param header: The evaluation header.
        :type header: list of str
        """

        self.header = header


class Evaluator:
    """
    Represents a metric evaluator.

    Example usage:

    >>> evaluator = Evaluator(ConsoleEvaluatorWriter(5))
    >>> evaluator.add_writer(CSVEvaluatorWriter("/some/path/to/results.csv"))
    >>> evaluator.add_label(0, "Background")
    >>> evaluator.add_label(1, "Nerve")
    >>> evaluator.add_metric(DiceCoefficient())
    >>> evaluator.add_metric(VolumeSimilarity())
    >>> evaluator.evaluate(prediction, ground_truth, "Patient1")
    The console output would be:
              ID       LABEL        DICE     VOLSMTY
        Patient1  Background     0.99955     0.99976
        Patient1       Nerve     0.70692     0.84278
    The results.csv would contain:
    ID;LABEL;DICE;VOLSMTY
    Patient1;Background;0.999548418549;0.999757743496
    Patient1;Nerve;0.70692469107;0.842776093884
    """

    def __init__(self, writer: IEvaluatorWriter):
        """
        Initializes a new instance of the Evaluator class.

        :param writer: An evaluator writer.
        :type writer: IEvaluatorWriter
        """

        self.metrics = []  # list of IMetrics
        self.writers = [writer]  # list of IEvaluatorWriters
        self.labels = {}  # dictionary of label: label_str
        self.is_header_written = False

    def add_label(self, label: int, description: str):
        """
        Adds a label with its description to the evaluation.

        :param label: The label.
        :type label: int
        :param description: The label's description.
        :type description: str
        """

        self.labels[label] = description

    def add_metric(self, metric: IMetric):
        """
        Adds a metric to the evaluation.

        :param metric: The metric.
        :type metric: IMetric
        """

        self.metrics.append(metric)
        self.is_header_written = False  # header changed due to new metric

    def add_writer(self, writer: IEvaluatorWriter):
        """
        Adds a writer to the evaluation.

        :param writer: The writer.
        :type writer: IEvaluatorWriter
        """

        self.writers.append(writer)
        self.is_header_written = False  # re-write header

    def evaluate(self, image: Union[sitk.Image, np.ndarray], ground_truth: Union[sitk.Image, np.ndarray],
                 evaluation_id: str):
        """
        Evaluates the metrics on the provided image and ground truth image.

        :param image: The image.
        :type image: sitk.Image
        :param ground_truth: The ground truth image.
        :type ground_truth: sitk.Image
        :param evaluation_id: The identification.
        :type evaluation_id: str
        """

        if not self.is_header_written:
            self.write_header()

        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
        if isinstance(ground_truth, sitk.Image):
            ground_truth = sitk.GetArrayFromImage(ground_truth)
        image_arr = image.flatten()
        ground_truth_arr = ground_truth.flatten()

        results = []  # clear results

        for label, label_str in self.labels.items():
            label_results = [evaluation_id, label_str]

            # we set all values equal to label = 1 (positive) and all other = 0 (negative)
            predictions = image_arr.copy()
            labels = ground_truth_arr.copy()

            if label != 0:
                predictions[predictions != label] = 0
                predictions[predictions == label] = 1
                labels[labels != label] = 0
                labels[labels == label] = 1
            else:
                max_value = max(predictions.max(), labels.max()) + 1
                predictions[predictions == label] = max_value
                predictions[predictions != max_value] = 0
                predictions[predictions == max_value] = 1
                labels[labels == label] = max_value
                labels[labels != max_value] = 0
                labels[labels == max_value] = 1

            # calculate the confusion matrix (used for most metrics)
            confusion_matrix = ConfusionMatrix(predictions, labels)

            # calculate the metrics
            for param_index, metric in enumerate(self.metrics):
                if isinstance(metric, IConfusionMatrixMetric):
                    metric.confusion_matrix = confusion_matrix

                # todo add other metric instances

                label_results.append(metric.calculate())

            results.append(label_results)

        # write the results
        for writer in self.writers:
            writer.write(results)

    def write_header(self):
        """
        Writes the header. 
        """

        header = ["ID", "LABEL"]

        for metric in self.metrics:
            header.append(str(metric))

        for writer in self.writers:
            writer.write_header(header)

        self.is_header_written = True
