"""Contains evaluation function"""
import csv
import os
from abc import ABCMeta, abstractmethod
from typing import Union
import SimpleITK as sitk
import numpy as np
from miapy.evaluation.metric import IMetric, IConfusionMatrixMetric, ConfusionMatrix, ISimpleITKImageMetric, INumpyArrayMetric


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

    def __init__(self, writer: IEvaluatorWriter=None):
        """
        Initializes a new instance of the Evaluator class.

        :param writer: One evaluator writer.
        :type writer: IEvaluatorWriter
        """

        self.metrics = []  # list of IMetrics
        self.writers = [writer] if writer is not None else []  # list of IEvaluatorWriters
        self.labels = {}  # dictionary of label: label_str
        self.is_header_written = False

    def add_label(self, label: Union[tuple, int], description: str):
        """
        Adds a label with its description to the evaluation.

        :param label: The label or a tuple of labels that should be merged.
        :type label: Union[tuple, int]
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
        """Evaluates the metrics on the provided image and ground truth image.

        Args:
            image (sitk.Image): The segmented image.
            ground_truth (sitk.Image): The ground truth image.
            evaluation_id (str): The identification of the evaluation.
        """

        if not self.is_header_written:
            self.write_header()

        image_array = sitk.GetArrayFromImage(image) if isinstance(image, sitk.Image) else image
        ground_truth_array = sitk.GetArrayFromImage(ground_truth) if isinstance(ground_truth, sitk.Image) else ground_truth

        results = []  # clear results

        for label, label_str in self.labels.items():
            label_results = [evaluation_id, label_str]

            # get only current label
            predictions = np.in1d(image_array.ravel(), label, True).reshape(image_array.shape).astype(np.uint8)
            labels = np.in1d(ground_truth_array.ravel(), label, True).reshape(ground_truth_array.shape).astype(np.uint8)

            # calculate the confusion matrix for IConfusionMatrixMetric
            confusion_matrix = ConfusionMatrix(predictions, labels)

            # flag indicating whether the images have been converted for ISimpleITKImageMetric
            converted_to_image = False
            predictions_as_image = None
            labels_as_image = None

            # calculate the metrics
            for param_index, metric in enumerate(self.metrics):
                if isinstance(metric, IConfusionMatrixMetric):
                    metric.confusion_matrix = confusion_matrix
                elif isinstance(metric, INumpyArrayMetric):
                    metric.ground_truth = labels
                    metric.segmentation = predictions
                elif isinstance(metric, ISimpleITKImageMetric):
                    if not converted_to_image:
                        predictions_as_image = sitk.GetImageFromArray(predictions)
                        predictions_as_image.CopyInformation(image)
                        labels_as_image = sitk.GetImageFromArray(labels)
                        labels_as_image.CopyInformation(ground_truth)
                        converted_to_image = True

                    metric.ground_truth = labels_as_image
                    metric.segmentation = predictions_as_image

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
