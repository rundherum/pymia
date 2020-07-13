"""The evaluator module provides classes to evaluate the metrics on predictions.

All evaluators inherit the :class:`pymia.evaluation.evaluator.Evaluator`, which contains a list of results after
calling :meth:`pymia.evaluation.evaluator.Evaluator.evaluate`. The results can be passed to a writer of the
:mod:`pymia.evaluation.writer` module.
"""
import abc
import typing

import numpy as np
import SimpleITK as sitk

import pymia.evaluation.metric as pymia_metric


class Result:

    def __init__(self, id_: str, label: str, metric: str, value):
        """Represents a result.

        Args:
            id_ (str): The identification of the result (e.g., the subject's name).
            label (str): The label of the result (e.g., the foreground).
            metric (str): The metric.
            value (int, float): The value of the metric.
        """
        self.id_ = id_
        self.label = label
        self.metric = metric
        self.value = value


class Evaluator(abc.ABC):

    def __init__(self, metrics: typing.List[pymia_metric.Metric]):
        """Evaluator base class.

        Args:
            metrics (list of pymia_metric.Metric): A list of metrics.
        """
        self.metrics = metrics
        self.results = []

    @abc.abstractmethod
    def evaluate(self,
                 prediction: typing.Union[sitk.Image, np.ndarray],
                 reference: typing.Union[sitk.Image, np.ndarray],
                 id_: str, **kwargs):
        """Evaluates the metrics on the provided prediction and reference.

        Args:
            prediction (typing.Union[sitk.Image, np.ndarray]): The prediction.
            reference (typing.Union[sitk.Image, np.ndarray]): The reference.
            id_ (str): The identification of the case to evaluate.
        """
        raise NotImplementedError

    def clear(self):
        """Clears the results."""
        self.results = []


class SegmentationEvaluator(Evaluator):

    def __init__(self, metrics: typing.List[pymia_metric.Metric], labels: dict):
        """Represents a segmentation evaluator, evaluating metrics on predictions against references.

        Args:
            metrics (list of pymia_metric.Metric): A list of metrics.
            labels (dict): A dictionary with labels (key of type int) and label descriptions (value of type string).
        """
        super().__init__(metrics)
        self.labels = labels

    def add_label(self, label: typing.Union[tuple, int], description: str):
        """Adds a label with its description to the evaluation.

        Args:
            label (Union[tuple, int]): The label or a tuple of labels that should be merged.
            description (str): The label's description.
        """
        self.labels[label] = description

    def evaluate(self,
                 prediction: typing.Union[sitk.Image, np.ndarray],
                 reference: typing.Union[sitk.Image, np.ndarray],
                 id_: str, **kwargs):
        """Evaluates the metrics on the provided prediction and reference image.

        Args:
            prediction (typing.Union[sitk.Image, np.ndarray]): The predicted image.
            reference (typing.Union[sitk.Image, np.ndarray]): The reference image.
            id_ (str): The identification of the case to evaluate.

        Raises:
            ValueError: If no labels are defined (see add_label).
        """

        if not self.labels:
            raise ValueError('No labels to evaluate defined')

        if isinstance(prediction, sitk.Image) and prediction.GetNumberOfComponentsPerPixel() > 1:
            raise ValueError('Image has more than one component per pixel')

        if isinstance(reference, sitk.Image) and reference.GetNumberOfComponentsPerPixel() > 1:
            raise ValueError('Image has more than one component per pixel')

        prediction_array = sitk.GetArrayFromImage(prediction) if isinstance(prediction, sitk.Image) else prediction
        reference_array = sitk.GetArrayFromImage(reference) if isinstance(reference, sitk.Image) else reference

        for label, label_str in self.labels.items():
            # get only current label
            prediction_of_label = np.in1d(prediction_array.ravel(), label, True).reshape(prediction_array.shape).astype(np.uint8)
            reference_of_label = np.in1d(reference_array.ravel(), label, True).reshape(reference_array.shape).astype(np.uint8)

            # calculate the confusion matrix for ConfusionMatrixMetric
            confusion_matrix = pymia_metric.ConfusionMatrix(prediction_of_label, reference_of_label)

            # for distance metrics
            distances = None

            # spacing depends on SimpleITK image properties or an isotropic spacing as fallback
            def get_spacing():
                if isinstance(prediction, sitk.Image):
                    return prediction.GetSpacing()[::-1]
                else:
                    return (1.0,) * reference_of_label.ndim  # use isotropic spacing of 1 mm

            # calculate the metrics
            for param_index, metric in enumerate(self.metrics):
                if isinstance(metric, pymia_metric.ConfusionMatrixMetric):
                    metric.confusion_matrix = confusion_matrix
                # ensure this is checked before NumpyArrayMetric as SpacingMetric is itself a NumpyArrayMetric
                elif isinstance(metric, pymia_metric.SpacingMetric):
                    metric.reference = reference_of_label
                    metric.prediction = prediction_of_label
                    metric.spacing = get_spacing()
                elif isinstance(metric, pymia_metric.NumpyArrayMetric):
                    metric.reference = reference_of_label
                    metric.prediction = prediction_of_label
                elif isinstance(metric, pymia_metric.DistanceMetric):
                    if distances is None:
                        # calculate distances only once
                        distances = pymia_metric.Distances(prediction_of_label, reference_of_label, get_spacing())
                    metric.distances = distances

                self.results.append(Result(id_, label_str, metric.metric, metric.calculate()))
