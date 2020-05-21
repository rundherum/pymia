"""The evaluator module simplifies the evaluation of results.

The module provides the possibility of calculate several evaluation metrics in parallel and output them in any format.
"""
import abc
import typing

import numpy as np
import SimpleITK as sitk

import pymia.evaluation.metric as pymia_metric


class Result:

    def __init__(self, id_: str, label: str, metric: str, value: float):
        self.id_ = id_
        self.label = label
        self.metric = metric
        self.value = value


class EvaluatorBase(abc.ABC):

    def __init__(self, metrics: typing.List[pymia_metric.IMetric]):
        self.metrics = metrics if metrics else []
        self.results = []

    @abc.abstractmethod
    def evaluate(self,
                 prediction: typing.Union[sitk.Image, np.ndarray],
                 reference: typing.Union[sitk.Image, np.ndarray],
                 id_: str, **kwargs):
        raise NotImplementedError

    def add_metric(self, metric: pymia_metric.IMetric):
        """Adds a metric to the evaluation.

        Args:
            metric (pymia_metric.IMetric): The metric.
        """
        self.metrics.append(metric)

    def clear_results(self):
        self.results = []


class Evaluator(EvaluatorBase):
    """Represents an evaluator, evaluating metrics on predictions against references."""

    def __init__(self, metrics: typing.List[pymia_metric.IMetric], labels: dict):
        """Initializes a new instance of the Evaluator class.

        Args:
            metrics (list of pymia_metric.IMetric): A list of metrics.
            labels (dict): A dictionary with labels (key of type int) and label descriptions (value of type string).
        """
        super().__init__(metrics)
        self.labels = labels if labels else {}

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
        """Evaluates the metrics on the provided image and ground truth image.

        Args:
            prediction (sitk.Image): The segmented image.
            reference (sitk.Image): The ground truth image.
            id_ (str): The identification of the subject to evaluate.

        Raises:
            ValueError: If no labels are defined (see add_label).
        """

        if not self.labels:
            raise ValueError('No labels to evaluate defined')

        if isinstance(prediction, sitk.Image) and prediction.GetNumberOfComponentsPerPixel() > 1:
            raise ValueError('Image has more than one component per pixel')

        if isinstance(reference, sitk.Image) and reference.GetNumberOfComponentsPerPixel() > 1:
            raise ValueError('Image has more than one component per pixel')

        image_array = sitk.GetArrayFromImage(prediction) if isinstance(prediction, sitk.Image) else prediction
        ground_truth_array = sitk.GetArrayFromImage(reference) if isinstance(reference, sitk.Image) else reference

        for label, label_str in self.labels.items():
            # get only current label
            predictions = np.in1d(image_array.ravel(), label, True).reshape(image_array.shape).astype(np.uint8)
            labels = np.in1d(ground_truth_array.ravel(), label, True).reshape(ground_truth_array.shape).astype(np.uint8)

            # calculate the confusion matrix for IConfusionMatrixMetric
            confusion_matrix = pymia_metric.ConfusionMatrix(predictions, labels)

            # flag indicating whether the images have been converted for ISimpleITKImageMetric
            converted_to_image = False
            predictions_as_image = None
            labels_as_image = None

            # for distance metrics
            distances = None

            # calculate the metrics
            for param_index, metric in enumerate(self.metrics):
                if isinstance(metric, pymia_metric.IConfusionMatrixMetric):
                    metric.confusion_matrix = confusion_matrix
                elif isinstance(metric, pymia_metric.INumpyArrayMetric):
                    metric.ground_truth = labels
                    metric.segmentation = predictions
                elif isinstance(metric, pymia_metric.ISimpleITKImageMetric):
                    if not converted_to_image:
                        if not isinstance(prediction, sitk.Image):
                            raise ValueError('SimpleITK image is required for SimpleITK-based metrics')
                        predictions_as_image = sitk.GetImageFromArray(predictions)
                        predictions_as_image.CopyInformation(prediction)
                        labels_as_image = sitk.GetImageFromArray(labels)
                        labels_as_image.CopyInformation(reference)
                        converted_to_image = True

                    metric.ground_truth = labels_as_image
                    metric.segmentation = predictions_as_image
                elif isinstance(metric, pymia_metric.IDistanceMetric):
                    if distances is None:
                        if isinstance(prediction, sitk.Image):
                            spacing = prediction.GetSpacing()[::-1]
                        else:
                            spacing = (1.0,) * labels.ndim  # use isotropic spacing of 1 mm

                        distances = pymia_metric.Distances(predictions, labels, spacing)

                    metric.distances = distances

                self.results.append(Result(id_, label_str, metric.metric, metric.calculate()))
