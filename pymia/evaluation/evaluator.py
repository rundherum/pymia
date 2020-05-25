"""The evaluator module simplifies the evaluation of results.

The module provides the possibility of calculate several evaluation metrics in parallel and output them in any format.
"""
import abc
import typing

import numpy as np
import SimpleITK as sitk

import pymia.evaluation.metric as pymia_metric


class Result:

    def __init__(self, id_: str, label: str, metric: str, value):
        self.id_ = id_
        self.label = label
        self.metric = metric
        self.value = value


class EvaluatorBase(abc.ABC):

    def __init__(self, metrics: typing.List[pymia_metric.IMetric]):
        self.metrics = metrics
        self.results = []

    @abc.abstractmethod
    def evaluate(self,
                 prediction: typing.Union[sitk.Image, np.ndarray],
                 reference: typing.Union[sitk.Image, np.ndarray],
                 id_: str, **kwargs):
        raise NotImplementedError

    def clear(self):
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

        prediction_array = sitk.GetArrayFromImage(prediction) if isinstance(prediction, sitk.Image) else prediction
        reference_array = sitk.GetArrayFromImage(reference) if isinstance(reference, sitk.Image) else reference

        for label, label_str in self.labels.items():
            # get only current label
            prediction_of_label = np.in1d(prediction_array.ravel(), label, True).reshape(prediction_array.shape).astype(np.uint8)
            reference_of_label = np.in1d(reference_array.ravel(), label, True).reshape(reference_array.shape).astype(np.uint8)

            # calculate the confusion matrix for IConfusionMatrixMetric
            confusion_matrix = pymia_metric.ConfusionMatrix(prediction_of_label, reference_of_label)

            # flag indicating whether the images have been converted for ISimpleITKImageMetric
            converted_to_image = False
            prediction_of_label_as_image = None
            reference_of_label_as_image = None

            # for distance metrics
            distances = None

            # calculate the metrics
            for param_index, metric in enumerate(self.metrics):
                if isinstance(metric, pymia_metric.IConfusionMatrixMetric):
                    metric.confusion_matrix = confusion_matrix
                elif isinstance(metric, pymia_metric.INumpyArrayMetric):
                    metric.reference = reference_of_label
                    metric.prediction = prediction_of_label
                elif isinstance(metric, pymia_metric.ISimpleITKImageMetric):
                    if not converted_to_image:
                        if not isinstance(prediction, sitk.Image):
                            raise ValueError('SimpleITK image is required for SimpleITK-based metrics')
                        prediction_of_label_as_image = sitk.GetImageFromArray(prediction_of_label)
                        prediction_of_label_as_image.CopyInformation(prediction)
                        reference_of_label_as_image = sitk.GetImageFromArray(reference_of_label)
                        reference_of_label_as_image.CopyInformation(reference)
                        converted_to_image = True

                    metric.reference = reference_of_label_as_image
                    metric.prediction = prediction_of_label_as_image
                elif isinstance(metric, pymia_metric.IDistanceMetric):
                    if distances is None:
                        if isinstance(prediction, sitk.Image):
                            spacing = prediction.GetSpacing()[::-1]
                        else:
                            spacing = (1.0,) * reference_of_label.ndim  # use isotropic spacing of 1 mm

                        distances = pymia_metric.Distances(prediction_of_label, reference_of_label, spacing)

                    metric.distances = distances

                self.results.append(Result(id_, label_str, metric.metric, metric.calculate()))
