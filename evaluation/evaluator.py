import SimpleITK as sitk

from evaluation.evaluator_writer import IEvaluatorWriter
from evaluation.metric import IMetric, IConfusionMatrixMetric, ConfusionMatrix


class Evaluator:
    """
    Represents a metric evaluator.
    """
    def __init__(self, writer: IEvaluatorWriter):
        """
        Initializes a new instance of the Evaluator class.
        :param writer: A writer.
        """

        self.metrics = []
        self.results = []
        self.labels = [0, 1]
        self.label_str = ["Background", "Target"]
        self.writers = [writer]

    def evaluate(self, image: sitk.Image, ground_truth: sitk.Image, write=True):
        """
        Evaluates the metrics on the provided image and ground truth image.
        :param image: The image.
        :param ground_truth: The ground truth image.
        :param write: Determines whether to write the 
        """

        image_arr = sitk.GetArrayFromImage(image)
        ground_truth_arr = sitk.GetArrayFromImage(ground_truth)

        overlap = sitk.LabelOverlapMeasuresImageFilter()

        self.results = []  # clear results

        # calculate the metrics
        for param_index, metric in enumerate(self.metrics):
            if isinstance(metric, IConfusionMatrixMetric):
                metric.confusion_matrix = None

            self.results.append(metric.calculate(image_arr, ground_truth_arr))

        # write the results
        for writer in self.writers:
            writer.write()

    def add_metric(self, metric: IMetric):
        self.metrics.append(metric)
