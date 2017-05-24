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
        :type writer: IEvaluatorWriter
        """

        self.metrics = []
        self.labels = [0, 1]
        self.label_str = ["Background", "Target"]
        self.writers = [writer]
        self.is_header_written = False

    def add_metric(self, metric: IMetric):
        """
        Adds a metric to the evaluation.
        :param metric: The metric.
        :type metric: IMetric
        """

        self.metrics.append(metric)

    def add_writer(self, writer: IEvaluatorWriter):
        """
        Adds a writer to the evaluation.
        :param writer: The writer.
        :type writer: IEvaluatorWriter
        """

        self.writers.append(writer)
        self.is_header_written = False  # header changed

    def evaluate(self, image: sitk.Image, ground_truth: sitk.Image):
        """
        Evaluates the metrics on the provided image and ground truth image.
        :param image: The image.
        :param ground_truth: The ground truth image.
        """

        if not self.is_header_written:
            self.write_header()

        image_arr = sitk.GetArrayFromImage(image)
        ground_truth_arr = sitk.GetArrayFromImage(ground_truth)

        confusion_matrix = ConfusionMatrix()  # calculate the confusion matrix

        results = []  # clear results

        # calculate the metrics
        for param_index, metric in enumerate(self.metrics):
            if isinstance(metric, IConfusionMatrixMetric):
                metric.confusion_matrix = confusion_matrix

            results.append(metric.calculate(image_arr, ground_truth_arr))

        # write the results
        for writer in self.writers:
            writer.write(results)

    def write_header(self):

        header = ["ID", "LABEL"]

        for metric in self.metrics:
            pass

        for writer in self.writers:
            writer.write_header(header)
