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

    def add_writer(self, writer: IEvaluatorWriter):
        """
        Adds a writer to the evaluation.
        :param writer: The writer.
        :type writer: IEvaluatorWriter
        """

        self.writers.append(writer)
        self.is_header_written = False  # header changed

    def evaluate(self, image: sitk.Image, ground_truth: sitk.Image, evaluation_id: str):
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

        image_arr = sitk.GetArrayFromImage(image).flatten()
        ground_truth_arr = sitk.GetArrayFromImage(ground_truth).flatten()

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

                # TODO add other metric instances

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
