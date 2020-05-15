import argparse
import glob
import os

import pymia.evaluation.evaluator as eval_
import pymia.evaluation.metric as metric


def init_evaluator(directory: str, result_file_name: str = 'results.csv') -> eval_.Evaluator:
    """Initializes an evaluator.
    Args:
        directory (str): The directory for the results file.
        result_file_name (str): The result file name (CSV file).
    Returns:
        eval.Evaluator: An evaluator.
    """
    os.makedirs(directory, exist_ok=True)  # generate result directory, if it does not exists

    evaluator = eval_.Evaluator(eval_.ConsoleEvaluatorWriter(5))
    evaluator.add_writer(eval_.CSVEvaluatorWriter(os.path.join(directory, result_file_name)))
    evaluator.add_label(1, 'WhiteMatter')
    evaluator.add_label(2, 'GreyMatter')
    evaluator.add_label(3, 'Hippocampus')
    evaluator.add_label(4, 'Amygdala')
    evaluator.add_label(5, 'Thalamus')
    evaluator.metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(percentile=95), metric.VolumeSimilarity()]
    return evaluator


def main(data_dir: str, result_file: str):
    result_dir = '/home/user/Desktop'  # todo
    evaluator = init_evaluator(result_dir)

    case_ids = []  # todo: names of the cases (string)
    predictions = []  # todo: list with SimpleITK or numpy array predictions
    labels = []  # todo: list with SimpleITK or numpy array labels

    for case, prediction, label in zip(case_ids, predictions, labels):
        evaluator.evaluate(prediction, label, case)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Result evaluation')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../out',
        help='Path to the data directory.'
    )

    parser.add_argument(
        '--result_file',
        type=str,
        default='../out/results.csv',
        help='Path to the result CSV file.'
    )

    args = parser.parse_args()
    main(args.data_dir, args.result_file)
