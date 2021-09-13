import argparse
import glob
import os

import numpy as np
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer
import SimpleITK as sitk


def main(data_dir: str, result_file: str, result_summary_file: str):
    # initialize metrics
    metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(percentile=95, metric='HDRFDST95'),
               metric.VolumeSimilarity()]

    # define the labels to evaluate
    labels = {1: 'WHITEMATTER',
              2: 'GREYMATTER',
              5: 'THALAMUS'
              }

    evaluator = eval_.SegmentationEvaluator(metrics, labels)

    # get subjects to evaluate
    subject_dirs = [subject for subject in glob.glob(os.path.join(data_dir, '*')) if
                    os.path.isdir(subject) and os.path.basename(subject).startswith('Subject')]

    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        print(f'Evaluating {subject_id}...')

        # load ground truth image and create artificial prediction by erosion
        ground_truth = sitk.ReadImage(os.path.join(subject_dir, f'{subject_id}_GT.mha'))
        prediction = ground_truth
        for label_val in labels.keys():
            # erode each label we are going to evaluate
            prediction = sitk.BinaryErode(prediction, [1] * prediction.GetDimension(), sitk.sitkBall, 0, label_val)

        # evaluate the "prediction" against the ground truth
        evaluator.evaluate(prediction, ground_truth, subject_id)

    # use two writers to report the results
    writer.CSVWriter(result_file).write(evaluator.results)

    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nAggregated statistic results...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Evaluation of results')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../example-data',
        help='Path to the data directory.'
    )

    parser.add_argument(
        '--result_file',
        type=str,
        default='../example-data/results.csv',
        help='Path to the result CSV file.'
    )

    parser.add_argument(
        '--result_summary_file',
        type=str,
        default='../example-data/results_summary.csv',
        help='Path to the result summary CSV file.'
    )

    args = parser.parse_args()
    main(args.data_dir, args.result_file, args.result_summary_file)
