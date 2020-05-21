import argparse
import glob
import os

import numpy as np
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer
import SimpleITK as sitk


def main(data_dir: str, result_file: str):
    # initialize the evaluator
    metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(percentile=95), metric.VolumeSimilarity()]
    labels = {1: 'DUMMY-LABEL',  # todo(fabianbalsiger): adapt labels to example
              }
    # evaluator = eval_.Evaluator(metrics, labels)
    evaluator = eval_.Evaluator(metrics, labels)

    # get list of subjects
    subject_dirs = [subject for subject in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(subject)]

    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        print(f'Evaluating {subject_id}...')

        # load ground truth image and create artificial prediction by erosion
        ground_truth = sitk.ReadImage(os.path.join(subject_dir, f'{subject_id}_GTN.mha'))
        prediction = sitk.BinaryErode(ground_truth)  # todo(fabianbalsiger): handle multi label dummy data (BRATS)

        # evaluate the "prediction" against the ground truth
        evaluator.evaluate(prediction, ground_truth, subject_id)

    # use two writers to report the results
    writer.CSVWriter(result_file).write(evaluator.results)
    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    writer.CSVStatisticsWriter(result_file.replace('.csv', '_summary.csv'),
                               functions={'MEAN': np.mean, 'STD': np.std}).write(evaluator.results)
    print('\nSummary results...')
    writer.ConsoleStatisticsWriter(functions={'MEAN': np.mean, 'STD': np.std}).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear_results()


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Result evaluation')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../dummy-data',
        help='Path to the data directory.'
    )

    parser.add_argument(
        '--result_file',
        type=str,
        default='../dummy-data/results.csv',
        help='Path to the result CSV file.'
    )

    args = parser.parse_args()
    main(args.data_dir, args.result_file)
