import argparse

try:
    import torch
except ImportError:
    raise ImportError('Module "torch" not found. Install "torch" to run this example.')


import numpy as np
import pymia.data.definition as defs
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer
import torch.utils.tensorboard as tensorboard


def main(hdf_file: str):
    epochs = 20
    batch_size_training = 10
    batch_size_testing = 10

    # initialize the evaluator
    metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(percentile=95), metric.VolumeSimilarity()]
    labels = {1: 'DUMMY-LABEL',  # todo(fabianbalsiger): adapt labels to example
              }
    evaluator = eval_.SegmentationEvaluator(metrics, labels)

    # initialize TensorBoard
    tb = tensorboard.SummaryWriter(run_dir)

    for epoch in epochs:
        pass

        for validation_batch in validation_dataset:
            assembled = subject_assembler.get_assembled_subject(sample[defs.KEY_SUBJECT_INDEX])
            prediction_image = pymia_conv.NumpySimpleITKImageBridge.convert(assembled, sample[defs.KEY_PROPERTIES])

            # evaluate the "prediction" against the ground truth
            evaluator.evaluate(prediction_image, label_image, sample[defs.KEY_SUBJECT])

        # calculate mean and standard deviation of each metric on the validation dataset
        results = writer.StatisticsAggregator(functions={'MEAN': np.mean, 'STD': np.std}).calculate(evaluator.results)
        # log to TensorBoard
        for result in results:
            tb.add_scalar(f'valid/{result.metric}-{result.id_}', result.value, epoch)

        # clear results such that the evaluator is ready for the next evaluation
        evaluator.clear()


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Training logging using pymia evaluation')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../dummy-data/dummy.h5',
        help='Path to the dataset file.'
    )

    args = parser.parse_args()
    main(args.hdf_file)
