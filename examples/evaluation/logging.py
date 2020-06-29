import argparse

# try:
import torch
# except ImportError:
#     raise ImportError('Module "torch" not found. Install "torch" to run this example.')


import numpy as np
import pymia.data.assembler as assm
import pymia.data.backends.pytorch as pymia_torch
import pymia.data.conversion as conv
import pymia.data.definition as defs
import pymia.data.extraction as extr
import pymia.data.transformation as tfm
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer
import torch.nn as nn
import torch.utils.data as torch_data
import torch.utils.tensorboard as tensorboard


def main(hdf_file: str, log_dir: str):
    # initialize the evaluator with the metrics and the labels to evaluate
    metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(percentile=95, metric='HDRFDST95'),
               metric.VolumeSimilarity()]
    labels = {1: 'WHITEMATTER',
              2: 'GREYMATTER',
              5: 'THALAMUS'
              }
    evaluator = eval_.SegmentationEvaluator(metrics, labels)

    # we want to log the mean and standard deviation of the metrics among all subjects of the dataset
    functions = {'MEAN': np.mean, 'STD': np.std}
    statistics_aggregator = writer.StatisticsAggregator(functions=functions)

    # initialize TensorBoard writer
    tb = tensorboard.SummaryWriter(log_dir)

    # initialize the data handling
    transform = tfm.Permute(permutation=(2, 0, 1), entries=(defs.KEY_IMAGES,))
    dataset = extr.PymiaDatasource(hdf_file, extr.SliceIndexing(), extr.DataExtractor(categories=(defs.KEY_IMAGES,)), transform)
    pytorch_dataset = pymia_torch.PytorchDatasetAdapter(dataset)
    loader = torch_data.dataloader.DataLoader(pytorch_dataset, batch_size=2, shuffle=False)

    assembler = assm.SubjectAssembler(dataset)
    direct_extractor = extr.ComposeExtractor([
        extr.SubjectExtractor(),  # extraction of the subject name
        extr.ImagePropertiesExtractor(),  # Extraction of image properties (origin, spacing, etc.) for storage
        extr.DataExtractor(categories=(defs.KEY_LABELS,))  # Extraction of "labels" entries for evaluation
    ])

    # initialize a dummy network, which returns a random prediction
    class DummyNetwork(nn.Module):

        def forward(self, x):
            return torch.randint(0, 5, (*x.size()[:-1], 1))

    dummy_network = DummyNetwork()
    torch.manual_seed(0)  # set seed for reproducibility

    nb_batches = len(loader)

    for epoch in range(10):
        for i, batch in enumerate(loader):
            # get the data from batch and predict
            x, sample_indices = batch[defs.KEY_IMAGES], batch[defs.KEY_SAMPLE_INDEX]
            prediction = dummy_network(x)

            # translate the prediction to numpy and back to (B)HWC (channel last)
            numpy_prediction = prediction.numpy().transpose((0, 2, 3, 1))

            # add the batch prediction to the assembler
            is_last = i == nb_batches - 1
            assembler.add_batch(numpy_prediction, sample_indices.numpy(), is_last)

            # Process the subjects/images that are fully assembled
            for subject_index in assembler.subjects_ready:
                subject_prediction = assembler.get_assembled_subject(subject_index)

                # Extract the target and image properties via direct extract
                direct_sample = dataset.direct_extract(direct_extractor, subject_index)
                reference, image_properties = direct_sample[defs.KEY_LABELS], direct_sample[defs.KEY_PROPERTIES]

                # convert prediction and reference back to SimpleITK images, as we want to use the physical spacing for the Hausdorff distance metric
                prediction_image = conv.NumpySimpleITKImageBridge.convert(subject_prediction, image_properties)
                reference_image = conv.NumpySimpleITKImageBridge.convert(reference, image_properties)

                # evaluate the prediction against the reference
                evaluator.evaluate(prediction_image, reference_image, direct_sample[defs.KEY_SUBJECT])

        # calculate mean and standard deviation of each metric
        results = statistics_aggregator.calculate(evaluator.results)
        # log to TensorBoard into category train
        for result in results:
            tb.add_scalar(f'train/{result.metric}-{result.id_}', result.value, epoch)

        # clear results such that the evaluator is ready for the next evaluation
        evaluator.clear()


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Logging the training progress')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../example-data/example-dataset.h5',
        help='Path to the dataset file.'
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default='../example-data/log',
        help='Path to the logging directory.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.log_dir)
