import argparse
import os
import sys
sys.path.append('./helper')

import numpy as np
import torch.utils.tensorboard as tensorboard
import torch.utils.data as torch_data
import torch.nn.functional as F
import torch.optim as optim
import torch

import pymia.data.assembler as assm
import pymia.data.augmentation as augm
import pymia.data.transformation as tfm
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer
import pymia.data.definition as defs
import pymia.data.extraction as extr
import pymia.data.backends.pytorch as pymia_torch

import helper.unet_pytorch as unet


device = 'cuda'


def main(hdf_file, log_dir):
    # initialize the evaluator with the metrics and the labels to evaluate
    metrics = [metric.DiceCoefficient()]
    labels = {1: 'WHITEMATTER',
              2: 'GREYMATTER',
              3: 'HIPPOCAMPUS',
              4: 'AMYGDALA',
              5: 'THALAMUS'}
    evaluator = eval_.SegmentationEvaluator(metrics, labels)

    # we want to log the mean and standard deviation of the metrics among all subjects of the dataset
    functions = {'MEAN': np.mean, 'STD': np.std}
    statistics_aggregator = writer.StatisticsAggregator(functions=functions)
    console_writer = writer.ConsoleStatisticsWriter(functions=functions)

    # initialize TensorBoard writer
    tb = tensorboard.SummaryWriter(os.path.join(log_dir, 'logging-example-torch'))

    # setup the training datasource
    train_subjects, valid_subjects = ['Subject_1', 'Subject_2', 'Subject_3'], ['Subject_4']
    extractor = extr.DataExtractor(categories=(defs.KEY_IMAGES, defs.KEY_LABELS))
    indexing_strategy = extr.SliceIndexing()

    augmentation_transforms = [augm.RandomRotation90(axes=(-2, -1)), augm.RandomMirror()]
    transforms = [tfm.Permute(permutation=(2, 0, 1)), tfm.Squeeze(entries=(defs.KEY_LABELS,))]
    train_transforms = tfm.ComposeTransform(transforms + augmentation_transforms)
    train_dataset = extr.PymiaDatasource(hdf_file, indexing_strategy, extractor, train_transforms,
                                         subject_subset=train_subjects)

    # setup the validation datasource
    valid_transforms = tfm.ComposeTransform([tfm.Permute(permutation=(2, 0, 1))])
    valid_dataset = extr.PymiaDatasource(hdf_file, indexing_strategy, extractor, valid_transforms,
                                         subject_subset=valid_subjects)
    direct_extractor = extr.ComposeExtractor(
        [extr.SubjectExtractor(),
         extr.ImagePropertiesExtractor(),
         extr.DataExtractor(categories=(defs.KEY_LABELS,))]
    )
    assembler = assm.SubjectAssembler(valid_dataset)

    # torch specific handling
    pytorch_train_dataset = pymia_torch.PytorchDatasetAdapter(train_dataset)
    train_loader = torch_data.dataloader.DataLoader(pytorch_train_dataset, batch_size=16, shuffle=True)

    pytorch_valid_dataset = pymia_torch.PytorchDatasetAdapter(valid_dataset)
    valid_loader = torch_data.dataloader.DataLoader(pytorch_valid_dataset, batch_size=16, shuffle=False)

    u_net = unet.UNetModel(ch_in=2, ch_out=6, n_channels=16, n_pooling=3).to(device)

    print(u_net)

    optimizer = optim.Adam(u_net.parameters(), lr=1e-3)
    train_batches = len(train_loader)

    # looping over the data in the dataset
    epochs = 100
    for epoch in range(epochs):
        u_net.train()
        print(f'Epoch {epoch + 1}/{epochs}')

        # training
        print('training')
        for i, batch in enumerate(train_loader):
            x, y = batch[defs.KEY_IMAGES].to(device), batch[defs.KEY_LABELS].to(device).long()
            logits = u_net(x)

            optimizer.zero_grad()
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            tb.add_scalar('train/loss', loss.item(), epoch*train_batches + i)
            print(f'[{i + 1}/{train_batches}]\tloss: {loss.item()}')

        # validation
        print('validation')
        with torch.no_grad():
            u_net.eval()
            valid_batches = len(valid_loader)
            for i, batch in enumerate(valid_loader):
                x, sample_indices = batch[defs.KEY_IMAGES].to(device), batch[defs.KEY_SAMPLE_INDEX]

                logits = u_net(x)
                prediction = logits.argmax(dim=1, keepdim=True)

                numpy_prediction = prediction.cpu().numpy().transpose((0, 2, 3, 1))

                is_last = i == valid_batches - 1
                assembler.add_batch(numpy_prediction, sample_indices.numpy(), is_last)

                for subject_index in assembler.subjects_ready:
                    subject_prediction = assembler.get_assembled_subject(subject_index)

                    direct_sample = train_dataset.direct_extract(direct_extractor, subject_index)
                    target, image_properties = direct_sample[defs.KEY_LABELS],  direct_sample[defs.KEY_PROPERTIES]

                    # evaluate the prediction against the reference
                    evaluator.evaluate(subject_prediction[..., 0], target[..., 0], direct_sample[defs.KEY_SUBJECT])

            # calculate mean and standard deviation of each metric
            results = statistics_aggregator.calculate(evaluator.results)
            # log to TensorBoard into category train
            for result in results:
                tb.add_scalar(f'valid/{result.metric}-{result.id_}', result.value, epoch)

            console_writer.write(evaluator.results)

            # clear results such that the evaluator is ready for the next evaluation
            evaluator.clear()


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Training example PyTorch')

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
