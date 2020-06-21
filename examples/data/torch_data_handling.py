import argparse

import torch.utils.data as torch_data
import torch.nn as nn
import torch

import pymia.data.assembler as assm
import pymia.data.transformation as tfm
import pymia.data.definition as defs
import pymia.data.extraction as extr
import pymia.data.backends.pytorch as pymia_torch




def main(hdf_file: str):

    hdf_file = '../example-data/example-data.h5'
    # train_subjects, valid_subjects = ['Subject_1', 'Subject_2', 'Subject_3'], ['Subject_4']

    extractor = extr.ComposeExtractor(
        [extr.DataExtractor(categories=(defs.KEY_IMAGES,)),
         extr.SubjectExtractor(),
         extr.IndexingExtractor(do_pickle=True),

         extr.ImageShapeExtractor()]
    )

    transform = tfm.Permute(permutation=(2, 1, 0), entries=(defs.KEY_IMAGES,))

    indexing_strategy = extr.SliceIndexing()
    dataset = extr.PymiaDatasource(hdf_file, indexing_strategy, extractor, transform)

    direct_extractor = extr.ComposeExtractor(
        [extr.ImagePropertiesExtractor(),
         extr.DataExtractor(categories=(defs.KEY_LABELS,))]
    )
    assembler = assm.SubjectAssembler()


    # torch specific handling
    pytorch_dataset = pymia_torch.PytorchDatasetAdapter(dataset)
    loader = torch_data.dataloader.DataLoader(pytorch_dataset, batch_size=2, shuffle=False)
    dummy_network = nn.Sequential(
        nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )
    torch.set_grad_enabled(False)

    for i, batch in enumerate(loader):

        x = batch[defs.KEY_IMAGES]

        prediction = dummy_network(x)

        is_last = i == len(loader) - 1
        assembler.add_batch(prediction.numpy(), batch, is_last)

        for subject_index in assembler.subjects_ready:
            subject_prediction = assembler.get_assembled_subject(subject_index)

            direct_sample = dataset.direct_extract(direct_extractor, subject_index)

            a = 1
            # do_eval(subject_prediction, direct_sample[defs.KEY_LABELS])




if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='PyTorch data access verification')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../example-data/dummy.h5',
        help='Path to the dataset file.'
    )

    args = parser.parse_args()
    main(args.hdf_file)
