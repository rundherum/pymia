import argparse

import torch.utils.data as torch_data

import pymia.data.definition as defs
import pymia.data.extraction as extr
import pymia.data.backends.pytorch as pymia_torch


def main(hdf_file: str):
    extractor = extr.ComposeExtractor([extr.NamesExtractor(),
                                       extr.DataExtractor(),
                                       extr.SelectiveDataExtractor(),
                                       extr.DataExtractor(('numerical',), ignore_indexing=True),
                                       extr.DataExtractor(('sex',), ignore_indexing=True),
                                       extr.DataExtractor(('mask',), ignore_indexing=False),
                                       extr.SubjectExtractor(),
                                       extr.IndexingExtractor(do_pickle=True)])

    direct_extractor = extr.ComposeExtractor([extr.SubjectExtractor(),
                                              extr.FilesExtractor(categories=(defs.KEY_IMAGES,
                                                                              defs.KEY_LABELS,
                                                                              'mask', 'numerical', 'sex')),
                                              extr.ImagePropertiesExtractor()])

    dataset = pymia_torch.PymiaTorchDataset(hdf_file, extr.SliceIndexing(), extractor)

    loader = torch_data.dataloader.DataLoader(dataset, batch_size=2, shuffle=False)

    for i, sample in enumerate(loader):

        for j in range(len(sample[defs.KEY_SUBJECT_INDEX])):
            subject_index = sample[defs.KEY_SUBJECT_INDEX][j].item()
            extracted_sample = dataset.direct_extract(direct_extractor, subject_index)


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
