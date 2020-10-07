import argparse

import matplotlib.pyplot as plt
import torch.utils.data as torch_data

import batchgenerators.transforms as bg_tfm
import pymia.data.transformation as tfm
import pymia.data.definition as defs
import pymia.data.extraction as extr
import pymia.data.backends.pytorch as pymia_torch


class BatchGenWrapper(tfm.Transform):

    def __init__(self, batchgen_tfm) -> None:
        super().__init__()
        self.batchgen_tfm = batchgen_tfm

    def __call__(self, sample: dict) -> dict:
        return self.batchgen_tfm(**sample)



def main(hdf_file):

    extractor = extr.DataExtractor(categories=(defs.KEY_IMAGES, defs.KEY_LABELS))

    augmentation_transforms = [BatchGenWrapper(bg_tfm.spatial_transforms.Rot90Transform(axes=(0,), data_key=defs.KEY_IMAGES, label_key=defs.KEY_LABELS, p_per_sample=1.))]
    # transforms = [tfm.Permute(permutation=(2, 0, 1))]
    train_transforms = tfm.ComposeTransform(augmentation_transforms)
    # train_transforms = tfm.ComposeTransform(augmentation_transforms + transforms)

    indexing_strategy = extr.SliceIndexing()
    dataset = extr.PymiaDatasource(hdf_file, indexing_strategy, extractor, train_transforms)

    sample = dataset[30]
    x, y = sample[defs.KEY_IMAGES], sample[defs.KEY_LABELS]

    plt.imsave('./x.png', x)
    plt.imsave('./y.png', y)

    # # torch specific handling
    # pytorch_dataset = pymia_torch.PytorchDatasetAdapter(dataset)
    # loader = torch_data.dataloader.DataLoader(pytorch_dataset, batch_size=2, shuffle=False)
    #
    # nb_batches = len(loader)
    #
    # # looping over the data in the dataset
    # for i, batch in enumerate(loader):
    #
    #     x, y = batch[defs.KEY_IMAGES], batch[defs.KEY_LABELS]




if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Creation')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../example-data/example-dataset.h5',
        help='Path to the dataset file.'
    )

    args = parser.parse_args()
    main(args.hdf_file)
