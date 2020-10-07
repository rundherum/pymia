import argparse
import os

import batchgenerators.transforms as bg_tfm
import matplotlib.pyplot as plt
import numpy as np
import torchio as tio

import pymia.data.transformation as tfm
import pymia.data.augmentation as augm
import pymia.data.definition as defs
import pymia.data.extraction as extr


class BatchgeneratorsTransform(tfm.Transform):
    """Example wrapper for `batchgenerators <https://github.com/MIC-DKFZ/batchgenerators>`_ transformations."""

    def __init__(self, transforms, entries=(defs.KEY_IMAGES, defs.KEY_LABELS)) -> None:
        super().__init__()
        self.transforms = transforms
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        # unsqueeze samples to add a batch dimensions, as required by batchgenerators
        for entry in self.entries:
            if entry not in sample:
                if tfm.raise_error_if_entry_not_extracted:
                    raise ValueError(tfm.ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

            np_entry = tfm.check_and_return(sample[entry], np.ndarray)
            sample[entry] = np.expand_dims(np_entry, 0)

        # apply batchgenerators transforms
        sample = self.transforms(**sample)  # todo: make loop over transforms

        # squeeze samples back to original format
        for entry in self.entries:
            np_entry = tfm.check_and_return(sample[entry], np.ndarray)
            sample[entry] = np_entry.squeeze(0)

        return sample


class TorchIOTransform(tfm.Transform):
    """Example wrapper for `TorchIO <https://github.com/fepegar/torchio>`_ transformations."""

    def __init__(self, transforms: list, entries=(defs.KEY_IMAGES, defs.KEY_LABELS)) -> None:
        super().__init__()
        self.transforms = transforms
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        # unsqueeze samples to be 4-D tensors, as required by TorchIO
        for entry in self.entries:
            if entry not in sample:
                if tfm.raise_error_if_entry_not_extracted:
                    raise ValueError(tfm.ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

            np_entry = tfm.check_and_return(sample[entry], np.ndarray)
            sample[entry] = np.expand_dims(np_entry, -1)

        # apply TorchIO transforms
        for t in self.transforms:
            sample = t(sample)

        # squeeze samples back to original format
        for entry in self.entries:
            np_entry = tfm.check_and_return(sample[entry].numpy(), np.ndarray)
            sample[entry] = np_entry.squeeze(-1)

        return sample


def plot_sample(plot_dir: str, id_: str, sample: dict):
    plt.imsave(os.path.join(plot_dir, f'{id_}_image_channel0.png'), sample[defs.KEY_IMAGES][0])
    plt.imsave(os.path.join(plot_dir, f'{id_}_image_channel1.png'), sample[defs.KEY_IMAGES][1])
    plt.imsave(os.path.join(plot_dir, f'{id_}_label.png'), sample[defs.KEY_LABELS])


def main(hdf_file, plot_dir):
    # setup the datasource
    extractor = extr.DataExtractor(categories=(defs.KEY_IMAGES, defs.KEY_LABELS))
    indexing_strategy = extr.SliceIndexing()
    dataset = extr.PymiaDatasource(hdf_file, indexing_strategy, extractor)

    seed = 42
    np.random.seed(seed)
    sample_idx = 55

    # set up transformations without augmentation
    transforms_augmentation = []
    transforms_before_augmentation = [tfm.Permute(permutation=(2, 0, 1)), ]  # to have the channel-dimension first
    transforms_after_augmentation = [tfm.Squeeze(entries=(defs.KEY_LABELS,)), ]  # get rid of the channel-dimension for the labels
    train_transforms = tfm.ComposeTransform(transforms_before_augmentation + transforms_augmentation + transforms_after_augmentation)
    dataset.set_transform(train_transforms)
    sample = dataset[sample_idx]
    plot_sample(plot_dir, 'none', sample)

    # augmentation with pymia
    transforms_augmentation = [augm.RandomRotation90(axes=(-2, -1)), augm.RandomMirror()]
    train_transforms = tfm.ComposeTransform(
        transforms_before_augmentation + transforms_augmentation + transforms_after_augmentation)
    dataset.set_transform(train_transforms)
    sample = dataset[sample_idx]
    plot_sample(plot_dir, 'pymia', sample)

    # augmentation with TorchIO
    transforms_augmentation = [TorchIOTransform(
        [tio.RandomFlip(axes=('LR'), flip_probability=1.0, keys=(defs.KEY_IMAGES, defs.KEY_LABELS), seed=seed),
         tio.RandomAffine(scales=(0.9, 1.2), degrees=(10), isotropic=False, default_pad_value='otsu',
                          image_interpolation='NEAREST', keys=(defs.KEY_IMAGES, defs.KEY_LABELS), seed=seed),
         ])]
    train_transforms = tfm.ComposeTransform(
        transforms_before_augmentation + transforms_augmentation + transforms_after_augmentation)
    dataset.set_transform(train_transforms)
    sample = dataset[sample_idx]
    plot_sample(plot_dir, 'torchio', sample)

    # augmentation with batchgenerators
    transforms_augmentation = [BatchgeneratorsTransform(
        bg_tfm.spatial_transforms.Rot90Transform(axes=(0, 1), data_key=defs.KEY_IMAGES, label_key=defs.KEY_LABELS,
                                                 p_per_sample=1.0))]
    train_transforms = tfm.ComposeTransform(
        transforms_before_augmentation + transforms_augmentation + transforms_after_augmentation)
    dataset.set_transform(train_transforms)
    sample = dataset[sample_idx]
    plot_sample(plot_dir, 'batchgenerators', sample)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Augmentation example')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../example-data/example-dataset.h5',
        help='Path to the dataset file.'
    )

    parser.add_argument(
        '--plot_dir',
        type=str,
        default='../example-data/log',
        help='Path to the plotting directory.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.plot_dir)
