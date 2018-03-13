import argparse
import enum
import os
import typing

import SimpleITK as sitk
import numpy as np

import miapy.data as miapy_data
import miapy.data.conversion as conv
import miapy.data.extraction as extr
import miapy.data.loading as miapy_load
import miapy.data.transformation as miapy_tfm
import miapy.data.creation.fileloader as file_load


def main(hdf_file: str):

    with extr.get_reader(hdf_file) as reader:
        extractor = extr.ComposeExtractor([extr.NamesExtractor(),
                                           extr.ImageExtractor(),
                                           extr.LabelExtractor(),
                                           extr.SupplementaryExtractor(('AGE', 'SEX'), entire_subject=True),
                                           extr.SupplementaryExtractor(('MASK',), entire_subject=False),
                                           extr.SubjectExtractor(),
                                           extr.FilesExtractor(),
                                           extr.IndexingExtractor(do_pickle_expression=False),
                                           extr.ImagePropertiesExtractor()])
        dataset = extr.ParameterizableDataset(reader, extr.SliceIndexing(), extractor)

        for i in range(len(dataset)):
            item = dataset[i]

            index_expr = item['index_expr']  # type: miapy_data.IndexExpression
            root = item['file_root']

            image = None  # type: sitk.Image
            for i, file in enumerate(item['image_files']):
                image = sitk.ReadImage(os.path.join(root, file))
                np_img = sitk.GetArrayFromImage(image)
                np_slice = np_img[index_expr.expression]
                np_slice = np_slice / np_slice.max()  # normalize
                if (np_slice != item['images'][..., i]).any():
                    raise ValueError('slice not equal')

            # for any image
            image_properties = conv.ImageProperties(image)

            if image_properties != item['image_properties']:
                raise ValueError('image properties not equal')

            for i, file in enumerate(item['label_files']):
                image = sitk.ReadImage(os.path.join(root, file))
                np_img = sitk.GetArrayFromImage(image)
                np_slice = np_img[index_expr.expression]
                if (np_slice != item['labels']).any():
                    raise ValueError('slice not equal')

            for i, file in enumerate(item['supplementary_files']):
                id_ = item['supplementary_names'][i]
                if id_ == 'MASK':
                    image = sitk.ReadImage(os.path.join(root, file))
                    np_img = sitk.GetArrayFromImage(image)
                    np_slice = np_img[index_expr.expression]
                    if (np_slice != item[id_]).any():
                        raise ValueError('slice not equal')
                else:
                    line = 0 if id_ == 'AGE' else 1
                    with open(os.path.join(root, file), 'r') as f:
                        value = f.readlines()[line].split(':')[1].strip()
                    if value != str(item[id_][0]):
                        raise ValueError('value not equal')


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Dataset creation')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='out/test/test.h5',
        help='Path to the dataset file.'
    )

    args = parser.parse_args()
    main(args.hdf_file)
