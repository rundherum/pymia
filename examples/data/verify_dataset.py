import argparse
import os

import SimpleITK as sitk
import numpy as np

import pymia.data as data
import pymia.data.conversion as conv
import pymia.data.definition as defs
import pymia.data.extraction as extr


def main(hdf_file: str):
    extractor = extr.ComposeExtractor([extr.NamesExtractor(),
                                       extr.DataExtractor(),
                                       extr.SelectiveDataExtractor(),
                                       extr.DataExtractor(('numerical', ), ignore_indexing=True),
                                       extr.DataExtractor(('sex', ), ignore_indexing=True),
                                       extr.DataExtractor(('mask', ), ignore_indexing=False),
                                       extr.SubjectExtractor(),
                                       extr.FilesExtractor(categories=(defs.KEY_IMAGES,
                                                                       defs.KEY_LABELS,
                                                                       'mask', 'numerical', 'sex')),
                                       extr.IndexingExtractor(),
                                       extr.ImagePropertiesExtractor()])
    dataset = extr.PymiaDatasource(hdf_file, extr.SliceIndexing(), extractor)

    for i in range(len(dataset)):
        item = dataset[i]

        index_expr = item[defs.KEY_INDEX_EXPR]  # type: data.IndexExpression
        root = item['file_root']

        image = None  # type: sitk.Image
        for i, file in enumerate(item['images_files']):
            image = sitk.ReadImage(os.path.join(root, file))
            np_img = sitk.GetArrayFromImage(image).astype(np.float32)
            np_img = (np_img - np_img.mean()) / np_img.std()
            np_slice = np_img[index_expr.expression]
            if (np_slice != item[defs.KEY_IMAGES][..., i]).any():
                raise ValueError('slice not equal')

        # for any image
        image_properties = conv.ImageProperties(image)

        if image_properties != item[defs.KEY_PROPERTIES]:
            raise ValueError('image properties not equal')

        for file in item['labels_files']:
            image = sitk.ReadImage(os.path.join(root, file))
            np_img = sitk.GetArrayFromImage(image)
            np_img = np.expand_dims(np_img, axis=-1)  # due to the convention of having the last dim as number of channels
            np_slice = np_img[index_expr.expression]
            if (np_slice != item[defs.KEY_LABELS]).any():
                raise ValueError('slice not equal')

        for file in item['mask_files']:
            image = sitk.ReadImage(os.path.join(root, file))
            np_img = sitk.GetArrayFromImage(image)
            np_img = np.expand_dims(np_img, axis=-1)  # due to the convention of having the last dim as number of channels
            np_slice = np_img[index_expr.expression]
            if (np_slice != item['mask']).any():
                raise ValueError('slice not equal')

        for file in item['numerical_files']:
            with open(os.path.join(root, file), 'r') as f:
                lines = f.readlines()
            age = float(lines[0].split(':')[1].strip())
            gpa = float(lines[1].split(':')[1].strip())
            if age != item['numerical'][0][0] or gpa != item['numerical'][0][1]:
                raise ValueError('value not equal')

        for file in item['sex_files']:
            with open(os.path.join(root, file), 'r') as f:
                sex = f.readlines()[2].split(':')[1].strip()
            if sex != str(item['sex'][0]):
                raise ValueError('value not equal')

    print('All test passed!')


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Dataset verification')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../example-data/dummy.h5',
        help='Path to the dataset file.'
    )

    args = parser.parse_args()
    main(args.hdf_file)
