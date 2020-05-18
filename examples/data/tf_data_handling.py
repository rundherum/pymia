import argparse

import tensorflow as tf

import pymia.data.extraction as extr
import pymia.data.backends.tensorflow as pymia_tf


def main(hdf_file: str):
    # todo(alain): add other extractors?
    extractor = extr.ComposeExtractor([extr.DataExtractor(),
                                       extr.SubjectExtractor()])

    direct_extractor = extr.ComposeExtractor([extr.SubjectExtractor(),
                                              extr.FilesExtractor(categories=('images', 'labels', 'mask', 'numerical', 'sex')),
                                              extr.ImagePropertiesExtractor()])

    data_source = extr.dataset.ParameterizableDataset(hdf_file, extr.SliceIndexing(), extractor)

    gen_fn = pymia_tf.get_tf_generator(data_source)
    dataset = tf.data.Dataset.from_generator(generator=gen_fn,
                                             output_types={'images': tf.float32, 'subject_index': tf.int64, 'subject': tf.string})

    dataset = dataset.batch(2)

    for i, sample in enumerate(dataset.as_numpy_iterator()):

        for j in range(len(sample['subject_index'])):
            subject_index = sample['subject_index'][j].item()
            extracted_sample = data_source.direct_extract(direct_extractor, subject_index)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Tensorflow data access verification')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../dummy-data/dummy.h5',
        help='Path to the dataset file.'
    )

    args = parser.parse_args()
    main(args.hdf_file)
