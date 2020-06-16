import argparse

import tensorflow as tf

import pymia.data.definition as defs
import pymia.data.extraction as extr
import pymia.data.backends.tensorflow as pymia_tf


def main(hdf_file: str):
    # todo(alain): add other extractors?
    extractor = extr.ComposeExtractor([extr.DataExtractor(),
                                       extr.SubjectExtractor()])

    direct_extractor = extr.ComposeExtractor([extr.SubjectExtractor(),
                                              extr.FilesExtractor(categories=(defs.KEY_IMAGES,
                                                                              defs.KEY_LABELS,
                                                                              'mask', 'numerical', 'sex')),
                                              extr.ImagePropertiesExtractor()])

    data_source = extr.dataset.PymiaDatasource(hdf_file, extr.SliceIndexing(), extractor)

    gen_fn = pymia_tf.get_tf_generator(data_source)
    dataset = tf.data.Dataset.from_generator(generator=gen_fn,
                                             output_types={defs.KEY_IMAGES: tf.float32,
                                                           defs.KEY_SUBJECT_INDEX: tf.int64,
                                                           defs.KEY_SUBJECT: tf.string})

    dataset = dataset.batch(2)

    for i, sample in enumerate(dataset.as_numpy_iterator()):

        for j in range(len(sample[defs.KEY_SUBJECT_INDEX])):
            subject_index = sample[defs.KEY_SUBJECT_INDEX][j].item()
            extracted_sample = data_source.direct_extract(direct_extractor, subject_index)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Tensorflow data access verification')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../example-data/dummy.h5',
        help='Path to the dataset file.'
    )

    args = parser.parse_args()
    main(args.hdf_file)
