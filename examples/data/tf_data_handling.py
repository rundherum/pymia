import argparse

import tensorflow as tf

import pymia.data.definition as defs
import pymia.data.extraction as extr
import pymia.data.transformation as tfm
import pymia.data.assembler as assm
import pymia.data.backends.tensorflow as pymia_tf


def main(hdf_file: str):
    hdf_file = '../example-data/example-data.h5'
    # train_subjects, valid_subjects = ['Subject_1', 'Subject_2', 'Subject_3'], ['Subject_4']

    extractor = extr.ComposeExtractor(
        [extr.DataExtractor(categories=(defs.KEY_IMAGES,)),
         # extr.SubjectExtractor(),
         extr.IndexingExtractor(do_pickle=True),

         # extr.ImageShapeExtractor()
         ]
    )

    transform = tfm.Permute(permutation=(2, 1, 0), entries=(defs.KEY_IMAGES,))

    indexing_strategy = extr.SliceIndexing()
    dataset = extr.PymiaDatasource(hdf_file, indexing_strategy, extractor, transform)

    direct_extractor = extr.ComposeExtractor(
        [extr.ImagePropertiesExtractor(),
         extr.DataExtractor(categories=(defs.KEY_LABELS,))]
    )
    assembler = assm.SubjectAssembler()

    import numpy as np

    gen_fn = pymia_tf.get_tf_generator(dataset)
    tf_dataset = tf.data.Dataset.from_generator(generator=gen_fn,
                                             output_types={defs.KEY_IMAGES: tf.float32,
                                                           defs.KEY_SUBJECT_INDEX: tf.int64,
                                                           defs.KEY_INDEX_EXPR: tf.string})

    tf_dataset = tf_dataset.batch(2)

    for i, sample in enumerate(tf_dataset):

        for j in range(len(sample[defs.KEY_SUBJECT_INDEX])):
            subject_index = sample[defs.KEY_SUBJECT_INDEX][j].item()
            extracted_sample = dataset.direct_extract(direct_extractor, subject_index)


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
