import argparse

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import pymia.data.definition as defs
import pymia.data.extraction as extr
import pymia.data.assembler as assm
import pymia.data.transformation as tfm
import pymia.data.backends.tensorflow as pymia_tf


def main(hdf_file: str):
    hdf_file = '../example-data/example-data.h5'
    # train_subjects, valid_subjects = ['Subject_1', 'Subject_2', 'Subject_3'], ['Subject_4']

    extractor = extr.ComposeExtractor(
        [extr.DataExtractor(categories=(defs.KEY_IMAGES,))]
    )

    # transform = tfm.Permute(permutation=(2, 0, 1), entries=(defs.KEY_IMAGES,))
    transform = None

    indexing_strategy = extr.SliceIndexing()
    # indexing_strategy = extr.PatchWiseIndexing((20, 20, 20))
    dataset = extr.PymiaDatasource(hdf_file, indexing_strategy, extractor, transform)

    direct_extractor = extr.ComposeExtractor(
        [extr.ImagePropertiesExtractor(),
         extr.DataExtractor(categories=(defs.KEY_LABELS, defs.KEY_IMAGES))]
    )
    assembler = assm.SubjectAssemblerNew(dataset)

    gen_fn = pymia_tf.get_tf_generator(dataset)
    tf_dataset = tf.data.Dataset.from_generator(generator=gen_fn,
                                             output_types={defs.KEY_IMAGES: tf.float32,
                                                           defs.KEY_SAMPLE_INDEX: tf.int64})
    tf_dataset = tf_dataset.batch(2)

    dummy_network = keras.Sequential([
        layers.Conv2D(8, kernel_size=3, padding='same'),
        layers.Conv2D(2, kernel_size=3, padding='same', activation='sigmoid')]
    )

    nb_batches = len(dataset) // 2

    for i, batch in enumerate(tf_dataset):
        x, sample_indices = batch[defs.KEY_IMAGES], batch[defs.KEY_SAMPLE_INDEX]

        prediction = dummy_network(x)

        is_last = i == nb_batches - 1

        numpy_prediction = prediction.numpy()
        image = x.numpy()

        to_assemble = {'image': image, 'prediction': numpy_prediction}
        assembler.add_batch(to_assemble, sample_indices.numpy(), is_last)

        are_ready = assembler.subjects_ready
        if len(are_ready) == 0:
            continue

        for subject_index in assembler.subjects_ready:
            subject_prediction = assembler.get_assembled_subject(subject_index)['image']

            direct_sample = dataset.direct_extract(direct_extractor, subject_index)

            orig_image = direct_sample[defs.KEY_IMAGES]
            a = (subject_prediction == orig_image).all()

            # do_eval(subject_prediction, direct_sample[defs.KEY_LABELS])


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
