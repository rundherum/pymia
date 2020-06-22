import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import pymia.data.definition as defs
import pymia.data.extraction as extr
import pymia.data.assembler as assm
import pymia.data.backends.tensorflow as pymia_tf


def main():
    hdf_file = '../example-data/example-dataset.h5'

    extractor = extr.ComposeExtractor(
        [extr.DataExtractor(categories=(defs.KEY_IMAGES,))]
    )

    transform = None

    indexing_strategy = extr.SliceIndexing()
    dataset = extr.PymiaDatasource(hdf_file, indexing_strategy, extractor, transform)

    direct_extractor = extr.ComposeExtractor(
        [extr.ImagePropertiesExtractor(),
         extr.DataExtractor(categories=(defs.KEY_LABELS, defs.KEY_IMAGES))]
    )
    assembler = assm.SubjectAssembler(dataset)

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

        numpy_prediction = prediction.numpy()

        is_last = i == nb_batches - 1
        assembler.add_batch(numpy_prediction, sample_indices.numpy(), is_last)

        for subject_index in assembler.subjects_ready:
            subject_prediction = assembler.get_assembled_subject(subject_index)

            direct_sample = dataset.direct_extract(direct_extractor, subject_index)

            # do_eval(subject_prediction, direct_sample[defs.KEY_LABELS])


if __name__ == '__main__':
    main()
