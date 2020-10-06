import argparse
import os
import sys
sys.path.append('./helper')

import numpy as np
import tensorflow as tf

import pymia.data.assembler as assm
import pymia.data.transformation as tfm
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer
import pymia.data.definition as defs
import pymia.data.extraction as extr
import pymia.data.backends.tensorflow as pymia_tf

# import helper.unet_keras as unet
import helper.unet_tf as unet

def main(hdf_file, log_dir):
    # initialize the evaluator with the metrics and the labels to evaluate
    metrics = [metric.DiceCoefficient()]
    labels = {1: 'WHITEMATTER',
              2: 'GREYMATTER',
              3: 'HIPPOCAMPUS',
              4: 'AMYGDALA',
              5: 'THALAMUS'}
    evaluator = eval_.SegmentationEvaluator(metrics, labels)

    # we want to log the mean and standard deviation of the metrics among all subjects of the dataset
    functions = {'MEAN': np.mean, 'STD': np.std}
    statistics_aggregator = writer.StatisticsAggregator(functions=functions)
    console_writer = writer.ConsoleStatisticsWriter(functions=functions)

    # initialize TensorBoard writer
    summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'logging-example-tensorflow'))

    # setup the training datasource
    train_subjects, valid_subjects = ['Subject_1', 'Subject_2', 'Subject_3'], ['Subject_4']
    extractor = extr.DataExtractor(categories=(defs.KEY_IMAGES, defs.KEY_LABELS))
    indexing_strategy = extr.SliceIndexing()

    # augmentation_transforms = [augm.RandomRotation90(), augm.RandomMirror()]
    augmentation_transforms = []
    transforms = [tfm.Squeeze(entries=(defs.KEY_LABELS,))]
    train_transforms = tfm.ComposeTransform(transforms + augmentation_transforms)
    train_dataset = extr.PymiaDatasource(hdf_file, indexing_strategy, extractor, train_transforms,
                                         subject_subset=train_subjects)

    # setup the validation datasource
    batch_size = 16
    valid_transforms = tfm.ComposeTransform([])
    valid_dataset = extr.PymiaDatasource(hdf_file, indexing_strategy, extractor, valid_transforms,
                                         subject_subset=valid_subjects)
    direct_extractor = extr.ComposeExtractor(
        [extr.SubjectExtractor(),
         extr.ImagePropertiesExtractor(),
         extr.DataExtractor(categories=(defs.KEY_LABELS,))]
    )
    assembler = assm.SubjectAssembler(valid_dataset)

    # torch specific handling
    train_gen_fn = pymia_tf.get_tf_generator(train_dataset)
    tf_train_dataset = tf.data.Dataset.from_generator(generator=train_gen_fn,
                                                      output_types={defs.KEY_IMAGES: tf.float32,
                                                                    defs.KEY_LABELS: tf.int64,
                                                                    defs.KEY_SAMPLE_INDEX: tf.int64})
    tf_train_dataset = tf_train_dataset.batch(batch_size).shuffle(len(train_dataset))


    valid_gen_fn = pymia_tf.get_tf_generator(valid_dataset)
    tf_valid_dataset = tf.data.Dataset.from_generator(generator=valid_gen_fn,
                                                      output_types={defs.KEY_IMAGES: tf.float32,
                                                                    defs.KEY_LABELS: tf.int64,
                                                                    defs.KEY_SAMPLE_INDEX: tf.int64})
    tf_valid_dataset = tf_valid_dataset.batch(batch_size)

    u_net = unet.build_model(channels=2, num_classes=6, layer_depth=3, filters_root=16)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

    train_batches = len(train_dataset) // batch_size

    # looping over the data in the dataset
    epochs = 100
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # training
        print('training')
        for i, batch in enumerate(tf_train_dataset):
            x, y = batch[defs.KEY_IMAGES], batch[defs.KEY_LABELS]

            with tf.GradientTape() as tape:
                logits = u_net(x, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)

            grads = tape.gradient(loss, u_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, u_net.trainable_variables))

            train_loss(loss)

            with summary_writer.as_default():
                tf.summary.scalar('train/loss', train_loss.result(), step=epoch*train_batches + i)
            print(f'[{i + 1}/{train_batches}]\tloss: {train_loss.result().numpy()}')

        # validation
        print('validation')
        valid_batches = len(valid_dataset) // batch_size
        for i, batch in enumerate(tf_valid_dataset):
            x, sample_indices = batch[defs.KEY_IMAGES], batch[defs.KEY_SAMPLE_INDEX]

            logits = u_net(x)
            prediction = tf.expand_dims(tf.math.argmax(logits, -1), -1)

            numpy_prediction = prediction.numpy()

            is_last = i == valid_batches - 1
            assembler.add_batch(numpy_prediction, sample_indices.numpy(), is_last)

            for subject_index in assembler.subjects_ready:
                subject_prediction = assembler.get_assembled_subject(subject_index)

                direct_sample = train_dataset.direct_extract(direct_extractor, subject_index)
                target, image_properties = direct_sample[defs.KEY_LABELS],  direct_sample[defs.KEY_PROPERTIES]

                # evaluate the prediction against the reference
                evaluator.evaluate(subject_prediction[..., 0], target[..., 0], direct_sample[defs.KEY_SUBJECT])

        # calculate mean and standard deviation of each metric
        results = statistics_aggregator.calculate(evaluator.results)
        # log to TensorBoard into category train
        with summary_writer.as_default():
            for result in results:
                tf.summary.scalar(f'valid/{result.metric}-{result.id_}', result.value, epoch)

        console_writer.write(evaluator.results)

        # clear results such that the evaluator is ready for the next evaluation
        evaluator.clear()


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

    parser.add_argument(
        '--log_dir',
        type=str,
        default='../example-data/log',
        help='Path to the logging directory.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.log_dir)