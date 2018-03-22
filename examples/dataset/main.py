import argparse

import numpy as np
import tensorflow as tf

import miapy.data.conversion as miapy_conv
import miapy.data.extraction as miapy_extr
import miapy.data.assembler as miapy_asmbl
import miapy.evaluation.evaluator as miapy_eval
import miapy.evaluation.metric as miapy_metric

import examples.dataset.config as cfg


def batch_to_feed_dict(x_placeholder, y_placeholder, batch, is_train: bool=True) -> dict:
    feed_dict = {x_placeholder: np.stack(batch['images'], axis=0).astype(np.float32)}
    if is_train:
        feed_dict[y_placeholder] = np.stack(batch['labels'], axis=0).astype(np.int16)

    return feed_dict


def collate_batch(batch) -> dict:
    # batch is a list of dicts -> change to dict of lists
    return dict(zip(batch[0], zip(*[d.values() for d in batch])))


def init_evaluator() -> miapy_eval.Evaluator:
    evaluator = miapy_eval.Evaluator(miapy_eval.ConsoleEvaluatorWriter(5))
    evaluator.add_label(1, 'Structure 1')
    evaluator.add_label(2, 'Structure 2')
    evaluator.add_label(3, 'Structure 3')
    evaluator.add_label(4, 'Structure 4')
    evaluator.metrics = [miapy_metric.DiceCoefficient()]
    return evaluator


def main(config_file: str):
    config = cfg.load(config_file, cfg.Configuration)
    print(config)

    indexing_strategy = miapy_extr.SliceIndexing()  # slice-wise extraction
    extraction_transform = None  # we do not want to apply any transformation on the slices after extraction
    # define an extractor for training, i.e. what information we would like to extract per sample
    train_extractor = miapy_extr.ComposeExtractor([miapy_extr.ImageExtractor(),
                                                   miapy_extr.LabelExtractor()])

    # define an extractor for testing, i.e. what information we would like to extract per sample
    test_extractor = miapy_extr.ComposeExtractor([miapy_extr.IndexingExtractor(),
                                                  miapy_extr.ImageExtractor(),
                                                  miapy_extr.LabelExtractor(),
                                                  miapy_extr.ImageShapeExtractor()])

    # define an extractor for evaluation, i.e. what information we would like to extract per sample
    eval_extractor = miapy_extr.ComposeExtractor([miapy_extr.SubjectExtractor(),
                                                  miapy_extr.LabelExtractor(),
                                                  miapy_extr.ImagePropertiesExtractor()])

    # define the data set
    dataset = miapy_extr.ParameterizableDataset(config.database_file,
                                                indexing_strategy,
                                                miapy_extr.SubjectExtractor(),  # for select_indices() below
                                                extraction_transform)

    # generate train / test split for data set
    # we use Subject_0, Subject_1 and Subject_2 for training and Subject_3 for testing
    sampler_ids_train = miapy_extr.select_indices(dataset,
                                                  miapy_extr.SubjectSelection(('Subject_0', 'Subject_1', 'Subject_2')))
    sampler_ids_test = miapy_extr.select_indices(dataset,
                                                 miapy_extr.SubjectSelection(('Subject_3')))

    # set up training data loader
    training_sampler = miapy_extr.SubsetRandomSampler(sampler_ids_train)
    training_loader = miapy_extr.DataLoader(dataset, config.batch_size_training, sampler=training_sampler,
                                            collate_fn=collate_batch, num_workers=1)

    # set up testing data loader
    testing_sampler = miapy_extr.SubsetSequentialSampler(sampler_ids_test)
    testing_loader = miapy_extr.DataLoader(dataset, config.batch_size_testing, sampler=testing_sampler,
                                           collate_fn=collate_batch, num_workers=1)

    # define TensorFlow placeholders
    sample = dataset.direct_extract(train_extractor, 0)  # extract a subject to obtain shape
    x = tf.placeholder(tf.float32, (None, ) + sample['images'].shape[1:])
    y = tf.placeholder(tf.int16, (None, ) + sample['labels'].shape[1:] + (1, ))

    evaluator = init_evaluator()  # initialize evaluator

    for epoch in range(config.epochs):  # epochs loop
        dataset.set_extractor(train_extractor)
        for batch in training_loader:  # batches for training
            feed_dict = batch_to_feed_dict(x, y, batch, True)
            # train model, e.g.:
            # sess.run([train_op, loss], feed_dict=feed_dict)

        # subject assembler for testing
        subject_assembler = miapy_asmbl.SubjectAssembler()

        dataset.set_extractor(test_extractor)
        for batch in testing_loader:  # batches for testing
            feed_dict = batch_to_feed_dict(x, y, batch, False)
            # test model, e.g.:
            # prediction = sess.run(y_model, feed_dict=feed_dict)
            prediction = np.stack(batch['labels'], axis=0)  # we use the labels as predictions such that we can validate the assembler
            subject_assembler.add_batch(prediction, batch)

        # evaluate all test images
        for subject_idx in list(subject_assembler.predictions.keys()):
            # convert prediction and labels back to SimpleITK images
            sample = dataset.direct_extract(eval_extractor, subject_idx)
            label_image = miapy_conv.NumpySimpleITKImageBridge.convert(sample['labels'],
                                                                       sample['image_properties'])

            assembled = subject_assembler.get_assembled_subject(sample['subject_index'])
            prediction_image = miapy_conv.NumpySimpleITKImageBridge.convert(assembled, sample['image_properties'])
            evaluator.evaluate(prediction_image, label_image, sample['subject'])  # evaluate prediction


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Deep learning for peripheral nerve segmentation')

    parser.add_argument(
        '--config_file',
        type=str,
        default='config.json',
        help='Path to the configuration file.'
    )

    args = parser.parse_args()
    main(args.config_file)
