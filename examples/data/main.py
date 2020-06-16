import argparse

import numpy as np
import torch.utils.data as torch_data

import pymia.data.conversion as pymia_conv
import pymia.data.definition as pymia_defs
import pymia.data.extraction as pymia_extr
import pymia.data.assembler as pymia_asmbl
import pymia.data.backends.pytorch as pymia_torch
import pymia.evaluation.evaluator as pymia_eval
import pymia.evaluation.metric as pymia_metric
import pymia.evaluation.writer as writer


def init_evaluator() -> pymia_eval.Evaluator:
    labels = {
        1: 'WhiteMatter',
        2: 'GreyMatter',
        3: 'Hippocampus',
        4: 'Amygdala',
        5: 'Thalamus'
    }
    metrics = [pymia_metric.DiceCoefficient()]
    return pymia_eval.Evaluator(metrics, labels)


def main(hdf_file: str):
    epochs = 20
    batch_size_training = 10
    batch_size_testing = 10

    indexing_strategy = pymia_extr.SliceIndexing()  # slice-wise extraction

    # define an extractor for training, i.e. what information we would like to extract per sample
    train_extractor = pymia_extr.ComposeExtractor([pymia_extr.NamesExtractor(),
                                                   pymia_extr.DataExtractor(),
                                                   pymia_extr.SelectiveDataExtractor()])

    # define an extractor for testing, i.e. what information we would like to extract per sample
    # not that usually we don't use labels for testing, i.e. the SelectiveDataExtractor is only used for this example
    test_extractor = pymia_extr.ComposeExtractor([pymia_extr.NamesExtractor(),
                                                  pymia_extr.IndexingExtractor(do_pickle=True),
                                                  pymia_extr.DataExtractor(),
                                                  pymia_extr.SelectiveDataExtractor(),
                                                  pymia_extr.ImageShapeExtractor()])

    # define an extractor for evaluation, i.e. what information we would like to extract per sample
    eval_extractor = pymia_extr.ComposeExtractor([pymia_extr.NamesExtractor(),
                                                  pymia_extr.SubjectExtractor(),
                                                  pymia_extr.SelectiveDataExtractor(),
                                                  pymia_extr.ImagePropertiesExtractor()])

    # define the data set
    dataset = pymia_torch.PymiaTorchDataset(hdf_file,
                                            indexing_strategy,
                                            pymia_extr.SubjectExtractor())  # for select_indices() below

    # generate train / test split for data set
    # we use Subject_0, Subject_1 and Subject_2 for training and Subject_3 for testing
    sampler_ids_train = pymia_extr.select_indices(dataset,
                                                  pymia_extr.SubjectSelection(('Subject_1', 'Subject_2', 'Subject_3')))
    sampler_ids_test = pymia_extr.select_indices(dataset,
                                                 pymia_extr.SubjectSelection(('Subject_4',)))

    # set up training data loader
    training_sampler = torch_data.SubsetRandomSampler(sampler_ids_train)
    training_loader = torch_data.DataLoader(dataset, batch_size_training, sampler=training_sampler, num_workers=1)

    # set up testing data loader
    testing_sampler = pymia_torch.SubsetSequentialSampler(sampler_ids_test)
    testing_loader = torch_data.DataLoader(dataset, batch_size_testing, sampler=testing_sampler, num_workers=1)

    sample = dataset.direct_extract(train_extractor, 0)  # extract a subject

    evaluator = init_evaluator()  # initialize evaluator

    for epoch in range(epochs):  # epochs loop
        dataset.set_extractor(train_extractor)
        for batch in training_loader:  # batches for training
            # feed_dict = batch_to_feed_dict(x, y, batch, True)  # e.g. for TensorFlow
            # train model, e.g.:
            # sess.run([train_op, loss], feed_dict=feed_dict)
            pass

        # subject assembler for testing
        subject_assembler = pymia_asmbl.SubjectAssembler()

        dataset.set_extractor(test_extractor)
        for batch in testing_loader:  # batches for testing
            # feed_dict = batch_to_feed_dict(x, y, batch, False)  # e.g. for TensorFlow
            # test model, e.g.:
            # prediction = sess.run(y_model, feed_dict=feed_dict)
            prediction = np.stack(batch[pymia_defs.KEY_LABELS], axis=0)  # we use the labels as predictions such that we can validate the assembler
            subject_assembler.add_batch(prediction, batch)

        # evaluate all test images
        for subject_idx in list(subject_assembler.predictions.keys()):
            # convert prediction and labels back to SimpleITK images
            sample = dataset.direct_extract(eval_extractor, subject_idx)
            label_image = pymia_conv.NumpySimpleITKImageBridge.convert(sample[pymia_defs.KEY_LABELS],
                                                                       sample[pymia_defs.KEY_PROPERTIES])

            assembled = subject_assembler.get_assembled_subject(sample[pymia_defs.KEY_SUBJECT_INDEX])
            prediction_image = pymia_conv.NumpySimpleITKImageBridge.convert(assembled,
                                                                            sample[pymia_defs.KEY_PROPERTIES])
            evaluator.evaluate(prediction_image, label_image, sample[pymia_defs.KEY_SUBJECT])  # evaluate prediction


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Training loop scaffold using pymia data handling.')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../example-data/dummy.h5',
        help='Path to the dataset file.'
    )

    args = parser.parse_args()
    main(args.hdf_file)
