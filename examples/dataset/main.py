import argparse

import miapy.data.extraction as miapy_extr

import examples.dataset.config as cfg


def train():
    pass


def test():
    pass


def main(config_file: str):
    config = cfg.load(config_file, cfg.Configuration)
    print(config)

    reader = miapy_extr.get_reader(config.database_file, direct_open=True)

    indexing_strategy = miapy_extr.SliceIndexing()
    extraction_transform = None
    train_extractor = miapy_extr.ComposeExtractor([miapy_extr.NamesExtractor(),
                                                   miapy_extr.SubjectExtractor(),
                                                   miapy_extr.IndexingExtractor(),
                                                   miapy_extr.ImageExtractor(),
                                                   miapy_extr.LabelExtractor(gt_mode='select',
                                                                             gt_selection=config.maps)])  # gt_selection=('B0map', 'PDmap', 'T1map', 'T2map')

    dataset = miapy_extr.ParameterizableDataset(reader, indexing_strategy, transform=extraction_transform)
    dataset.set_extractor(train_extractor)


    training_sampler = miapy_extr.SubsetRandomSampler(sampler_ids_train)
    training_loader = miapy_extr.DataLoader(dataset, config.batch_size, sampler=training_sampler,
                                            collate_fn=ext.collate_batch, num_workers=1)

    testing_sampler = miapy_extr.SubsetSequentialSampler(sampler_ids_test)
    testing_loader = miapy_extr.DataLoader(dataset, 200, sampler=testing_sampler,
                                           collate_fn=ext.collate_batch, num_workers=1)

    for epoch in range(1):  # epochs loop
        for batch in training_loader:
            feed_dict = mdl_general.batch_to_feed_dict2(x, y, batch, dtype, phase_train, True, y_mask)
            # train model

        # subject assembler
        for batch in testing_loader:
            feed_dict = mdl_general.batch_to_feed_dict2(x, y, batch, dtype, phase_train, False, y_mask)
            # test model


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
