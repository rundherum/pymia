import torch.utils.data.dataset as torch_data

import pymia.data.extraction as extr
import pymia.data.transformation as tfm


# the PymiaDataSource already satisfies the torch.dataset interface
class PymiaTorchDataset(extr.PymiaDatasource, torch_data.Dataset):
    def __init__(self, dataset_path: str,
                 indexing_strategy: extr.IndexingStrategy = None,
                 extractor: extr.Extractor = None,
                 transform: tfm.Transform = None, subject_subset: list = None, init_reader_once=True) -> None:
        # todo(fabian): verify
        # dirty hack but I did not find any other solution since super().__init__() is calling both parent inits and
        # raises an error if there are arguments. At the same time we would like the PymiaDataSource interface not be
        # changed much
        extr.PymiaDatasource.__init__(self, dataset_path, indexing_strategy, extractor, transform,
                                      subject_subset, init_reader_once)
