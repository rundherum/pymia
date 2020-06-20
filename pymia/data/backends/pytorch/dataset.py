import torch.utils.data.dataset as torch_data

import pymia.data.extraction as extr
import pymia.data.transformation as tfm


class PymiaTorchDataset(extr.PymiaDatasource, torch_data.Dataset):

    def __init__(self, dataset_path: str,
                 indexing_strategy: extr.IndexingStrategy = None,
                 extractor: extr.Extractor = None,
                 transform: tfm.Transform = None, subject_subset: list = None, init_reader_once=True) -> None:
        """Represents a `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_
        wrapper class for :class:`.PymiaDatasource`.

        See :class:`.PymiaDatasource` for the description of the arguments."""

        # calling parent __init__() directly since super().__init__() is calling both parent inits and
        # raises an error if there are arguments.
        extr.PymiaDatasource.__init__(self, dataset_path, indexing_strategy, extractor, transform,
                                      subject_subset, init_reader_once)


class PytorchDatasetAdapter(torch_data.Dataset):

    def __init__(self, datasource: extr.PymiaDatasource) -> None:
        """A wrapper class for :class:`.PymiaDatasource` to fit the `
        torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_ interface.

        Args:
            datasource (.PymiaDatasource): The pymia datasource instance.
        """

        super().__init__()
        self.datasource = datasource

    def __len__(self) -> int:
        return len(self.datasource)

    def __getitem__(self, index: int):
        return self.datasource[index]