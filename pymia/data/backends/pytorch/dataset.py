import torch.utils.data.dataset as torch_data

import pymia.data.extraction as extr


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