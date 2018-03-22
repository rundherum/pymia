import torch.utils.data.dataset as data

import miapy.data.transformation as tfm
import miapy.data.indexexpression as expr
from . import reader as rd
from . import indexing as idx
from . import extractor as extr


class ParameterizableDataset(data.Dataset):

    def __init__(self, dataset_path: str, indexing_strategy: idx.IndexingStrategy=None, extractor: extr.Extractor=None,
                 transform: tfm.Transform = None, init_reader_once=True) -> None:
        self.dataset_path = dataset_path
        if indexing_strategy is None:
            indexing_strategy = idx.EmptyIndexing()
        self.indexing_strategy = indexing_strategy
        self.extractor = extractor
        self.transform = transform
        self.init_reader_once=init_reader_once

        self.indices = []
        self.reader = None
        with rd.get_reader(dataset_path) as reader:
            for i, subject in enumerate(reader.get_subject_entries()):
                subject_indices = self.indexing_strategy(reader.get_shape(subject))
                subject_and_indices = zip(len(subject_indices) * [i], subject_indices)
                self.indices.extend(subject_and_indices)

    def __del__(self):
        self.close_reader()

    def close_reader(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def set_extractor(self, extractor: extr.Extractor):
        self.extractor = extractor

    def direct_extract(self, extractor: extr.Extractor, subject_index: int, index_expr: expr.IndexExpression=None,
                       transform=None):
        if index_expr is None:
            index_expr = expr.IndexExpression()

        params = {'subject_index': subject_index, 'index_expr': index_expr}
        extracted = {}

        if not self.init_reader_once:
            with rd.get_reader(self.dataset_path) as reader:
                extractor.extract(reader, params, extracted)
        else:
            if self.reader is None:
                self.reader = rd.get_reader(self.dataset_path, direct_open=True)
            extractor.extract(self.reader, params, extracted)

        if transform:
            extracted = transform(extracted)

        return extracted

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        subject_index, index_expr = self.indices[item]
        return self.direct_extract(self.extractor, subject_index, index_expr, self.transform)
