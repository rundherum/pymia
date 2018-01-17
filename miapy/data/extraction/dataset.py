import torch.utils.data.dataset as data

import miapy.data.transformation as tfm
import miapy.data.indexexpression as expr
from . import reader as rd
from . import indexing as idx
from . import extractor as extr


class ParametrizableDataset(data.Dataset):

    def __init__(self, reader: rd.Reader, indexing_strategy: idx.IndexingStrategy, extractor: extr.Extractor=None,
                 transform: tfm.Transform = None) -> None:
        self.reader = reader
        self.indexing_strategy = indexing_strategy
        self.extractor = extractor
        self.transform = transform

        # todo: allow indices as argument with mutual exclusivity with strategy
        self.indices = []
        for i, subject in enumerate(reader.get_subject_entries()):
            subject_indices = self.indexing_strategy(self.reader.get_shape(subject))
            subject_and_indices = zip(len(subject_indices) * [i], subject_indices)
            self.indices.extend(subject_and_indices)

    def set_extractor(self, extractor: extr.Extractor):
        self.extractor = extractor

    def direct_extract(self, extractor: extr.Extractor, subject_index: int, index_expr: expr.IndexExpression=None, transform=None):
        if index_expr is None:
            index_expr = expr.IndexExpression()

        params = {'subject_index': subject_index, 'index_expr': index_expr}
        extracted = {}
        extractor.extract(self.reader, params, extracted)

        if transform:
            extracted = transform(extracted)

        return extracted

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        subject_index, index_expr = self.indices[item]
        return self.direct_extract(self.extractor, subject_index, index_expr, self.transform)
