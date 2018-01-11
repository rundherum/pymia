import torch.utils.data.dataset as data

import miapy.data.transformation as tfm
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
        self.subject_index_to_entry = {}
        for i, subject_entry in enumerate(self.reader.get_subject_entries()):
            self.subject_index_to_entry[i] = subject_entry
            subject_indices = self.indexing_strategy(self.reader.read(subject_entry).shape)
            subject_and_indices = zip(len(subject_indices) * [i], subject_indices)
            self.indices.extend(subject_and_indices)

    def set_extractor(self, extractor: extr.Extractor):
        self.extractor = extractor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        subject_index, index_expr = self.indices[item]
        params = {'subject_index': subject_index, 'index_expr': index_expr}
        extracted = {}
        self.extractor.extract(self.reader, params, extracted)

        if self.transform:
            extracted = self.transform(extracted)

        return extracted
