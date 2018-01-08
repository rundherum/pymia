import typing as t

import torch.utils.data.dataset as data

import data.transformation as tfm
import data.extraction.reader as r
import data.extraction.indexing as idx
import data.extraction.extractor as extr


class ParametrizableDataset(data.Dataset):

    def __init__(self, reader: r.Reader, indexing_strategy: idx.IndexingStrategy, extractors: t.List[extr.Extractor],
                 transform: tfm.Transform = None) -> None:
        self.reader = reader
        self.extractors = extractors
        self.indexing_strategy = indexing_strategy
        self.transform = transform

        for extractor in self.extractors:
            extractor.set_reader(self.reader)

        self.indices = []
        self.subject_index_to_entry = {}
        for i, subject_entry in enumerate(self.reader.get_subject_entries()):
            self.subject_index_to_entry[i] = subject_entry
            subject_indices = self.indexing_strategy(self.reader.read(subject_entry).shape)
            subject_and_indices = zip(len(subject_indices) * [i], subject_indices)
            self.indices.extend(subject_and_indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        subject_index, index_expr = self.indices[item]
        params = {'subject_index': subject_index, 'index_expr': index_expr}
        extracted = {}
        for extractor in self.extractors:
            extractor.extract(params, extracted)

        if self.transform:
            extracted = self.transform(extracted)

        return extracted
