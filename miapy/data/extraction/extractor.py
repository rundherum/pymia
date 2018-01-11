import abc

import numpy as np

import miapy.data.indexexpression as util
from . import reader as rd


class Extractor(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        pass


class ComposeExtractor(Extractor):

    def __init__(self, extractors) -> None:
        super().__init__()
        self.extractors = extractors

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        for e in self.extractors:
            e.extract(reader, params, extracted)


class NamesExtractor(Extractor):

    def __init__(self, cache=True) -> None:
        super().__init__()
        self.cache = cache
        self.cached_result = None

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        if not self.cache or self.cached_result is None:
            d = self._extract(reader)
            self.cached_result = d
        else:
            d = self.cached_result

        for k, v in d.items():
            extracted[k] = v

    @staticmethod
    def _extract(reader: rd.Reader):
        sequence_names = reader.read('meta/sequence_names')
        d = {'sequence_names': sequence_names}
        if reader.has('meta/gt_names'):
            d['gt_names'] = reader.read('meta/gt_names')
        return d


class SubjectExtractor(Extractor):

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        subject_index_expr = util.IndexExpression(params['subject_index'])
        extracted['subject'] = reader.read('meta/subjects', subject_index_expr)


class IndexingExtractor(Extractor):

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        extracted['index_expr'] = params['index_expr']


class FilesExtractor(Extractor):

    def __init__(self, cache=True) -> None:
        super().__init__()
        self.cache = cache
        self.cached_file_root = None

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        subject_index_expr = util.IndexExpression(params['subject_index'])

        if not self.cache or self.cached_file_root is None:
            file_root = reader.read('meta/file_root')
            self.cached_file_root = file_root
        else:
            file_root = self.cached_file_root

        extracted['file_root'] = file_root
        extracted['sequence_files'] = reader.read('meta/sequence_files', subject_index_expr)
        if reader.has('meta/gt_files'):
            extracted['gt_files'] = reader.read('meta/gt_files', subject_index_expr)


class DataExtractor(Extractor):

    def __init__(self, gt_mode='all', gt_selection=None) -> None:
        super().__init__()
        self.entry_base_names = None
        if gt_mode not in ('all', 'random', 'select'):
            raise ValueError('gt_mode must be "all", "random" or "select"')
        self.gt_mode = gt_mode
        if gt_mode == 'select' and isinstance(gt_selection, (tuple, list)):
            raise ValueError('for mode "select" the selection can only be one selection (string)')
        if isinstance(gt_selection, str):
            gt_selection = (gt_selection,)
        self.gt_selection = gt_selection

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        if self.entry_base_names is None:
            entries = reader.get_subject_entries()
            self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

        subject_index = params['subject_index']
        index_expr = params['index_expr']

        base_name = self.entry_base_names[subject_index]
        extracted['images'] = reader.read('data/sequences/{}'.format(base_name), index_expr)
        if reader.has('data/gts'):
            np_gts = reader.read('data/gts/{}'.format(base_name), index_expr)

            if self.gt_mode != 'all':
                if 'gt_names' not in extracted:
                    raise ValueError('selection of gt requires gt_names to be extracted')
                gt_names = extracted['gt_names']  # type: list
                if self.gt_mode == 'random':
                    selection_indices = [gt_names.index(s) for s in self.gt_selection]
                    index = np.random.choice(selection_indices)
                else:
                    # mode == 'select'
                    index = gt_names.index(self.gt_selection[0])

                # todo: inverse index_expr in order to add index to it
                np_gts = np_gts[..., index]
                # maintaining gt dims
                np_gts = np.expand_dims(np_gts, -1)

            extracted['labels'] = np_gts

