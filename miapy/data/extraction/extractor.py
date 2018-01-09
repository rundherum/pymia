import abc

import miapy.data.indexexpression as util
from . import reader as rd


class Extractor(metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        self.reader = None  # type: r.Reader

    def set_reader(self, reader: rd.Reader):
        self.reader = reader

    @abc.abstractmethod
    def extract(self, params: dict, extracted: dict) -> None:
        pass


class ComposeExtractor(Extractor):

    def __init__(self, extractors) -> None:
        super().__init__()
        self.extractors = extractors

    def set_reader(self, reader: rd.Reader):
        for e in self.extractors:
            e.set_reader(reader)

    def extract(self, params: dict, extracted: dict) -> None:
        for e in self.extractors:
            e.extract(params, extracted)


class NamesExtractor(Extractor):

    def __init__(self, cache=True) -> None:
        super().__init__()
        self.cache = cache
        self.cached_result = None

    def extract(self, params: dict, extracted: dict) -> None:
        if not self.cache or self.cached_result is None:
            d = self._extract()
            self.cached_result = d
        else:
            d = self.cached_result

        for k, v in d.items():
            extracted[k] = v

    def _extract(self):
        sequence_names = self.reader.read('meta/sequence_names')
        d = {'sequence_names': sequence_names}
        if self.reader.has('meta/gt_names'):
            d['gt_names'] = self.reader.read('meta/gt_names')
        return d


class SubjectExtractor(Extractor):

    def extract(self, params: dict, extracted: dict) -> None:
        subject_index_expr = util.IndexExpression(params['subject_index'])
        extracted['subject'] = self.reader.read('meta/subjects', subject_index_expr)


class FilesExtractor(Extractor):

    def __init__(self, cache=True) -> None:
        super().__init__()
        self.cache = cache
        self.cached_file_root = None

    def extract(self, params: dict, extracted: dict) -> None:
        subject_index_expr = util.IndexExpression(params['subject_index'])

        if not self.cache or self.cached_file_root is None:
            file_root = self.reader.read('meta/file_root')
            self.cached_file_root = file_root
        else:
            file_root = self.cached_file_root

        extracted['file_root'] = file_root
        extracted['sequence_files'] = self.reader.read('meta/sequence_files', subject_index_expr)
        if self.reader.has('meta/gt_files'):
            extracted['gt_files'] = self.reader.read('meta/gt_files', subject_index_expr)


class DataExtractor(Extractor):

    def __init__(self, sequence_selection=None, gt_selection=None) -> None:
        super().__init__()
        self.entry_base_names = None
        self.sequence_selection = sequence_selection
        self.gt_selection = gt_selection

    def set_reader(self, reader: rd.Reader):
        super().set_reader(reader)
        entries = self.reader.get_subject_entries()
        self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

    def extract(self, params: dict, extracted: dict) -> None:
        subject_index = params['subject_index']
        index_expr = params['index_expr']

        base_name = self.entry_base_names[subject_index]
        extracted['images'] = self.reader.read('data/sequences/{}'.format(base_name), index_expr)
        if self.reader.has('data/gts'):
            extracted['labels'] = self.reader.read('data/gts/{}'.format(base_name), index_expr)
