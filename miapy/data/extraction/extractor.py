import abc
import pickle

import numpy as np
import SimpleITK as sitk

import miapy.data.conversion as conv
import miapy.data.definition as df
import miapy.data.indexexpression as expr
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
        image_names = reader.read(df.NAMES_IMAGE)
        d = {'image_names': image_names}
        if reader.has(df.NAMES_LABEL):
            d['label_names'] = reader.read(df.NAMES_LABEL)
        if reader.has(df.NAMES_SUPPL):
            d['supplementary_names'] = reader.read(df.NAMES_SUPPL)
        return d


class SubjectExtractor(Extractor):

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        extracted['subject_index'] = params['subject_index']
        subject_index_expr = expr.IndexExpression(params['subject_index'])
        extracted['subject'] = reader.read(df.SUBJECT, subject_index_expr)


class IndexingExtractor(Extractor):

    def __init__(self, do_pickle=False) -> None:
        super().__init__()
        self.do_pickle = do_pickle

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        extracted['subject_index'] = params['subject_index']
        index_expression = params['index_expr']
        if self.do_pickle:
            # pickle to prevent from problems since own class
            index_expression = pickle.dumps(index_expression)
        extracted['index_expr'] = index_expression


class ImagePropertiesExtractor(Extractor):
    """Extracts the image properties."""

    def __init__(self, do_pickle=False) -> None:
        super().__init__()
        self.do_pickle = do_pickle

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        subject_index_expr = expr.IndexExpression(params['subject_index'])

        shape = reader.read(df.INFO_SHAPE, subject_index_expr).tolist()
        direction = reader.read(df.INFO_DIRECTION, subject_index_expr).tolist()
        spacing = reader.read(df.INFO_SPACING, subject_index_expr).tolist()
        origin = reader.read(df.INFO_ORIGIN, subject_index_expr).tolist()

        # todo: everything in memory?
        image = sitk.Image(shape, sitk.sitkUInt8)
        image.SetDirection(direction)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        # todo number_of_components_per_pixel and pixel_id

        img_properties = conv.ImageProperties(image)
        if self.do_pickle:
            # pickle to prevent from problems since own class
            img_properties = pickle.dumps(img_properties)
        extracted['image_properties'] = img_properties


class FilesExtractor(Extractor):

    def __init__(self, cache=True) -> None:
        super().__init__()
        self.cache = cache
        self.cached_file_root = None

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        subject_index_expr = expr.IndexExpression(params['subject_index'])

        if not self.cache or self.cached_file_root is None:
            file_root = reader.read(df.FILES_ROOT)
            self.cached_file_root = file_root
        else:
            file_root = self.cached_file_root

        extracted['file_root'] = file_root
        extracted['image_files'] = reader.read(df.FILES_IMAGE, subject_index_expr)
        if reader.has(df.FILES_LABEL):
            extracted['label_files'] = reader.read(df.FILES_LABEL, subject_index_expr)
        if reader.has(df.FILES_SUPPL):
            extracted['supplementary_files'] = reader.read(df.FILES_SUPPL, subject_index_expr)


class ImageExtractor(Extractor):

    def __init__(self, entire_subject=False) -> None:
        super().__init__()
        self.entry_base_names = None
        self.entire_subject = entire_subject

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        if self.entry_base_names is None:
            entries = reader.get_subject_entries()
            self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

        subject_index = params['subject_index']
        index_expr = params['index_expr']

        base_name = self.entry_base_names[subject_index]
        if self.entire_subject:
            np_image = reader.read('{}/{}'.format(df.DATA_IMAGE, base_name))
        else:
            np_image = reader.read('{}/{}'.format(df.DATA_IMAGE, base_name), index_expr)
        extracted['images'] = np_image


class LabelExtractor(Extractor):

    def __init__(self, gt_mode='all', gt_selection=None, entire_subject=False) -> None:
        super().__init__()
        self.entry_base_names = None
        if gt_mode not in ('all', 'random', 'select'):
            raise ValueError('gt_mode must be "all", "random", or "select"')

        self.gt_mode = gt_mode
        if isinstance(gt_selection, str):
            gt_selection = (gt_selection,)
        self.gt_selection = gt_selection
        self.entire_subject = entire_subject

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        if self.entry_base_names is None:
            entries = reader.get_subject_entries()
            self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

        if not reader.has(df.DATA_LABEL):
            raise ValueError('FullSubjectExtractor requires GT to exist')

        subject_index = params['subject_index']
        index_expr = params['index_expr']

        base_name = self.entry_base_names[subject_index]

        if self.entire_subject:
            np_gts = reader.read('{}/{}'.format(df.DATA_LABEL, base_name))
        else:
            np_gts = reader.read('{}/{}'.format(df.DATA_LABEL, base_name), index_expr)

        if self.gt_mode != 'all':
            if 'label_names' not in extracted:
                raise ValueError('selection of labels requires label_names to be extracted (use NamesExtractor)')
            label_names = extracted['label_names']  # type: list
            if self.gt_mode == 'random':
                selection_indices = [label_names.index(s) for s in self.gt_selection]
                index = np.random.choice(selection_indices)
            elif self.gt_mode == 'select' and isinstance(self.gt_selection, (tuple, list)):
                selection_indices = np.array([label_names.index(s) for s in self.gt_selection])
                extracted['labels'] = np.take(np_gts, selection_indices, axis=-1)
                return
            else:
                # mode == 'select'
                index = label_names.index(self.gt_selection[0])

            # todo: inverse index_expr in order to add index to it
            np_gts = np_gts[..., index]
            # maintaining gt dims
            np_gts = np.expand_dims(np_gts, -1)

        extracted['labels'] = np_gts


class LabelShapeExtractor(Extractor):

    def __init__(self, gt_mode='all', gt_selection=None) -> None:
        super().__init__()
        self.entry_base_names = None
        if gt_mode not in ('all', 'random', 'select'):
            raise ValueError('gt_mode must be "all", "random", or "select"')

        self.gt_mode = gt_mode
        if isinstance(gt_selection, str):
            gt_selection = (gt_selection,)
        self.gt_selection = gt_selection

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        if self.entry_base_names is None:
            entries = reader.get_subject_entries()
            self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

        if not reader.has(df.DATA_LABEL):
            raise ValueError('ShapeExtractor requires GT to exist')

        subject_index = params['subject_index']

        base_name = self.entry_base_names[subject_index]

        np_gts = reader.read('{}/{}'.format(df.DATA_LABEL, base_name))

        if self.gt_mode != 'all':
            if 'label_names' not in extracted:
                raise ValueError('selection of labels requires label_names to be extracted (use NamesExtractor)')
            label_names = extracted['label_names']  # type: list
            if self.gt_mode == 'random':
                selection_indices = [label_names.index(s) for s in self.gt_selection]
                index = np.random.choice(selection_indices)
            elif self.gt_mode == 'select' and isinstance(self.gt_selection, (tuple, list)):
                selection_indices = np.array([label_names.index(s) for s in self.gt_selection])
                extracted['shape'] = np.take(np_gts, selection_indices, axis=-1).shape
                return
            else:
                # mode == 'select'
                index = label_names.index(self.gt_selection[0])

            # todo: inverse index_expr in order to add index to it
            np_gts = np_gts[..., index]
            # maintaining gt dims
            np_gts = np.expand_dims(np_gts, -1)

        extracted['shape'] = np_gts.shape


class SupplementaryExtractor(Extractor):

    def __init__(self, entries: tuple, entire_subject=False) -> None:
        super().__init__()
        self.entries = entries
        self.entire_subject = entire_subject
        self.entry_base_names = None

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        if self.entry_base_names is None:
            entries = reader.get_subject_entries()
            self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

        subject_index = params['subject_index']
        index_expr = params['index_expr']

        base_name = self.entry_base_names[subject_index]
        for entry in self.entries:
            if self.entire_subject:
                extracted[entry] = reader.read('{}/{}/{}'.format(df.DATA, entry, base_name))
            else:
                extracted[entry] = reader.read('{}/{}/{}'.format(df.DATA, entry, base_name), index_expr)
