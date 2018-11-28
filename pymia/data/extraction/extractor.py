import abc
import pickle
import typing as t

import numpy as np
import SimpleITK as sitk

import pymia.data.conversion as conv
import pymia.data.definition as df
import pymia.data.indexexpression as expr
from . import reader as rd


class Extractor(metaclass=abc.ABCMeta):
    """Represents an extractor that extracts data from a dataset."""

    @abc.abstractmethod
    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        pass


class ComposeExtractor(Extractor):
    """Composes multiple Extractor objects."""

    def __init__(self, extractors) -> None:
        super().__init__()
        self.extractors = extractors

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        for e in self.extractors:
            e.extract(reader, params, extracted)


class NamesExtractor(Extractor):
    """Extracts the names of the entries within a category (i.e. the name of the enum identifying the data).

    The names are of type str.
    """

    def __init__(self, cache: bool=True, categories=('images', 'labels')) -> None:
        super().__init__()
        self.cache = cache
        self.cached_result = None
        self.categories = categories

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        if not self.cache or self.cached_result is None:
            d = self._extract(reader)
            self.cached_result = d
        else:
            d = self.cached_result

        for k, v in d.items():
            extracted[k] = v

    def _extract(self, reader: rd.Reader):
        d = {}
        for category in self.categories:
            d['{}_names'.format(category)] = reader.read(df.NAMES_PLACEHOLDER.format(category))
        return d


class SubjectExtractor(Extractor):
    """Extracts the subject's identification.

    The subject's identification is of type str.
    """

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        extracted['subject_index'] = params['subject_index']
        subject_index_expr = expr.IndexExpression(params['subject_index'])
        extracted['subject'] = reader.read(df.SUBJECT, subject_index_expr)


class IndexingExtractor(Extractor):
    """Extracts the index expression.

    The index expression is of type IndexExpression.
    """

    def __init__(self, do_pickle: bool=False) -> None:
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
    """Extracts the image properties.

    The image properties are of type ImageProperties.
    """

    def __init__(self, do_pickle: bool=False) -> None:
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
        extracted['properties'] = img_properties


class FilesExtractor(Extractor):
    """Extracts the file paths.

    The file paths are of type str.
    """

    def __init__(self, cache: bool=True, categories=('images', 'labels')) -> None:
        super().__init__()
        self.cache = cache
        self.cached_file_root = None
        self.categories = categories

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        subject_index_expr = expr.IndexExpression(params['subject_index'])

        if not self.cache or self.cached_file_root is None:
            file_root = reader.read(df.FILES_ROOT)
            self.cached_file_root = file_root
        else:
            file_root = self.cached_file_root

        extracted['file_root'] = file_root

        for category in self.categories:
            extracted['{}_files'.format(category)] = reader.read(df.FILES_PLACEHOLDER.format(category),
                                                                 subject_index_expr)


class SelectiveDataExtractor(Extractor):
    """Extracts data of a given category selectively."""

    def __init__(self, selection=None, category: str='labels') -> None:
        """Initializes a new instance of the SelectiveDataExtractor class.

        Args:
            selection (str or tuple): Entries within the category to select.
                If selection is None, the class has the same behaviour as the DataExtractor and selects all entries.
            category (str): The category to extract data from.
        """
        super().__init__()
        self.entry_base_names = None

        if isinstance(selection, str):
            selection = (selection,)
        self.selection = selection
        self.category = category

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        if '{}_names'.format(self.category) not in extracted:
            raise ValueError('selection of labels requires label_names to be extracted (use NamesExtractor)')

        if self.entry_base_names is None:
            entries = reader.get_subject_entries()
            self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

        if not reader.has(df.DATA_PLACEHOLDER.format(self.category)):
            raise ValueError('SelectiveDataExtractor requires {} to exist'.format(self.category))

        subject_index = params['subject_index']
        index_expr = params['index_expr']

        base_name = self.entry_base_names[subject_index]
        data = reader.read('{}/{}'.format(df.DATA_PLACEHOLDER.format(self.category), base_name), index_expr)
        label_names = extracted['{}_names'.format(self.category)]  # type: list

        if self.selection is None:
            extracted[self.category] = data
        else:
            selection_indices = np.array([label_names.index(s) for s in self.selection])
            extracted[self.category] = np.take(data, selection_indices, axis=-1)


class RandomDataExtractor(Extractor):
    """Extracts data of a given category randomly."""

    def __init__(self, selection=None, category: str='labels') -> None:
        """Initializes a new instance of the RandomDataExtractor class.

        Args:
            selection (str or tuple): Entries within the category to select an entry randomly from.
                If selection is None, an entry from all entries is randomly selected.
            selection (str or tuple): Note that if selection is None, all keys are considered.
            category (str): The category to extract data from.
        """
        super().__init__()
        self.entry_base_names = None

        if isinstance(selection, str):
            selection = (selection,)
        self.selection = selection
        self.category = category

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        if '{}_names'.format(self.category) not in extracted:
            raise ValueError('selection of labels requires label_names to be extracted (use NamesExtractor)')

        if self.entry_base_names is None:
            entries = reader.get_subject_entries()
            self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

        if not reader.has(df.DATA_PLACEHOLDER.format(self.category)):
            raise ValueError('SelectiveDataExtractor requires {} to exist'.format(self.category))

        subject_index = params['subject_index']
        index_expr = params['index_expr']

        base_name = self.entry_base_names[subject_index]
        data = reader.read('{}/{}'.format(df.DATA_PLACEHOLDER.format(self.category), base_name), index_expr)
        label_names = extracted['{}_names'.format(self.category)]  # type: list

        if self.selection is None:
            selection_indices = np.array(range(len(label_names)))
        else:
            selection_indices = np.array([label_names.index(s) for s in self.selection])

        random_index = [np.random.choice(selection_indices)]  # as list to keep the last dimension with np.take
        extracted[self.category] = np.take(data, random_index, axis=-1)


class ImageShapeExtractor(Extractor):

    def __init__(self, numpy_format: bool=True) -> None:
        super().__init__()
        self.numpy_format = numpy_format

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        subject_index_expr = expr.IndexExpression(params['subject_index'])

        shape = reader.read(df.INFO_SHAPE, subject_index_expr)
        if self.numpy_format:
            tmp = shape[0]
            shape[0] = shape[-1]
            shape[-1] = tmp

        extracted['shape'] = tuple(shape.tolist())


class DataExtractor(Extractor):

    def __init__(self, categories=('images',), ignore_indexing: bool=False) -> None:
        super().__init__()
        self.categories = categories
        self.ignore_indexing = ignore_indexing
        self.entry_base_names = None

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        if self.entry_base_names is None:
            entries = reader.get_subject_entries()
            self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

        subject_index = params['subject_index']
        index_expr = params['index_expr']

        base_name = self.entry_base_names[subject_index]
        for category in self.categories:
            if self.ignore_indexing:
                data = reader.read('{}/{}'.format(df.DATA_PLACEHOLDER.format(category), base_name))
            else:
                data = reader.read('{}/{}'.format(df.DATA_PLACEHOLDER.format(category), base_name), index_expr)
            extracted[category] = data


class PadDataExtractor(Extractor):

    def __init__(self, padding: t.Union[tuple, t.List[tuple]], extractor: Extractor, pad_fn=None):
        super().__init__()
        if not (hasattr(extractor, 'categories') or hasattr(extractor, 'category')):
            raise ValueError('argument extractor needs to have the property "categories" or "category"')

        self.extractor = extractor
        self.pad_fn = pad_fn

        if self.pad_fn is None:
            self.pad_fn = PadDataExtractor.zero_pad

        if isinstance(padding[0], int):
            padding = [(pad, pad) for pad in padding]
        index_diffs = np.asarray(padding)
        index_diffs[:, 0] = -index_diffs[:, 0]
        self.index_diffs = index_diffs

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        index_expr = params['index_expr']  # type: expr.IndexExpression

        # Make sure all indexing is done with slices (Example: (16,) will be changed to (slice(16, 17, None),) which
        # is equivalent), otherwise the following steps will be wrong; .
        if any(isinstance(s, int) for s in index_expr.expression):
            index_expr.set_indexing([slice(s, s + 1) if isinstance(s, int) else s for s in index_expr.expression])

        padded_indexing = np.asarray(index_expr.get_indexing()) + self.index_diffs
        padded_shape = tuple((padded_indexing[:, 1] - padded_indexing[:, 0]).tolist())

        sub_indexing = padded_indexing.copy()
        sub_indexing[padded_indexing > 0] = 0
        sub_indexing = -sub_indexing

        padded_indexing[padded_indexing < 0] = 0  # cannot slice outside the boundary in negative (but positive works!)
        padded_index_expr = expr.IndexExpression(padded_indexing.tolist())

        padded_params = params.copy()
        padded_params['index_expr'] = padded_index_expr
        self.extractor.extract(reader, padded_params, extracted)

        categories = self.extractor.categories if hasattr(self.extractor, 'categories') else [self.extractor.category]

        for category in categories:
            data = extracted[category]

            full_pad_shape = padded_shape + data.shape[len(padded_shape):]
            if full_pad_shape != data.shape:
                # we could not fully extract the padded shape, use pad_fn to pad data
                extracted[category] = self.pad_fn(data, full_pad_shape, sub_indexing)

    @staticmethod
    def zero_pad(data: np.ndarray, pad_shape, sub_indexing):
        pad_data = np.zeros(pad_shape, dtype=data.dtype)

        sub_indexing[:, 1] = sub_indexing[:, 0] + data.shape[:sub_indexing.shape[0]]
        sub_index_expr = expr.IndexExpression(sub_indexing.tolist())

        pad_data[sub_index_expr.expression] = data
        return pad_data
