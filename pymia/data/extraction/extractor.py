import abc
import pickle
import typing

import numpy as np
import SimpleITK as sitk

import pymia.data.conversion as conv
import pymia.data.definition as defs
import pymia.data.indexexpression as expr
from . import reader as rd


class Extractor(abc.ABC):
    """Interface unifying the extraction of data from a dataset."""

    @abc.abstractmethod
    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        """Extract data from the dataset.

        Args:
            reader (.Reader): Reader instance that can read from dataset.
            params (dict): Extraction parameters containing information such as subject index and index expression.
            extracted (dict): The dictionary to put the extracted data in.
        """
        pass


class ComposeExtractor(Extractor):

    def __init__(self, extractors: list) -> None:
        """ Composes many :class:`.Extractor` instances and behaves like an single :class:`.Extractor` instance.

        Args:
            extractors (list): A list of :class:`.Extractor` instances.
        """
        super().__init__()
        self.extractors = extractors

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        """see :meth:`.Extractor.extract`"""
        for e in self.extractors:
            e.extract(reader, params, extracted)


class NamesExtractor(Extractor):
    def __init__(self, cache: bool = True, categories=(defs.KEY_IMAGES, defs.KEY_LABELS)) -> None:
        """ Extracts the names of the entries within a category (e.g. "Flair", "T1" for the category "images").

        Added key to :obj:`extracted`:

        - :const:`pymia.data.definition.KEY_PLACEHOLDER_NAMES` with :obj:`str` content

        Args:
            cache (bool): Whether to cache the results. If :code:`True`, the dataset is only accessed once.
                :code:`True` is often preferred since the name entries are typically unique in the dataset.
            categories (tuple): Categories for which to extract the names.
        """
        super().__init__()
        self.cache = cache
        self.cached_result = None
        self.categories = categories

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        """see :meth:`.Extractor.extract`"""
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
            d[defs.KEY_PLACEHOLDER_NAMES.format(category)] = reader.read(defs.LOC_NAMES_PLACEHOLDER.format(category))
        return d


class SubjectExtractor(Extractor):
    """Extracts the subject's identification.

    Added key to :obj:`extracted`:

    - :const:`pymia.data.definition.KEY_SUBJECT_INDEX` with :obj:`int` content
    - :const:`pymia.data.definition.KEY_SUBJECT` with :obj:`str` content
    """

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        """see :meth:`.Extractor.extract`"""
        extracted[defs.KEY_SUBJECT_INDEX] = params[defs.KEY_SUBJECT_INDEX]
        subject_index_expr = expr.IndexExpression(params[defs.KEY_SUBJECT_INDEX])
        extracted[defs.KEY_SUBJECT] = reader.read(defs.LOC_SUBJECT, subject_index_expr)


class IndexingExtractor(Extractor):
    """Extracts the index expression.

    Added key to :obj:`extracted`:

    - :const:`pymia.data.definition.KEY_SUBJECT_INDEX` with :obj:`int` content
    - :const:`pymia.data.definition.KEY_INDEX_EXPR` with :class:`.IndexExpression` content
    """

    def __init__(self, do_pickle: bool = False) -> None:
        super().__init__()
        self.do_pickle = do_pickle

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        """see :meth:`.Extractor.extract`"""
        extracted[defs.KEY_SUBJECT_INDEX] = params[defs.KEY_SUBJECT_INDEX]
        index_expression = params[defs.KEY_INDEX_EXPR]
        if self.do_pickle:
            # pickle to prevent from problems since own class
            index_expression = pickle.dumps(index_expression)
        extracted[defs.KEY_INDEX_EXPR] = index_expression


class ImagePropertiesExtractor(Extractor):
    def __init__(self, do_pickle: bool = False) -> None:
        """
        Extracts the image properties.

        Added key to :obj:`extracted`:

        - :const:`pymia.data.definition.KEY_PROPERTIES` with :class:`.ImageProperties` content (or string if :code:`do_pickle`)

        Args:
            do_pickle (bool): whether to pickle the extracted :class:`.ImageProperties` instance.
                This allows usage in multiprocessing environment.
        """
        super().__init__()
        self.do_pickle = do_pickle

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        """see :meth:`.Extractor.extract`"""
        subject_index_expr = expr.IndexExpression(params[defs.KEY_SUBJECT_INDEX])

        shape = reader.read(defs.LOC_INFO_SHAPE, subject_index_expr).tolist()
        direction = reader.read(defs.LOC_INFO_DIRECTION, subject_index_expr).tolist()
        spacing = reader.read(defs.LOC_INFO_SPACING, subject_index_expr).tolist()
        origin = reader.read(defs.LOC_INFO_ORIGIN, subject_index_expr).tolist()

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
        extracted[defs.KEY_PROPERTIES] = img_properties


class FilesExtractor(Extractor):

    def __init__(self, cache: bool = True, categories=(defs.KEY_IMAGES, defs.KEY_LABELS)) -> None:
        """Extracts the file paths.

        Added key to :obj:`extracted`:

        - :const:`pymia.data.definition.KEY_FILE_ROOT` with :obj:`str` content
        - :const:`pymia.data.definition.KEY_PLACEHOLDER_FILES` with :obj:`str` content

        Args:
            cache (bool): Whether to cache the results. If :code:`True`, the dataset is only accessed once.
                :code:`True` is often preferred since the file name entries are typically unique in the dataset
                (i.e. independent of data chunks).
            categories (tuple): Categories for which to extract the file names.
        """
        super().__init__()
        self.cache = cache
        self.cached_file_root = None
        self.categories = categories

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        """see :meth:`.Extractor.extract`"""
        subject_index_expr = expr.IndexExpression(params[defs.KEY_SUBJECT_INDEX])

        if not self.cache or self.cached_file_root is None:
            file_root = reader.read(defs.LOC_FILES_ROOT)
            self.cached_file_root = file_root
        else:
            file_root = self.cached_file_root

        extracted[defs.KEY_FILE_ROOT] = file_root

        for category in self.categories:
            extracted[defs.KEY_PLACEHOLDER_FILES.format(category)] = reader.read(defs.LOC_FILES_PLACEHOLDER.format(category),
                                                                                 subject_index_expr)


class SelectiveDataExtractor(Extractor):

    def __init__(self, selection=None, category: str = defs.KEY_LABELS) -> None:
        """Extracts data of a given category selectively.

        Adds :obj:`category` as key to :obj:`extracted`.

        Args:
            selection (str, tuple): Entries (e.g., "T1", "T2") within the category to select.
                If selection is None, the class has the same behaviour as the DataExtractor and selects all entries.
            category (str): The category (e.g. "images") to extract data from.

        Note:
            Requires results of :class:`NamesExtractor` in :obj:`extracted`.
        """
        super().__init__()
        self.entry_base_names = None

        if isinstance(selection, str):
            selection = (selection,)
        self.selection = selection
        self.category = category

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        """see :meth:`.Extractor.extract`"""
        if defs.KEY_PLACEHOLDER_NAMES.format(self.category) not in extracted:
            raise ValueError('selection of labels requires label_names to be extracted (use NamesExtractor)')

        if self.entry_base_names is None:
            entries = reader.get_subject_entries()
            self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

        if not reader.has(defs.LOC_DATA_PLACEHOLDER.format(self.category)):
            raise ValueError('SelectiveDataExtractor requires {} to exist'.format(self.category))

        subject_index = params[defs.KEY_SUBJECT_INDEX]
        index_expr = params[defs.KEY_INDEX_EXPR]

        base_name = self.entry_base_names[subject_index]
        data = reader.read('{}/{}'.format(defs.LOC_DATA_PLACEHOLDER.format(self.category), base_name), index_expr)
        label_names = extracted[defs.KEY_PLACEHOLDER_NAMES.format(self.category)]  # type: list

        if self.selection is None:
            extracted[self.category] = data
        else:
            selection_indices = np.array([label_names.index(s) for s in self.selection])
            extracted[self.category] = np.take(data, selection_indices, axis=-1)


class RandomDataExtractor(Extractor):

    def __init__(self, selection=None, category: str = defs.KEY_LABELS) -> None:
        """Extracts data of a given category randomly.

        Adds :obj:`category` as key to :obj:`extracted`.

        Args:
            selection (str, tuple): Entries (e.g., "T1", "T2") within the category to select an entry randomly from.
                If selection is None, an entry from all entries is randomly selected.
            category (str): The category (e.g. "images") to extract data from.

        Note:
            Requires results of :class:`NamesExtractor` in :obj:`extracted`.
        """
        super().__init__()
        self.entry_base_names = None

        if isinstance(selection, str):
            selection = (selection,)
        self.selection = selection
        self.category = category

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        """see :meth:`.Extractor.extract`"""
        if defs.KEY_PLACEHOLDER_NAMES.format(self.category) not in extracted:
            raise ValueError('selection of labels requires label_names to be extracted (use NamesExtractor)')

        if self.entry_base_names is None:
            entries = reader.get_subject_entries()
            self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

        if not reader.has(defs.LOC_DATA_PLACEHOLDER.format(self.category)):
            raise ValueError('SelectiveDataExtractor requires {} to exist'.format(self.category))

        subject_index = params[defs.KEY_SUBJECT_INDEX]
        index_expr = params[defs.KEY_INDEX_EXPR]

        base_name = self.entry_base_names[subject_index]
        data = reader.read('{}/{}'.format(defs.LOC_DATA_PLACEHOLDER.format(self.category), base_name), index_expr)
        label_names = extracted[defs.KEY_PLACEHOLDER_NAMES.format(self.category)]  # type: list

        if self.selection is None:
            selection_indices = np.array(range(len(label_names)))
        else:
            selection_indices = np.array([label_names.index(s) for s in self.selection])

        random_index = [np.random.choice(selection_indices)]  # as list to keep the last dimension with np.take
        extracted[self.category] = np.take(data, random_index, axis=-1)


class ImageShapeExtractor(Extractor):

    def __init__(self, numpy_format: bool = True) -> None:
        """Extracts the shape of an image.

        Added key to :obj:`extracted`:

        - :const:`pymia.data.definition.KEY_SHAPE` with :obj:`tuple` content

        Args:
            numpy_format (bool): Whether the shape is numpy or ITK format (first and last dimension are swapped).
        """
        super().__init__()
        self.numpy_format = numpy_format

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        """see :meth:`.Extractor.extract`"""
        subject_index_expr = expr.IndexExpression(params[defs.KEY_SUBJECT_INDEX])

        shape = reader.read(defs.LOC_INFO_SHAPE, subject_index_expr)
        if self.numpy_format:
            tmp = shape[0]
            shape[0] = shape[-1]
            shape[-1] = tmp

        extracted[defs.KEY_SHAPE] = tuple(shape.tolist())


class DataExtractor(Extractor):

    def __init__(self, categories=(defs.KEY_IMAGES, ), ignore_indexing: bool = False) -> None:
        """Extracts data of a given category.

        Adds :obj:`category` as key to :obj:`extracted`.

        Args:
            categories (tuple): Categories for which to extract the names.
            ignore_indexing (bool): Whether to ignore the indexing in :obj:`params`. This is useful when extracting
                entire images.
        """
        super().__init__()
        self.categories = categories
        self.ignore_indexing = ignore_indexing
        self.entry_base_names = None

    def extract(self, reader: rd.Reader, params: dict, extracted: dict) -> None:
        """see :meth:`.Extractor.extract`"""
        if self.entry_base_names is None:
            entries = reader.get_subject_entries()
            self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

        subject_index = params[defs.KEY_SUBJECT_INDEX]
        index_expr = params[defs.KEY_INDEX_EXPR]

        base_name = self.entry_base_names[subject_index]
        for category in self.categories:
            if self.ignore_indexing:
                data = reader.read('{}/{}'.format(defs.LOC_DATA_PLACEHOLDER.format(category), base_name))
            else:
                data = reader.read('{}/{}'.format(defs.LOC_DATA_PLACEHOLDER.format(category), base_name), index_expr)
            extracted[category] = data


class PadDataExtractor(Extractor):

    def __init__(self, padding: typing.Union[tuple, typing.List[tuple]], extractor: Extractor, pad_fn=None):
        """Pads the data extracted by :obj:`extractor`

        Args:
            padding (tuple, list): Lengths of the tuple or the list must be equal to the number of dimensions of the extracted
                data. If tuple, values are considered as symmetric padding in each dimension. If list, the each entry must
                consist of a tuple indicating (left, right) padding for one dimension.
            extractor (.Extractor): The extractor performing the extraction of the data to be padded.
            pad_fn (callable, optional): Optional function performing the padding. Default is :meth:`PadDataExtractor.zero_pad`.
        """
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
        """see :meth:`.Extractor.extract`"""
        index_expr = params[defs.KEY_INDEX_EXPR]  # type: expr.IndexExpression

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
        padded_params[defs.KEY_INDEX_EXPR] = padded_index_expr
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
        """"""
        pad_data = np.zeros(pad_shape, dtype=data.dtype)

        sub_indexing[:, 1] = sub_indexing[:, 0] + data.shape[:sub_indexing.shape[0]]
        sub_index_expr = expr.IndexExpression(sub_indexing.tolist())

        pad_data[sub_index_expr.expression] = data
        return pad_data
