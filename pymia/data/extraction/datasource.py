import pymia.data.definition as defs
import pymia.data.indexexpression as expr
import pymia.data.transformation as tfm
from . import extractor as extr
from . import indexing as idx
from . import reader as rd


class PymiaDatasource:

    def __init__(self, dataset_path: str,
                 indexing_strategy: idx.IndexingStrategy = None,
                 extractor: extr.Extractor = None,
                 transform: tfm.Transform = None,
                 subject_subset: list = None,
                 init_reader_once: bool = True) -> None:
        """Provides convenient and adaptable reading of the data from a created dataset.

        Args:
            dataset_path (str): The path to the dataset to be read from.
            indexing_strategy(.IndexingStrategy): Strategy defining how the data is indexed for reading.
            extractor (.Extractor): Extractor or multiple extractors (:class:`.ComposeExtractor`) extracting the desired
                data from the dataset.
            transform (.Transform): Transformation(s) to be applied to the extracted data.
            subject_subset (list): A list of subject identifiers defining a subset of subject to be processed.
            init_reader_once (bool): Whether the reader is initialized once or for every retrieval (default: :code:`True`)

        Examples:
            The class mainly allows to modes of operation. The first mode is by extracting the data by index.

            >>> ds = PymiaDatasource(...)
            >>> for i in range(len(ds)):
            >>>     sample = ds[i]

            The second mode of operation is by directly extracting data.

            >>> ds = PymiaDatasource(...)
            >>> # Different from ds[index] since the extractor and transform override the ones in ds
            >>> sample = ds.direct_extract(extractor, index, transform=transform)

            Typically, the first mode is use to loop over the entire dataset as fast as possible, extracting just the necessary
            information, such as data chunks (e.g., slice, patch, sub-volume). Less critical information
            (e.g. image shape, orientation) not required with every chunk of data can independently be extracted with the second
            mode of operation.
        """

        self.dataset_path = dataset_path
        self.indexing_strategy = None
        self.extractor = extractor
        self.transform = transform
        self.subject_subset = subject_subset
        self.init_reader_once = init_reader_once
        self.indices = []
        """list: A list containing all sample indices. This is a mapping from item `i` to tuple 
        `(subject_index, index_expression)`."""
        self.reader = None

        # init indices
        if indexing_strategy is None:
            indexing_strategy = idx.EmptyIndexing()
        self.set_indexing_strategy(indexing_strategy, subject_subset)

    def close_reader(self):
        """Close the reader."""
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def set_extractor(self, extractor: extr.Extractor):
        """Set the extractor(s).

        Args:
            extractor (.Extractor): Extractor or multiple extractors (:class:`.ComposeExtractor`) extracting the desired
                data from the dataset.
        """
        self.extractor = extractor

    def set_indexing_strategy(self, indexing_strategy: idx.IndexingStrategy, subject_subset: list = None):
        """Set (or modify) the indexing strategy.

        Args:
            indexing_strategy (.IndexingStrategy): Strategy defining how the data is indexed for reading.
            subject_subset (list): A list of subject identifiers defining a subset of subject to be processed.
        """
        self.indices.clear()
        self.indexing_strategy = indexing_strategy
        with rd.get_reader(self.dataset_path) as reader:
            all_subjects = reader.get_subjects()
            last_shape = None  # remember shape to optimize initialization
            for subject_idx in range(len(all_subjects)):
                if subject_subset is None or all_subjects[subject_idx] in subject_subset:
                    current_shape = reader.get_shape(subject_idx)
                    if not last_shape == current_shape:
                        subject_indices = self.indexing_strategy(current_shape)
                        last_shape = current_shape

                    subject_and_indices = zip(len(subject_indices) * [subject_idx], subject_indices)
                    self.indices.extend(subject_and_indices)

    def set_transform(self, transform: tfm.Transform):
        """Set the transform.

        Args:
            transform (.Transform): Transformation(s) to be applied to the extracted data.
        """
        self.transform = transform

    def get_subjects(self):
        """"Get all the subjects in the dataset.

        Returns:
            list: All subject identifiers in the dataset.
        """
        with rd.get_reader(self.dataset_path) as reader:
            return reader.get_subjects()

    def direct_extract(self, extractor: extr.Extractor, subject_index: int, index_expr: expr.IndexExpression = None,
                       transform: tfm.Transform = None):
        """Extract data directly, bypassing the extractors and transforms of the instance.

        The purpose of this method is to enable extraction of data that is not required for every data chunk
        (e.g., slice, patch, sub-volume) but only from time to time e.g., image shape, origin.

        Args:
            extractor (.Extractor): Extractor or multiple extractors (:class:`.ComposeExtractor`) extracting the desired
                data from the dataset.
            subject_index (int): Index of the subject to be extracted.
            index_expr (.IndexExpression): The indexing to extract a chunk of data only.
                Not required if only image related information (e.g., image shape, origin) should be extracted.
                Needed when desiring a chunk of data (e.g., slice, patch, sub-volume).
            transform (.Transform): Transformation(s) to be applied to the extracted data.

        Returns:
            dict: Extracted data in a dictionary. Keys are defined by the used :class:`.Extractor`.
        """
        if index_expr is None:
            index_expr = expr.IndexExpression()

        params = {defs.KEY_SUBJECT_INDEX: subject_index, defs.KEY_INDEX_EXPR: index_expr}
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
        extracted = self.direct_extract(self.extractor, subject_index, index_expr, self.transform)
        extracted[defs.KEY_SAMPLE_INDEX] = item
        return extracted

    def __del__(self):
        self.close_reader()
