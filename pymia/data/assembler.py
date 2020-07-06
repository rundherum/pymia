import abc
import typing

import numpy as np

import pymia.data.definition as defs
import pymia.data.transformation as tfm
import pymia.data.extraction as extr


class Assembler(abc.ABC):
    """Interface for assembling images from batch, which contain parts (chunks) of the images only."""

    @abc.abstractmethod
    def add_batch(self, to_assemble, sample_indices, last_batch=False, **kwargs):
        """Add the batch results to be assembled.

        Args:
            to_assemble (object, dict): object or dictionary of objects to be assembled to an image.
            sample_indices (iterable): iterable of all the sample indices in the processed batch
            last_batch (bool): Whether the current batch is the last.
        """
        pass

    @abc.abstractmethod
    def get_assembled_subject(self, subject_index: int):
        """
        Args:
            subject_index (int): Index of the assembled subject to be retrieved.

        Returns:
            object: The assembled data of the subject (might be multiple arrays).
        """
        pass

    @property
    @abc.abstractmethod
    def subjects_ready(self):
        """list, set: The indices of the subjects that are finished assembling."""
        pass


def numpy_zeros(shape: tuple, assembling_key: str, subject_index: int):
    return np.zeros(shape)


class SubjectAssembler(Assembler):

    def __init__(self, datasource: extr.PymiaDatasource, zero_fn=numpy_zeros, assemble_interaction_fn=None):
        """Assembles predictions of one or multiple subjects.

        Assumes that the network output, i.e. to_assemble, is of shape (B, ..., C)
        where B is the batch size and C is the numbers of channels (must be at least 1) and ... refers to an arbitrary image
        dimension.

        Args:
            datasource (.PymiaDatasource): The datasource.
            zero_fn: A function that initializes the numpy array to hold the predictions.
                Args: shape: tuple with the shape of the subject's labels.
                Returns: A np.ndarray
            assemble_interaction_fn (callable, optional): A `callable` that may modify the sample and indexing before adding
                the data to the assembled array. This enables handling special cases. Must follow the
                :code:`.AssembleInteractionFn.__call__` interface. By default neither data nor indexing is modified.
        """
        self.datasource = datasource
        self.zero_fn = zero_fn
        self.assemble_interaction_fn = assemble_interaction_fn
        self._subjects_ready = set()
        self.predictions = {}

    @property
    def subjects_ready(self):
        """see :meth:`Assembler.subjects_ready`"""
        return self._subjects_ready.copy()

    def add_batch(self, to_assemble: typing.Union[np.ndarray, typing.Dict[str, np.ndarray]], sample_indices: np.ndarray,
                  last_batch=False, **kwargs):
        """see :meth:`Assembler.add_batch`"""
        if not isinstance(to_assemble, dict):
            to_assemble = {'__prediction': to_assemble}

        sample_indices = sample_indices.tolist()  # to ensure that not np.int64 entries, but int

        for batch_idx, sample_idx in enumerate(sample_indices):
            self.add_sample(to_assemble, batch_idx, sample_idx)

        if last_batch:
            # to prevent from last batch to be ignored
            self.end()

    def end(self):
        self._subjects_ready = set(self.predictions.keys())

    def add_sample(self, to_assemble, batch_idx, sample_idx):
        subject_index, index_expression = self.datasource.indices[sample_idx]

        # initialize subject
        if subject_index not in self.predictions and not self.predictions:  # fist subject
            self.predictions[subject_index] = self._init_new_subject(to_assemble, subject_index)
        elif subject_index not in self.predictions:  # new subject
            self._subjects_ready = set(self.predictions.keys())
            self.predictions[subject_index] = self._init_new_subject(to_assemble, subject_index)

        for key in to_assemble:
            data = to_assemble[key][batch_idx]
            if self.assemble_interaction_fn:
                data, index_expression = self.assemble_interaction_fn(key, data, index_expression)
            self.predictions[subject_index][key][index_expression.expression] = data

    def _init_new_subject(self, to_assemble, subject_index):
        subject_prediction = {}

        extractor = extr.ImagePropertyShapeExtractor(numpy_format=True)
        subject_shape = self.datasource.direct_extract(extractor, subject_index)[defs.KEY_SHAPE]
        for key in to_assemble:
            assemble_shape = subject_shape + (to_assemble[key].shape[-1],)
            subject_prediction[key] = self.zero_fn(assemble_shape, key, subject_index)
        return subject_prediction

    def get_assembled_subject(self, subject_index: int):
        """see :meth:`Assembler.get_assembled_subject`"""
        try:
            self._subjects_ready.remove(subject_index)
        except KeyError:
            # check if subject is assembled but not listed as ready
            # this can happen if only one subject was assembled or last
            if subject_index not in self.predictions:
                raise ValueError('Subject with index {} not in assembler'.format(subject_index))
        assembled = self.predictions.pop(subject_index)
        if '__prediction' in assembled:
            return assembled['__prediction']
        return assembled


def mean_merge_fn(planes: list):
    return np.stack(planes).mean(axis=0)


class AssembleInteractionFn:
    """ Function interface enabling interaction with the `index_expression` and the `data` before it gets added to the
    assembled `prediction` in :class:`.SubjectAssembler`.


    .. automethod:: __call__
    """

    def __call__(self, key, data, index_expr, **kwargs):
        """

        Args:
            key (str): The identifier or key of the data.
            data (numpy.ndarray): The data.
            index_expr (.IndexExpression): The current index_expression that might be modified.
            **kwargs (dict): Any other arguments

        Returns:
            tuple: Modified `data` and modified `index_expression`

        """
        raise NotImplementedError()


class ApplyTransformInteractionFn(AssembleInteractionFn):

    def __init__(self, transform: tfm.Transform) -> None:
        self.transform = transform

    def __call__(self, key, data, index_expr, **kwargs):
        temp = tfm.raise_error_if_entry_not_extracted
        tfm.raise_error_entry_not_extracted = False
        ret = self.transform({key: data, defs.KEY_INDEX_EXPR: index_expr})
        tfm.raise_error_entry_not_extracted = temp

        return ret[key], ret[defs.KEY_INDEX_EXPR]


class PlaneSubjectAssembler(Assembler):

    def __init__(self, datasource: extr.PymiaDatasource, merge_fn=mean_merge_fn, zero_fn=numpy_zeros):
        """Assembles predictions of one or multiple subjects where predictions are made in all three planes.

        This class assembles the prediction from all planes (axial, coronal, sagittal) and merges the prediction
        according to :code:`merge_fn`.

        Assumes that the network output, i.e. to_assemble, is of shape (B, ..., C)
        where B is the batch size and C is the numbers of channels (must be at least 1) and ... refers to an arbitrary image
        dimension.

        Args:
            datasource (.PymiaDatasource): The datasource
            merge_fn: A function that processes a sample.
                Args: planes: list with the assembled prediction for all planes.
                Returns: Merged numpy.ndarray
            zero_fn: A function that initializes the numpy array to hold the predictions.
                Args: shape: tuple with the shape of the subject's labels, id: str identifying the subject.
                Returns: A np.ndarray
        """
        self.datasource = datasource
        self.planes = {}  # type: typing.Dict[int, SubjectAssembler]
        self._subjects_ready = set()
        self.zero_fn = zero_fn
        self.merge_fn = merge_fn

    @property
    def subjects_ready(self):
        """see :meth:`Assembler.subjects_ready`"""
        return self._subjects_ready.copy()

    def add_batch(self, to_assemble: typing.Union[np.ndarray, typing.Dict[str, np.ndarray]], sample_indices: np.ndarray,
                  last_batch=False, **kwargs):
        """see :meth:`Assembler.add_batch`"""

        if not isinstance(to_assemble, dict):
            to_assemble = {'__prediction': to_assemble}

        sample_indices = sample_indices.tolist()  # to ensure that not np.int64 entries, but int

        for batch_idx, sample_idx in enumerate(sample_indices):
            subject_index, index_expression = self.datasource.indices[sample_idx]

            plane_dimension = self._get_plane_dimension(index_expression)

            if plane_dimension not in self.planes:
                self.planes[plane_dimension] = SubjectAssembler(self.datasource, self.zero_fn)

            indexing = index_expression.get_indexing()
            if not isinstance(indexing, list):
                indexing = [indexing]

            extractor = extr.ImagePropertyShapeExtractor(numpy_format=True)
            required_plane_shape = self.datasource.direct_extract(extractor, subject_index)[defs.KEY_SHAPE]

            index_at_plane = indexing[plane_dimension]
            if isinstance(index_at_plane, tuple):
                # is a range in the off plane direction (tuple)
                required_off_plane_size = index_at_plane[1] - index_at_plane[0]
                required_plane_shape[plane_dimension] = required_off_plane_size
            else:  # isinstance of int
                # is one slice in off plane direction (int)
                required_plane_shape.pop(plane_dimension)
            transform = tfm.SizeCorrection(tuple(required_plane_shape), entries=tuple(to_assemble.keys()))
            self.planes[plane_dimension].assemble_interaction_fn = ApplyTransformInteractionFn(transform)

            self.planes[plane_dimension].add_sample(to_assemble, batch_idx, sample_idx)

        ready = None
        for plane_assembler in self.planes.values():
            if last_batch:
                plane_assembler.end()

            if ready is None:
                ready = set(plane_assembler.subjects_ready)
            else:
                ready.intersection_update(plane_assembler.subjects_ready)
        self._subjects_ready = ready

    def get_assembled_subject(self, subject_index: int):
        """see :meth:`Assembler.get_assembled_subject`"""
        try:
            self._subjects_ready.remove(subject_index)
        except KeyError:
            # check if subject is assembled but not listed as ready
            # this can happen if only one subject was assembled or last
            if subject_index not in self.planes[0].predictions:
                raise ValueError('Subject with index {} not in assembler'.format(subject_index))

        assembled = {}
        for plane in self.planes.values():
            ret_val = plane.get_assembled_subject(subject_index)
            if not isinstance(ret_val, dict):
                ret_val = {'__prediction': ret_val}
            for key, value in ret_val.items():
                assembled.setdefault(key, []).append(value)

        for key in assembled:
            assembled[key] = self.merge_fn(assembled[key])

        if '__prediction' in assembled:
            return assembled['__prediction']
        return assembled

    @staticmethod
    def _get_plane_dimension(index_expr):
        for i, entry in enumerate(index_expr.expression):
            if isinstance(entry, int):
                return i
            if isinstance(entry, slice) and entry != slice(None):
                return i


class Subject2dAssembler(Assembler):

    def __init__(self, datasource: extr.PymiaDatasource) -> None:
        """Assembles predictions of two-dimensional images.

        Two-dimensional images do not specifically require assembling. For pipeline compatibility reasons this class provides
        , nevertheless, a implementation for the two-dimensional case.
        
        Args:
            datasource (.PymiaDatasource): The datasource
        """
        super().__init__()
        self.datasource = datasource
        self._subjects_ready = set()
        self.predictions = {}

    @property
    def subjects_ready(self):
        """see :meth:`Assembler.subjects_ready`"""
        return self._subjects_ready.copy()

    def add_batch(self, to_assemble: typing.Union[np.ndarray, typing.Dict[str, np.ndarray]], sample_indices: np.ndarray,
                  last_batch=False, **kwargs):
        """see :meth:`Assembler.add_batch`"""

        if not isinstance(to_assemble, dict):
            to_assemble = {'__prediction': to_assemble}

        sample_indices = sample_indices.tolist()  # to ensure that not np.int64 entries, but int

        for batch_idx, sample_idx in enumerate(sample_indices):
            subject_index, _ = self.datasource.indices[sample_idx]

            for key in to_assemble:
                self.predictions.setdefault(subject_index, {})[key] = to_assemble[key][batch_idx]
                self._subjects_ready.add(subject_index)

    def get_assembled_subject(self, subject_index):
        """see :meth:`Assembler.get_assembled_subject`"""
        try:
            self._subjects_ready.remove(subject_index)
        except KeyError:
            # check if subject is assembled but not listed as ready
            # this can happen if only one subject was assembled or last
            if subject_index not in self.predictions:
                raise ValueError('Subject with index {} not in assembler'.format(subject_index))
        assembled = self.predictions.pop(subject_index)
        if '__prediction' in assembled:
            return assembled['__prediction']
        return assembled
