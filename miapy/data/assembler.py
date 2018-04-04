import abc
import pickle

import numpy as np

from . import transformation as tfm


class Assembler(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def add_batch(self, prediction, batch: dict):
        pass


def numpy_zeros(shape: tuple, id_: str):
    return np.zeros(shape)


class SubjectAssembler(Assembler):
    """Assembles predictions of one or multiple subjects."""

    def __init__(self, zero_fn=numpy_zeros):
        self.predictions = {}
        self.subjects_ready = set()
        self.zero_fn = zero_fn

    def add_batch(self, to_assemble, batch: dict, last_batch=False, transform: tfm.Transform=None):
        # todo(fabianbalsiger): replace tfm.Transform by callback
        if 'subject_index' not in batch:
            raise ValueError('SubjectAssembler requires "subject_index" to be extracted (use IndexingExtractor)')
        if 'index_expr' not in batch:
            raise ValueError('SubjectAssembler requires "index_expr" to be extracted (use IndexingExtractor)')
        if 'shape' not in batch:
            raise ValueError('SubjectAssembler requires "shape" to be extracted (use ImageShapeExtractor)')

        if not isinstance(to_assemble, dict):
            to_assemble = {'__prediction': to_assemble}

        for idx in range(len(batch['subject_index'])):
            self.add_sample(to_assemble, batch, idx, transform)

        if last_batch:
            # to prevent from last batch to be ignored
            self.end()

    def end(self):
        self.subjects_ready = set(self.predictions.keys())

    def add_sample(self, to_assemble, batch, idx, transform: tfm.Transform=None):
        # todo(fabianbalsiger): replace tfm.Transform by callback
        # initialize subject
        subject_index = batch['subject_index'][idx]

        if subject_index not in self.predictions and not self.predictions:
            self.predictions[subject_index] = self._init_new_subject(batch, to_assemble, idx)
        elif subject_index not in self.predictions:
            self.subjects_ready = set(self.predictions.keys())
            self.predictions[subject_index] = self._init_new_subject(batch, to_assemble, idx)

        index_expr = batch['index_expr'][idx]
        if isinstance(index_expr, bytes):
            # is pickled
            index_expr = pickle.loads(index_expr)

        for key in to_assemble:
            data = to_assemble[key][idx]
            if transform is not None:
                data = transform({key: data, 'batch': batch, 'batch_idx': idx})[key]
            self.predictions[subject_index][key][index_expr.expression] = data

    def _init_new_subject(self, batch, to_assemble, idx):
        subject_prediction = {}
        for key in to_assemble:
            subject_shape = batch['shape'][idx]
            if to_assemble[key].ndim > len(subject_shape) and to_assemble[key].shape[-1] > 1:
                subject_shape += (to_assemble[key].shape[-1],)
            subject_prediction[key] = self.zero_fn(subject_shape, key)
        return subject_prediction

    def get_assembled_subject(self, subject_index: int):
        """Gets the prediction of a subject.

        Args:
            subject_index (int): The subject's index.

        Returns:
            np.ndarray: The prediction of the subject.
        """
        try:
            self.subjects_ready.remove(subject_index)
        except KeyError:
            # check if subject is assembled but not listed as ready
            # this can happen if only one subject was assembled or last
            if subject_index not in self.predictions:
                raise ValueError('Subject with index {} not in assembler'.format(subject_index))
        assembled = self.predictions.pop(subject_index)
        if '__prediction' in assembled:
            return assembled['__prediction']
        return assembled

