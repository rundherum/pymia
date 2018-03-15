import abc
import pickle

import numpy as np


class Assembler(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def add_sample(self, prediction, batch: dict):
        pass


class SubjectAssembler(Assembler):
    """Assembles predictions of one or multiple subjects."""

    def __init__(self):
        self.predictions = {}
        self.subjects_ready = set()

    def add_sample(self, to_assemble, batch: dict, last_batch=False):
        if 'subject_index' not in batch:
            raise ValueError('SubjectAssembler requires "subject_index" to be extracted (use IndexingExtractor)')
        if 'index_expr' not in batch:
            raise ValueError('SubjectAssembler requires "index_expr" to be extracted (use IndexingExtractor)')
        if 'shape' not in batch:
            raise ValueError('SubjectAssembler requires "shape" to be extracted (use ShapeExtractor)')

        if not isinstance(to_assemble, dict):
            to_assemble = {'__prediction': to_assemble}

        for idx, subject_index in enumerate(batch['subject_index']):
            # initialize subject
            if subject_index not in self.predictions and not self.predictions:
                self.predictions[subject_index] = {key: np.zeros(batch['shape'][idx]) for key in to_assemble}
            elif subject_index not in self.predictions:
                self.subjects_ready = set(self.predictions.keys())
                self.predictions[subject_index] ={key: np.zeros(batch['shape'][idx]) for key in to_assemble}

            index_expr = batch['index_expr'][idx]
            if isinstance(index_expr, str):
                # is pickled
                index_expr = pickle.loads(index_expr)

            for key in to_assemble:
                self.predictions[subject_index][key][index_expr.expression] = to_assemble[key][index_expr.expression]

            if last_batch:
                # to prevent from last batch to be ignored
                self.subjects_ready = set(self.predictions.keys())

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
                raise ValueError('Subject "{}" not in assembler'.format(subject_index))
        assembled = self.predictions.pop(subject_index)
        if '__prediction' in assembled:
            return assembled['__prediction']
        return assembled
