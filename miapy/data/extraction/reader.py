import abc

import h5py
import numpy as np

import miapy.data.indexexpression as expr
import miapy.data.definition as df


class Reader(metaclass=abc.ABCMeta):

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    @abc.abstractmethod
    def get_subject_entries(self) -> list:
        pass

    @abc.abstractmethod
    def get_shape(self, entry: str) -> list:
        pass

    @abc.abstractmethod
    def get_subjects(self) -> list:
        pass

    @abc.abstractmethod
    def read(self, entry: str, index: expr.IndexExpression=None):
        pass

    @abc.abstractmethod
    def has(self, entry: str) -> bool:
        pass

    @abc.abstractmethod
    def open(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class Hdf5Reader(Reader):

    def __init__(self, file_name: str, open_file: bool=False) -> None:
        self.h5 = None  # type: h5py.File
        self.file_name = file_name

        if open_file:
            self.open()

    def get_subject_entries(self) -> list:
        return ['{}/{}'.format(df.DATA_IMAGE, k) for k in self.h5[df.DATA_IMAGE].keys()]

    def get_shape(self, entry: str) -> list:
        return self.h5[entry].shape

    def get_subjects(self) -> list:
        return self.read(df.SUBJECT)

    def read(self, entry: str, index: expr.IndexExpression=None):
        if index is None:
            data = self.h5[entry][()]  # need () instead of util.IndexExpression(None) [which is equal to slice(None)]
        else:
            data = self.h5[entry][index.expression]

        if isinstance(data, np.ndarray) and data.dtype == np.object:
            return data.tolist()
        # if h5py.check_dtype(vlen=self.h5[entry].dtype) == str and not isinstance(data, str):
        #     return data.tolist()
        return data

    def has(self, entry: str) -> bool:
        return entry in self.h5

    def open(self):
        self.h5 = h5py.File(self.file_name, 'r')

    def close(self):
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None
