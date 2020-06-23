import abc
import os

import h5py
import numpy as np

import pymia.data.definition as defs
import pymia.data.indexexpression as expr


class Reader(abc.ABC):

    def __init__(self, file_path: str) -> None:
        """Abstract dataset reader.

        Args:
            file_path(str): The path to the dataset file.
        """
        super().__init__()
        self.file_path = file_path

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    @abc.abstractmethod
    def get_subject_entries(self) -> list:
        """Get the dataset entries holding the subject's data.

        Returns:
            list: The list of subject entry strings.
        """
        pass

    @abc.abstractmethod
    def get_shape(self, subject_index: int) -> list:
        """Get the shape from an entry.

        Args:
            subject_index(int): The index of the subject.

        Returns:
            list: The shape of each dimension.
        """
        pass

    @abc.abstractmethod
    def get_subjects(self) -> list:
        """Get the subject names in the dataset.

        Returns:
            list: The list of subject names.
        """
        pass

    @abc.abstractmethod
    def read(self, entry: str, index: expr.IndexExpression = None):
        """Read a dataset entry.

        Args:
            entry(str): The dataset entry.
            index(expr.IndexExpression): The slicing expression.

        Returns:
            The read data.
        """
        pass

    @abc.abstractmethod
    def has(self, entry: str) -> bool:
        """Check whether a dataset entry exists.

        Args:
            entry(str): The dataset entry.

        Returns:
            bool: Whether the entry exists.
        """
        pass

    @abc.abstractmethod
    def open(self):
        """Open the reader."""
        pass

    @abc.abstractmethod
    def close(self):
        """Close the reader."""
        pass


class Hdf5Reader(Reader):
    """Represents the dataset reader for HDF5 files."""

    def __init__(self, file_path: str, category=defs.KEY_IMAGES) -> None:
        """Initializes a new instance.

        Args:
            file_path(str): The path to the dataset file.
            category(str): The category of an entry that defines the shape request
        """
        super().__init__(file_path)
        self.h5 = None  # type: h5py.File
        self.category = category

    def get_subject_entries(self) -> list:
        """see :meth:`.Reader.get_subject_entries`"""
        nb_subjects = len(self.get_subjects())
        return [defs.subject_index_to_str(i, nb_subjects) for i in range(nb_subjects)]

    def get_shape(self, subject_index: int) -> list:
        """see :meth:`.Reader.get_shape`"""
        return self.read(defs.LOC_SHAPE_PLACEHOLDER.format(self.category), expr.IndexExpression(subject_index)).tolist()

    def get_subjects(self) -> list:
        """see :meth:`.Reader.get_subjects`"""
        return self.read(defs.LOC_SUBJECT)

    def read(self, entry: str, index: expr.IndexExpression = None):
        """see :meth:`.Reader.read`"""
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
        """see :meth:`.Reader.has`"""
        return entry in self.h5

    def open(self):
        """see :meth:`.Reader.open`"""
        self.h5 = h5py.File(self.file_path, mode='r', libver='latest')

    def close(self):
        """see :meth:`.Reader.close`"""
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None


def get_reader(file_path: str, direct_open: bool = False) -> Reader:
    """Get the dataset reader corresponding to the file extension.

    Args:
        file_path(str): The path to the dataset file.
        direct_open(bool): Whether the file should directly be opened.

    Returns:
        Reader: Reader corresponding to dataset file extension.
    """

    extension = os.path.splitext(file_path)[1]
    if extension not in reader_registry:
        raise ValueError('unknown dataset file extension "{}"'.format(extension))

    reader = reader_registry[extension](file_path)
    if direct_open:
        reader.open()
    return reader


reader_registry = {'.h5': Hdf5Reader, '.hdf5': Hdf5Reader}
"""Registry defining the mapping between file extension and :class:`.Reader` class. 
    Alternative writers need to be added to this registry in order to use :func:`.get_reader`."""