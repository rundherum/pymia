import abc
import os

import h5py
import numpy as np

import pymia.data.indexexpression as expr
import pymia.data.definition as df


class Reader(metaclass=abc.ABCMeta):
    """Represents the abstract dataset reader."""

    def __init__(self, file_path: str) -> None:
        """Initializes a new instance.

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
    def get_shape(self, entry: str) -> list:
        """Get the shape from an entry.

        Args:
            entry(str): The dataset entry.

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
    def read(self, entry: str, index: expr.IndexExpression=None):
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

    def __init__(self, file_path: str, category='images') -> None:
        """Initializes a new instance.

        Args:
            file_path(str): The path to the dataset file.
            category(str): The category of an entry that contains data of all subjects
        """
        super().__init__(file_path)
        self.h5 = None  # type: h5py.File
        self.category = category

    def get_subject_entries(self) -> list:
        group = df.DATA_PLACEHOLDER.format(self.category)
        return ['{}/{}'.format(group, k) for k in sorted(self.h5[group].keys())]

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
        self.h5 = h5py.File(self.file_path, mode='r', libver='latest')

    def close(self):
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None


def get_reader(file_path: str, direct_open: bool=False) -> Reader:
    """ Get the dataset reader corresponding to the file extension.

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
