"""The loading module holds classes to load data."""
import abc
import enum
import typing
import os


class FilePathGenerator(metaclass=abc.ABCMeta):
    """Represents an abstract file path generator.

    This class is used in :py:class:`FileSystemDataCrawler` to convert a human readable data identifier to an data file path,
    which allows to load the data.
    """

    @staticmethod
    @abc.abstractmethod
    def get_full_file_path(id_: str, root_dir: str, file_key, file_extension: str) -> str:
        """Gets the full file path for a data file.

        Args:
            id_ (str): The data's identification.
            root_dir (str): The data file's root directory.
            file_key (object): A human readable identifier used to identify the data file.
            file_extension (str): The data's file extension.

        Returns:
            str: The data's full file path.
        """
        raise NotImplementedError()


class DirectoryFilter(metaclass=abc.ABCMeta):
    """Represents an abstract directory filter.

    This class is used in  :py:class:`FileSystemDataCrawler` to filter a list of directories.
    """

    @staticmethod
    @abc.abstractmethod
    def filter_directories(dirs: typing.List[str]) -> typing.List[str]:
        """Filters a list of directories.

        Args:
            dirs (List[str]): A list of directories.

        Returns:
            List[str]: The filtered list of directories.
        """
        raise NotImplementedError()
