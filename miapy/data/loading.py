"""The loading module holds classes to load data."""
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import List
import os


class FilePathGenerator(metaclass=ABCMeta):
    """Represents an abstract file path generator.

    This class is used in :py:class:`FileSystemDataCrawler` to convert a human readable data identifier to an data file path,
    which allows to load the data.
    """

    @staticmethod
    @abstractmethod
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


class DirectoryFilter(metaclass=ABCMeta):
    """Represents an abstract directory filter.

    This class is used in  :py:class:`FileSystemDataCrawler` to filter a list of directories.
    """

    @staticmethod
    @abstractmethod
    def filter_directories(dirs: List[str]) -> List[str]:
        """Filters a list of directories.

        Args:
            dirs (List[str]): A list of directories.

        Returns:
            List[str]: The filtered list of directories.
        """
        raise NotImplementedError()


class FileSystemDataCrawler:
    """Represents a file system data crawler.

    todo(fabianbalsiger): finish doc

    Examples:
        Suppose we have the following directory structure::

            path/to/root_dir
                ./Patient1
                    ./Image.mha
                    ./GroundTruth.mha
                ./Patient2
                    ./Image.mha
                    ./GroundTruth.mha
                ./Atlas
                    ./Atlas.mha

        We can use the following code to load the images in the directories `Patient1` and `Patient2`:

        >>> class MyImgType(Enum):
        >>>     T1 = 1
        >>>     GroundTruth = 2
        >>>
        >>> class MyFilePathGenerator(FilePathGenerator):
        >>>     @staticmethod
        >>>     def get_full_file_path(id_: str, root_dir: str, file_key, file_extension: str) -> str:
        >>>         if file_key == MyImgType.T1:
        >>>             file_name = 'Image'
        >>>         elif file_key == MyImgType.GroundTruth:
        >>>             file_name = 'GroundTruth'
        >>>         else:
        >>>             raise ValueError('Unknown key')
        >>>
        >>>         return os.path.join(root_dir, file_name + file_extension)
        >>>
        >>> class MyDirFilter(DirectoryFilter):
        >>>     @staticmethod
        >>>     def filter_directories(dirs: List[str]) -> List[str]:
        >>>         return [dir for dir in dirs if dir.lower().__contains__('patient')]
        >>>
        >>> crawler = FileSystemDataCrawler('path/to/root_dir',
        >>>                                 [MyImgType.T1, MyImgType.GroundTruth],
        >>>                                 dir_filter=MyDirFilter(),
        >>>                                 file_extension='.mha')
        >>> for id_, path in crawler.data.items():
        >>>     print(id_, path)
        Patient1 path/to/root_dir/Patient1
        Patient2 path/to/root_dir/Patient2
    """
    
    def __init__(self,
                 root_dir: str,
                 file_keys: list,
                 file_path_generator: FilePathGenerator,
                 dir_filter: DirectoryFilter=None,
                 file_extension: str='.nii.gz'):
        """Initializes a new instance of the FileSystemDataCrawler class.

        Args:
            root_dir (str): The path to the root directory, which contains subdirectories with the data.
            file_keys (list): A list of objects, which represent human readable data identifiers
                (one identifier for each data file to crawl).
            file_path_generator (FilePathGenerator): A file path generator, which converts a human readable
                data identifier to an data file path.
            dir_filter (DirectoryFilter): A directory filter, which filters a list of directories.
            file_extension (str): The data file extension (with or without dot).
        """
        super().__init__()

        self.root_dir = root_dir
        self.dir_filter = dir_filter
        self.file_keys = file_keys
        self.file_path_generator = file_path_generator
        self.file_extension = file_extension if file_extension.startswith('.') else '.' + file_extension

        # dict with key=id (i.e, directory name), value=path to data directory
        self.data = {}  # dict with key=id (i.e, directory name), value=dict with key=file_keys and value=path to file

        data_dir = self._crawl_directories()
        self._crawl_data(data_dir)

    def _crawl_data(self, data_dir: dict):
        """Crawls the data inside a directory."""

        for id_, path in data_dir.items():
            data_dict = {id_: path}  # init dict with id_ pointing to path
            for item in self.file_keys:
                file_path = self.file_path_generator.get_full_file_path(id_, path, item, self.file_extension)
                data_dict[item] = file_path

            self.data[id_] = data_dict

    def _crawl_directories(self):
        """Crawls the directories, which contain data.

        Returns:
            dict: A dictionary where the keys are the directory names and the values the full path to the directory.
        """

        if not os.path.isdir(self.root_dir):
            raise ValueError('root_dir should point to an existing directory')

        # search the root directory for data directories
        data_dirs = next(os.walk(self.root_dir))[1]

        if self.dir_filter:
            # filter the data directories
            data_dirs = self.dir_filter.filter_directories(data_dirs)

        return {
            data_dir: os.path.join(self.root_dir, data_dir)
            for data_dir in data_dirs
            if any(file.endswith(self.file_extension) for file  # check if directory contains data files
                   in os.listdir(os.path.join(self.root_dir, data_dir)))
        }
