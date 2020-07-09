import abc
import typing

import numpy as np
import SimpleITK as sitk

import pymia.data.conversion as conv


class Load(abc.ABC):
    """Interface for loading the data during the dataset creation in :meth:`.Traverser.traverse`
    
    .. automethod:: __call__
    """

    @abc.abstractmethod
    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) \
            -> typing.Tuple[np.ndarray, typing.Union[conv.ImageProperties, None]]:
        """Loads the data from the file system according to the implementation.

        Args:
            file_name (str): Path to the corresponding data.
            id_ (str): Identifier for the entry of the category, e.g, "Flair".
            category (str): Name of the category, e.g., 'images'.
            subject_id (str): Identifier of the current subject.

        Returns:
            tuple: A numpy array containing the loaded data and :class:`.ImageProperties` describing the data.
            :class:`.ImageProperties` is :code:`None` if the loaded data does not contain further properties.
        """
        pass


class LoadDefault(Load):
    """The default loader.

     It loads every data item (id/entry, category) for each subject as :code:`sitk.Image`
     and the corresponding :class:`ImageProperties`.
    """

    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) -> \
            typing.Tuple[np.ndarray, typing.Union[conv.ImageProperties, None]]:
        img = sitk.ReadImage(file_name)
        return sitk.GetArrayFromImage(img), conv.ImageProperties(img)
