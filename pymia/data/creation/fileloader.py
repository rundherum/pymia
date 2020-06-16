import abc
import typing

import numpy as np
import SimpleITK as sitk

import pymia.data.conversion as conv


class Load(abc.ABC):

    @abc.abstractmethod
    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) \
            -> typing.Tuple[np.ndarray, typing.Union[conv.ImageProperties, None]]:
        pass


class LoadDefault(Load):

    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) -> \
            typing.Tuple[np.ndarray, typing.Union[conv.ImageProperties, None]]:
        img = sitk.ReadImage(file_name)
        return sitk.GetArrayFromImage(img), conv.ImageProperties(img)
