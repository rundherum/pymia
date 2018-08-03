import abc
import typing as t

import numpy as np
import SimpleITK as sitk

import pymia.data.conversion as conv
import pymia.data.subjectfile as subj


class Load(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) \
            -> t.Tuple[np.ndarray, t.Union[conv.ImageProperties, None]]:
        pass


class LoadDefault(Load):

    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) -> \
            t.Tuple[np.ndarray, t.Union[conv.ImageProperties, None]]:
        img = sitk.ReadImage(file_name)
        return sitk.GetArrayFromImage(img), conv.ImageProperties(img)



