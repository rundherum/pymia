import abc

import numpy as np
import SimpleITK as sitk


class Loader(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def load_image(self, file_name: str, id_: str=None):
        pass

    @abc.abstractmethod
    def load_label(self, file_name: str, id_: str=None):
        pass

    @abc.abstractmethod
    def get_ndarray(self, image):
        pass


class NumPyLoader(Loader):
    """A NumPy loader."""

    def load_image(self, file_name: str, id_: str=None):
        return np.load(file_name)

    def load_label(self, file_name: str, id_: str=None):
        return np.load(file_name)

    def get_ndarray(self, image):
        return image


class SitkLoader(Loader):

    def load_image(self, file_name: str, id_: str=None):
        return sitk.ReadImage(file_name, sitk.sitkFloat32)

    def load_label(self, file_name: str, id_: str=None):
        return sitk.ReadImage(file_name, sitk.sitkUInt8)

    def get_ndarray(self, image):
        return sitk.GetArrayFromImage(image)
