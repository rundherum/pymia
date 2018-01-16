import abc

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


class SitkLoader(Loader):

    def load_image(self, file_name: str, id_: str=None):
        return sitk.ReadImage(file_name, sitk.sitkFloat32)

    def load_label(self, file_name: str, id_: str=None):
        return sitk.ReadImage(file_name, sitk.sitkUInt8)

    def get_ndarray(self, image):
        return sitk.GetArrayFromImage(image)
