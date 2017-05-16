import SimpleITK as sitk
from abc import ABCMeta, abstractmethod
from preprocessing.IPreProcessFilterParams import IPreProcessFilterParams


class IPreProcessFilter(metaclass=ABCMeta):
    """
    Pre-processing filter interface.
    """

    @abstractmethod
    def execute(self, image: sitk.Image, params: IPreProcessFilterParams=None) -> sitk.Image:
        """
        Executes a pre-processing filter on an image. 
        :param image: The image.
        :param params: The pre-processing parameters.
        :return: The pre-processed image.
        """
        raise NotImplementedError()
