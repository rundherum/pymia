import SimpleITK as sitk
from preprocessing.IPreProcessFilter import IPreProcessFilter
from preprocessing.IPreProcessFilterParams import IPreProcessFilterParams


class RescaleIntensity(IPreProcessFilter):
    """
    Represents a rescale intensity filter.
    """
    def __init__(self, min_intensity, max_intensity):
        """
        Initializes a new instance of the RescaleIntensity class.
        :param min_intensity: The min intensity value.
        :param max_intensity: The max intensity value.
        """
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def execute(self, image: sitk.Image, params: IPreProcessFilterParams=None) -> sitk.Image:
        """
        Executes an intensity rescaling on an image. 
        :param image: The image.
        :param params: The intensity rescaling filter parameters (None since no parameters are used).
        :return: The intensity rescaled image.
        """

        return sitk.RescaleIntensity(image, self.min_intensity, self.max_intensity)

    def __str__(self):
        """
        Get a nicely printable string representation.
        :return: String representation.
        """
        return 'BiasFieldCorrector:\n' \
               ' min_intensity: {self.min_intensity}\n' \
               ' max_intensity: {self.max_intensity}\n' \
            .format(self=self)
