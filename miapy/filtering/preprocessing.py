"""Enables the enhancement of images before their use with other algorithms."""
import SimpleITK as sitk
from miapy.filtering.filter import IFilter, IFilterParams


class BiasFieldCorrectorParams(IFilterParams):
    """
    Bias field correction filter parameters.
    """

    def __init__(self, mask):
        """
        Initializes a new instance of the BiasFieldCorrectorParams class.

        :param mask: A mask image (0=background; 1=mask).

        Example:
        To generate a default mask use Otsu's thresholding:
        >>> sitk.OtsuThreshold( image, 0, 1, 200 )
        """
        self.mask = mask


class BiasFieldCorrector(IFilter):
    """
    Represents a bias field correction filter.
    """
    def __init__(self, shrink_factor=4, number_of_iterations=4, number_of_fitting_levels=3):
        """
        Initializes a new instance of the BiasFieldCorrector class.

        :param shrink_factor: 
        :param number_of_iterations: 
        :param number_of_fitting_levels: 
        """
        super().__init__()
        self.shrink_factor = shrink_factor
        self.number_of_iterations = number_of_iterations
        self.number_of_fitting_levels = number_of_fitting_levels

    def execute(self, image: sitk.Image, params: BiasFieldCorrectorParams=None) -> sitk.Image:
        """
        Executes a bias field correction on an image.

        :param image: The image.
        :param params: The bias field correction filter parameters.
        :return: The bias field corrected image.
        """

        if params is None:
            raise ValueError("params need to be provided")

        return sitk.N4BiasFieldCorrection(image,
                                          params.mask, self.shrink_factor,
                                          self.number_of_iterations,
                                          self.number_of_fitting_levels)

        # convergenceThreshold = 0.001, VectorUInt32
        # maximumNumberOfIterations, double
        # biasFieldFullWidthAtHalfMaximum = 0.15, double
        # wienerFilterNoise = 0.01, uint32_t
        # numberOfHistogramBins = 200, VectorUInt32
        # numberOfControlPoints, uint32_t
        # splineOrder = 3) -> Image

    def __str__(self):
        """
        Gets a nicely printable string representation.

        :return: String representation.
        """
        return 'BiasFieldCorrector:\n' \
               ' shrink_factor:            {self.shrink_factor}\n' \
               ' number_of_iterations:     {self.number_of_iterations}\n' \
               ' number_of_fitting_levels: {self.number_of_fitting_levels}\n' \
            .format(self=self)


class GradientAnisotropicDiffusion(IFilter):
    """
    Represents a gradient anisotropic diffusion filter.
    """
    def __init__(self,
                 time_step: float=0.125,
                 conductance: int=3,
                 conductance_scaling_update_interval: int=1,
                 no_iterations: int=5):
        """
        Initializes a new instance of the GradientAnisotropicDiffusion class.

        :param time_step: 
        :param conductance: (the higher the smoother the edges).
        :param conductance_scaling_update_interval: 
        :param no_iterations: 
        """
        super().__init__()
        self.time_step = time_step
        self.conductance = conductance
        self.conductance_scaling_update_interval = conductance_scaling_update_interval
        self.no_iterations = no_iterations

    def execute(self, image: sitk.Image, params: IFilterParams=None) -> sitk.Image:
        """
        Executes a gradient anisotropic diffusion on an image.

        :param image: The image.
        :param params: The gradient anisotropic diffusion filter parameters.
        :return: The smoothed image.
        """
        return sitk.GradientAnisotropicDiffusion(image,
                                                 self.time_step,
                                                 self.conductance,
                                                 self.conductance_scaling_update_interval,
                                                 self.no_iterations)

    def __str__(self):
        """
        Gets a nicely printable string representation.

        :return: String representation.
        """
        return 'GradientAnisotropicDiffusion:\n' \
               ' time_step:                           {self.time_step}\n' \
               ' conductance:                         {self.conductance}\n' \
               ' conductance_scaling_update_interval: {self.conductance_scaling_update_interval}\n' \
               ' no_iterations:                       {self.no_iterations}\n' \
            .format(self=self)


class RescaleIntensity(IFilter):
    """
    Represents a rescale intensity filter.
    """
    def __init__(self, min_intensity, max_intensity):
        """
        Initializes a new instance of the RescaleIntensity class.

        :param min_intensity: The min intensity value.
        :param max_intensity: The max intensity value.
        """
        super().__init__()
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def execute(self, image: sitk.Image, params: IFilterParams=None) -> sitk.Image:
        """
        Executes an intensity rescaling on an image.

        :param image: The image.
        :param params: The intensity rescaling filter parameters (None since no parameters are used).
        :return: The intensity rescaled image.
        """

        return sitk.RescaleIntensity(image, self.min_intensity, self.max_intensity)

    def __str__(self):
        """
        Gets a nicely printable string representation.

        :return: String representation.
        """
        return 'RescaleIntensity:\n' \
               ' min_intensity: {self.min_intensity}\n' \
               ' max_intensity: {self.max_intensity}\n' \
            .format(self=self)


class HistogramMatcher:
    """A learning method to align the intensity ranges of images."""
