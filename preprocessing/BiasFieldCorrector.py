import SimpleITK as sitk
from preprocessing.IPreProcessFilter import IPreProcessFilter
from preprocessing.IPreProcessFilterParams import IPreProcessFilterParams


class BiasFieldCorrectorParams(IPreProcessFilterParams):
    """
    Bias field correction filter parameters.
    """

    def __init__(self, mask):
        """
        Initializes a new instance of the BiasFieldCorrectorParams class.
        :param mask: A mask image (0=background; 1=mask).

        Example:
        To generate a default mask use Otsu thresholding:
        >>> sitk.OtsuThreshold( image, 0, 1, 200 )
        """
        self.mask = mask

class BiasFieldCorrector(IPreProcessFilter):
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
        sitk.N4BiasFieldCorrection()
        return sitk.N4BiasFieldCorrection(image,
                                          params.mask, self.shrink_factor,
                                   self.number_of_iterations, self.number_of_fitting_levels)


        # convergenceThreshold = 0.001, VectorUInt32
        # maximumNumberOfIterations, double
        # biasFieldFullWidthAtHalfMaximum = 0.15, double
        # wienerFilterNoise = 0.01, uint32_t
        # numberOfHistogramBins = 200, VectorUInt32
        # numberOfControlPoints, uint32_t
        # splineOrder = 3) -> Image

    def __str__(self):
        """
        Get a nicely printable string representation.
        :return: String representation.
        """
        return 'BiasFieldCorrector:\n' \
               ' shrink_factor:            {self.shrink_factor}\n' \
               ' number_of_iterations:     {self.number_of_iterations}\n' \
               ' number_of_fitting_levels: {self.number_of_fitting_levels}\n' \
            .format(self=self)
