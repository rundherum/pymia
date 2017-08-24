"""Enables the enhancement of images before their use with other algorithms."""
import SimpleITK as sitk
from miapy.filtering.filter import IFilter, IFilterParams


class BiasFieldCorrectorParams(IFilterParams):
    """
    Bias field correction filter parameters.
    """

    def __init__(self, mask: sitk.Image):
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
    def __init__(self, shrink_factor=1, convergence_threshold=0.001, max_iterations=(50, 50, 50, 50),
                 fullwidth_at_halfmax = 0.15, fiter_noise=0.01, histogram_bins=200, control_points=(4,4,4),
                 spline_order=3):
        """
        Initializes a new instance of the BiasFieldCorrector class.

        :param shrink_factor:
        :param convergence_threshold:
        :param max_iterations:
        :param fullwidth_at_halfmax:
        :param fiter_noise:
        :param histogram_bins:
        :param control_points:
        :param spline_order:
        """
        super().__init__()
        self.shrink_factor = shrink_factor
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.fullwidth_at_halfmax = fullwidth_at_halfmax
        self.filter_noise = fiter_noise
        self.histogram_bins = histogram_bins
        self.control_points = control_points
        self.spline_order = spline_order

    def execute(self, image: sitk.Image, params: BiasFieldCorrectorParams=None) -> sitk.Image:
        """
        Executes a bias field correction on an image.

        :param image: The image.
        :param params: The bias field correction filter parameters.
        :return: The bias field corrected image.
        """

        mask = params.mask if params is not None else sitk.OtsuThreshold(image, 0, 1, 200)
        if self.shrink_factor > 1:
            raise ValueError('shrinking is not supported yet')
        return sitk.N4BiasFieldCorrection(image, mask, self.convergence_threshold, self.max_iterations,
                                          self.fullwidth_at_halfmax, self.filter_noise, self.histogram_bins,
                                          self.control_points, self.spline_order)

    def __str__(self):
        """
        Gets a nicely printable string representation.

        :return: String representation.
        """
        return 'BiasFieldCorrector:\n' \
               ' shrink_factor:            {self.shrink_factor}\n' \
               ' convergence_threshold:    {self.convergence_threshold}\n' \
               ' max_iterations:           {self.max_iterations}\n' \
               ' fullwidth_at_halfmax:     {self.fullwidth_at_halfmax}\n' \
               ' filter_noise:             {self.filter_noise}\n' \
               ' histogram_bins:           {self.histogram_bins}\n' \
               ' control_points:           {self.control_points}\n' \
               ' spline_order:             {self.spline_order}\n' \
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


class HistogramMatcherParams(IFilterParams):
    """
    Histogram matching filter parameters.
    """

    def __init__(self, reference_image: sitk.Image):
        """Initializes a new instance of the HistogramMatcherParams class.

        Args:
            reference_image (sitk.Image): Reference image for the matching.
        """
        self.reference_image = reference_image


class HistogramMatcher(IFilter):
    """A method to align the intensity ranges of images."""

    def __init__(self, histogram_levels=256, match_points=1, threshold_mean_intensity=True):
        """Initializes a new instance of the HistogramMatcher class.

        Args:
            histogram_levels (int): Number of histogram levels.
            match_points (int): Number of match points.
            threshold_mean_intensity (bool): Threshold at mean intensity.
        """
        super().__init__()
        self.histogram_levels = histogram_levels
        self.match_points = match_points
        self.threshold_mean_intensity = threshold_mean_intensity

    def execute(self, image: sitk.Image, params: HistogramMatcherParams = None) -> sitk.Image:
        """Matches the image intensity histogram to a reference.

        Args:
            image (sitk.Image): The image.
            params (IFilterParams): The parameters (unused).

        Returns:
            sitk.Image: The filtered image.
        """
        if params is None:
            raise ValueError('Parameter with reference image is required')

        return sitk.HistogramMatching(image, params.reference_image, self.histogram_levels, self.match_points,
                                      self.threshold_mean_intensity)

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'HistogramMatcher:\n' \
               ' histogram_levels:           {self.histogram_levels}\n' \
               ' match_points:               {self.match_points}\n' \
               ' threshold_mean_intensity:   {self.threshold_mean_intensity}\n' \
            .format(self=self)

