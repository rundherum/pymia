"""Enables the enhancement of images before their use with other algorithms."""
import SimpleITK as sitk

import miapy.filtering.filter as miapy_fltr


class BiasFieldCorrectorParams(miapy_fltr.IFilterParams):
    """Bias field correction filter parameters."""

    def __init__(self, mask: sitk.Image):
        """Initializes a new instance of the BiasFieldCorrectorParams class.

        Args:
            mask (sitk.Image): A mask image (0=background; 1=mask).

        Examples:
        To generate a default mask use Otsu's thresholding:
        >>> sitk.OtsuThreshold( image, 0, 1, 200 )
        """
        self.mask = mask


class BiasFieldCorrector(miapy_fltr.IFilter):
    """Represents a bias field correction filter."""

    def __init__(self, shrink_factor=1, convergence_threshold=0.001, max_iterations=(50, 50, 50, 50),
                 fullwidth_at_halfmax = 0.15, fiter_noise=0.01, histogram_bins=200, control_points=(4,4,4),
                 spline_order=3):
        """Initializes a new instance of the BiasFieldCorrector class.

        Args:
            shrink_factor (float): The shrink factor. A higher factor decreases the computational time.
            convergence_threshold (float): The threshold to stop the optimizer.
            max_iterations (list of int): The maximum number of optimizer iterations at each level.
            fullwidth_at_halfmax (float): ?
            fiter_noise (float): Wiener filter noise.
            histogram_bins (int): Number of histogram bins.
            control_points (list of int): The number of spline control points.
            spline_order (int): The spline order.
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
        """Executes a bias field correction on an image.

        Args:
            image (sitk.Image): The image.
            params (BiasFieldCorrectorParams): The bias field correction filter parameters.

        Returns:
            sitk.Image: The bias field corrected image.
        """

        mask = params.mask if params is not None else sitk.OtsuThreshold(image, 0, 1, 200)
        if self.shrink_factor > 1:
            raise ValueError('shrinking is not supported yet')
        return sitk.N4BiasFieldCorrection(image, mask, self.convergence_threshold, self.max_iterations,
                                          self.fullwidth_at_halfmax, self.filter_noise, self.histogram_bins,
                                          self.control_points, self.spline_order)

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'BiasFieldCorrector:\n' \
               ' shrink_factor:         {self.shrink_factor}\n' \
               ' convergence_threshold: {self.convergence_threshold}\n' \
               ' max_iterations:        {self.max_iterations}\n' \
               ' fullwidth_at_halfmax:  {self.fullwidth_at_halfmax}\n' \
               ' filter_noise:          {self.filter_noise}\n' \
               ' histogram_bins:        {self.histogram_bins}\n' \
               ' control_points:        {self.control_points}\n' \
               ' spline_order:          {self.spline_order}\n' \
            .format(self=self)


class GradientAnisotropicDiffusion(miapy_fltr.IFilter):
    """Represents a gradient anisotropic diffusion filter."""

    def __init__(self,
                 time_step: float=0.125,
                 conductance: int=3,
                 conductance_scaling_update_interval: int=1,
                 no_iterations: int=5):
        """Initializes a new instance of the GradientAnisotropicDiffusion class.

        Args:
            time_step (float): The time step.
            conductance (int): The conductance (the higher the smoother the edges).
            conductance_scaling_update_interval: TODO
            no_iterations (int): Number of iterations.
        """
        super().__init__()
        self.time_step = time_step
        self.conductance = conductance
        self.conductance_scaling_update_interval = conductance_scaling_update_interval
        self.no_iterations = no_iterations

    def execute(self, image: sitk.Image, params: miapy_fltr.IFilterParams=None) -> sitk.Image:
        """Executes a gradient anisotropic diffusion on an image.

        Args:
            image (sitk.Image): The image.
            params (miapy_fltr.IFilterParams): The parameters (unused).

        Returns:
            sitk.Image: The smoothed image.
        """
        return sitk.GradientAnisotropicDiffusion(sitk.Cast(image, sitk.sitkFloat32),
                                                 self.time_step,
                                                 self.conductance,
                                                 self.conductance_scaling_update_interval,
                                                 self.no_iterations)

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'GradientAnisotropicDiffusion:\n' \
               ' time_step:                           {self.time_step}\n' \
               ' conductance:                         {self.conductance}\n' \
               ' conductance_scaling_update_interval: {self.conductance_scaling_update_interval}\n' \
               ' no_iterations:                       {self.no_iterations}\n' \
            .format(self=self)


class RescaleIntensity(miapy_fltr.IFilter):
    """Represents a rescale intensity filter."""

    def __init__(self, min_intensity, max_intensity):
        """Initializes a new instance of the RescaleIntensity class.

        Args:
            min_intensity (int): The min intensity value.
            max_intensity (int): The max intensity value.
        """
        super().__init__()
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def execute(self, image: sitk.Image, params: miapy_fltr.IFilterParams=None) -> sitk.Image:
        """Executes an intensity rescaling on an image.

        Args:
            image (sitk.Image): The image.
            params (miapy_fltr.IFilterParams): The parameters (unused).

        Returns:
            sitk.Image: The intensity rescaled image.
        """

        return sitk.RescaleIntensity(image, self.min_intensity, self.max_intensity)

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'RescaleIntensity:\n' \
               ' min_intensity: {self.min_intensity}\n' \
               ' max_intensity: {self.max_intensity}\n' \
            .format(self=self)


class HistogramMatcherParams(miapy_fltr.IFilterParams):
    """Histogram matching filter parameters."""

    def __init__(self, reference_image: sitk.Image):
        """Initializes a new instance of the HistogramMatcherParams class.

        Args:
            reference_image (sitk.Image): Reference image for the matching.
        """
        self.reference_image = reference_image


class HistogramMatcher(miapy_fltr.IFilter):
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
            params (HistogramMatcherParams): The parameters.

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
               ' histogram_levels:         {self.histogram_levels}\n' \
               ' match_points:             {self.match_points}\n' \
               ' threshold_mean_intensity: {self.threshold_mean_intensity}\n' \
            .format(self=self)
