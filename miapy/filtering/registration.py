"""The registration module contains classes for image registration.

Image registration aims to align two images using a particular transformation.
miapy currently supports multi-modal rigid registration, i.e. align two images of different modalities
using a rigid transformation (rotation, translation, reflection, or their combination).

See Also:
    `ITK Registration <https://itk.org/Doxygen/html/RegistrationPage.html>`_
    `ITK Software Guide Registration <https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch3.html>`_
"""
from enum import Enum

import matplotlib
matplotlib.use('Agg')  # use matplotlib without having a window appear
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

import miapy.filtering.filter as miapy_fltr


class RegistrationType(Enum):
    """Represents the registration transformation type."""
    AFFINE = 1
    RIGID = 2


class MultiModalRegistrationParams(miapy_fltr.IFilterParams):
    """Represents parameters for the multi-modal rigid registration."""

    def __init__(self, fixed_image: sitk.Image, fixed_image_mask: sitk.Image=None, plot_directory_path: str=''):
        """Initializes a new instance of the MultiModalRegistrationParams class.

        Args:
            fixed_image (sitk.Image): The fixed image for the registration.
            fixed_image_mask (sitk.Image): A mask for the fixed image to limit the registration.
            plot_directory_path (str): Path to the directory where to plot the registration progress if any.
                Note that this increases the computational time.
        """

        self.fixed_image = fixed_image
        self.fixed_image_mask = fixed_image_mask
        self.plot_directory_path = plot_directory_path


class MultiModalRegistration(miapy_fltr.IFilter):
    """Represents a multi-modal image registration filter.

    The filter estimates a 3-dimensional rigid or affine transformation between images of different modalities using
    - Mutual information similarity metric
    - Linear interpolation
    - Gradient descent optimization

    Examples:

    The following example shows the usage of the MultiModalRegistration class.

    >>> fixed_image = sitk.ReadImage('/path/to/image/fixed.mha')
    >>> moving_image = sitk.ReadImage('/path/to/image/moving.mha')
    >>> registration = MultiModalRegistration()  # specify parameters to your needs
    >>> parameters = MultiModalRegistrationParams(fixed_image)
    >>> registered_image = registration.execute(moving_image, parameters)
    """

    def __init__(self,
                 registration_type: RegistrationType=RegistrationType.RIGID,
                 number_of_histogram_bins: int=200,
                 learning_rate: float=1.0,
                 step_size: float=0.001,
                 number_of_iterations: int=200,
                 relaxation_factor: float=0.5,
                 shrink_factors: [int]=(2, 1, 1),
                 smoothing_sigmas: [float]=(2, 1, 0),
                 sampling_percentage: float=0.2):
        """Initializes a new instance of the MultiModalRegistration class.

        Args:
            registration_type (RegistrationType): The type of the registration ('rigid' or 'affine').
            number_of_histogram_bins (int): The number of histogram bins.
            learning_rate (float): The optimizer's learning rate.
            step_size (float): The optimizer's step size. Each step in the optimizer is at least this large.
            number_of_iterations (int): The maximum number of optimization iterations.
            relaxation_factor (float): The relaxation factor to penalize abrupt changes during optimization.
            shrink_factors ([int]): The shrink factors at each shrinking level (from high to low).
            smoothing_sigmas ([int]):  The Gaussian sigmas for smoothing at each shrinking level (in physical units).
            sampling_percentage (float): Fraction of voxel of the fixed image that will be used for registration (0, 1].
                Typical values range from 0.01 (1 %) for low detail images to 0.2 (20 %) for high detail images.
                The higher the fraction, the higher the computational time.
        """
        super().__init__()

        if len(shrink_factors) != len(smoothing_sigmas):
            raise ValueError("shrink_factors and smoothing_sigmas need to be same length")

        self.registration_type = registration_type
        self.number_of_histogram_bins = number_of_histogram_bins
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.number_of_iterations = number_of_iterations
        self.relaxation_factor = relaxation_factor
        self.shrink_factors = shrink_factors
        self.smoothing_sigmas = smoothing_sigmas
        self.sampling_percentage = sampling_percentage

        registration = sitk.ImageRegistrationMethod()

        # similarity metric
        # will compare how well the two images match each other
        # registration.SetMetricAsJointHistogramMutualInformation(self.number_of_histogram_bins, 1.5)
        registration.SetMetricAsMattesMutualInformation(self.number_of_histogram_bins)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(self.sampling_percentage)

        # An image gradient calculator based on ImageFunction is used instead of image gradient filters
        # set to True uses GradientRecursiveGaussianImageFilter
        # set to False uses CentralDifferenceImageFunction
        # see also https://itk.org/Doxygen/html/classitk_1_1ImageToImageMetricv4.html
        registration.SetMetricUseFixedImageGradientFilter(False)
        registration.SetMetricUseMovingImageGradientFilter(False)

        # interpolator
        # will evaluate the intensities of the moving image at non-rigid positions
        registration.SetInterpolator(sitk.sitkLinear)

        # optimizer
        # is required to explore the parameter space of the transform in search of optimal values of the metric
        registration.SetOptimizerAsRegularStepGradientDescent(learningRate=self.learning_rate,
                                                              minStep=self.step_size,
                                                              numberOfIterations=self.number_of_iterations,
                                                              relaxationFactor=self.relaxation_factor,
                                                              gradientMagnitudeTolerance=1e-4,
                                                              estimateLearningRate=registration.EachIteration,
                                                              maximumStepSizeInPhysicalUnits=0.0)
        registration.SetOptimizerScalesFromPhysicalShift()

        # setup for the multi-resolution framework
        registration.SetShrinkFactorsPerLevel(self.shrink_factors)
        registration.SetSmoothingSigmasPerLevel(self.smoothing_sigmas)
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        self.registration = registration

    def execute(self, image: sitk.Image, params: MultiModalRegistrationParams=None) -> sitk.Image:
        """Executes a multi-modal rigid registration.

        Args:
            image (sitk.Image): The moving image.
            params (MultiModalRegistrationParams): The parameters, which contain the fixed image.

        Returns:
            sitk.Image: The registered image.
        """

        if params is None:
            raise ValueError("params is not defined")

        # set a transform that is applied to the moving image to initialize the registration
        if self.registration_type == RegistrationType.RIGID:
            transform_type = sitk.VersorRigid3DTransform()
        elif self.registration_type == RegistrationType.AFFINE:
            transform_type = sitk.AffineTransform(3)
        else:
            raise ValueError('not supported registration_type')

        initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(params.fixed_image, image.GetPixelIDValue()),
                                                              image,
                                                              transform_type,
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
        self.registration.SetInitialTransform(initial_transform, inPlace=True)

        if params.fixed_image_mask:
            self.registration.SetMetricFixedMask(params.fixed_image_mask)

        if params.plot_directory_path:
            RegistrationPlotter(self.registration, params.fixed_image, image, initial_transform, params.plot_directory_path)

        transform = self.registration.Execute(sitk.Cast(params.fixed_image, sitk.sitkFloat32),
                                              sitk.Cast(image, sitk.sitkFloat32))

        if self.verbose:
            print('MultiModalRegistration:\n Final metric value: {0}'.format(self.registration.GetMetricValue()))
            print(' Optimizer\'s stopping condition, {0}'.format(
                self.registration.GetOptimizerStopConditionDescription()))
        elif self.number_of_iterations == self.registration.GetOptimizerIteration():
            print('MultiModalRegistration: Optimizer terminated at number of iterations and did not converge!')

        return sitk.Resample(image, params.fixed_image, transform, sitk.sitkLinear, 0.0, image.GetPixelIDValue())

    def __str__(self):
        """Gets a nicely printable string representation.

        Returns:
            str: The string representation.
        """

        return 'MultiModalRegistration:\n' \
               ' registration_type:        {self.registration_type}\n' \
               ' number_of_histogram_bins: {self.number_of_histogram_bins}\n' \
               ' learning_rate:            {self.learning_rate}\n' \
               ' step_size:                {self.step_size}\n' \
               ' number_of_iterations:     {self.number_of_iterations}\n' \
               ' relaxation_factor:        {self.relaxation_factor}\n' \
               ' shrink_factors:           {self.shrink_factors}\n' \
               ' smoothing_sigmas:         {self.smoothing_sigmas}\n' \
               ' sampling_percentage:      {self.sampling_percentage}\n' \
            .format(self=self)


class RegistrationPlotter:
    """Represents a plotter for SimpleITK registrations."""

    def __init__(self, registration: sitk.ImageRegistrationMethod,
                 fixed_image: sitk.Image,
                 image: sitk.Image,
                 transform: sitk.Transform,
                 path: str):
        """

        Args:
            registration (sitk.ImageRegistrationMethod): The registration method.
            fixed_image (sitk.Image): The fixed image.
            image (sitk.Image): The moving image.
            transform (sitk.Transform): The transformation.
            path (str): Path to the directory where to save the plots.
        """
        self.metric_values = []
        self.multires_iterations = []

        registration.AddCommand(sitk.sitkStartEvent, self._start_plot)
        registration.AddCommand(sitk.sitkEndEvent, self._end_plot)
        registration.AddCommand(sitk.sitkMultiResolutionIterationEvent, self._update_multiresolution_iterations)
        registration.AddCommand(sitk.sitkIterationEvent, lambda: self._save_plot(registration,
                                                                                 fixed_image,
                                                                                 image,
                                                                                 transform,
                                                                                 path))

    def _end_plot(self):
        """Callback for the EndEvent."""
        plt.close()

    def _start_plot(self):
        """Callback for the StartEvent."""
        self.metric_values = []
        self.multires_iterations = []

    def _update_multiresolution_iterations(self):
        """Callback for the MultiResolutionIterationEvent."""
        self.multires_iterations.append(len(self.metric_values))

    def _save_plot(self, registration_method, fixed, moving, transform, file_name_prefix):
        """Callback for the IterationEvent.

        Saves an image including the visualization of the registered images and the metric value plot.
        """

        self.metric_values.append(registration_method.GetMetricValue())
        # Plot the similarity metric values; resolution changes are marked with a blue star
        plt.plot(self.metric_values, 'r')
        plt.plot(self.multires_iterations, [self.metric_values[index] for index in self.multires_iterations], 'b*')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)

        # todo(fabianbalsiger): format precision of legends
        # plt.axes().yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:2f}'))
        # plt.axes().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:d}'))
        # _, ax = plt.subplots()
        # ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:f2}'))

        # todo(fabianbalsiger): add margin to the left side of the plot

        # Convert the plot to a SimpleITK image (works with the agg matplotlib backend, doesn't work
        # with the default - the relevant method is canvas_tostring_rgb())
        plt.gcf().canvas.draw()
        plot_data = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8, sep='')
        plot_data = plot_data.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        plot_image = sitk.GetImageFromArray(plot_data, isVector=True)

        # Extract the central axial slice from the two volumes, compose it using the transformation and alpha blend it
        alpha = 0.7

        central_index = round((fixed.GetSize())[2] / 2)

        moving_transformed = sitk.Resample(moving, fixed, transform,
                                           sitk.sitkLinear, 0.0,
                                           moving.GetPixelIDValue())
        # Extract the central slice in xy and alpha blend them
        combined = (1.0 - alpha) * sitk.Normalize(fixed[:, :, central_index]) + \
                   alpha * sitk.Normalize(moving_transformed[:, :, central_index])

        # Assume the alpha blended images are isotropic and rescale intensity
        # values so that they are in [0,255], convert the grayscale image to
        # color (r,g,b).
        combined_slices_image = sitk.Cast(sitk.RescaleIntensity(combined), sitk.sitkUInt8)
        combined_slices_image = sitk.Compose(combined_slices_image,
                                             combined_slices_image,
                                             combined_slices_image)

        self._write_combined_image(combined_slices_image, plot_image,
                                   file_name_prefix + format(len(self.metric_values), '03d') + '.png')

    def _write_combined_image(self, image1, image2, file_name):
        """Writes an image including the visualization of the registered images and the metric value plot."""
        combined_image = sitk.Image((image1.GetWidth() + image2.GetWidth(), max(image1.GetHeight(), image2.GetHeight())),
                                    image1.GetPixelID(), image1.GetNumberOfComponentsPerPixel())

        image1_destination = [0, 0]
        image2_destination = [image1.GetWidth(), 0]

        if image1.GetHeight() > image2.GetHeight():
            image2_destination[1] = round((combined_image.GetHeight() - image2.GetHeight()) / 2)
        else:
            image1_destination[1] = round((combined_image.GetHeight() - image1.GetHeight()) / 2)

        combined_image = sitk.Paste(combined_image, image1, image1.GetSize(), (0, 0), image1_destination)
        combined_image = sitk.Paste(combined_image, image2, image2.GetSize(), (0, 0), image2_destination)
        sitk.WriteImage(combined_image, file_name)
