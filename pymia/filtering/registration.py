"""The registration module provides classes for image registration.

See Also:
    - `ITK Registration <https://itk.org/Doxygen/html/RegistrationPage.html>`_

    - `ITK Software Guide Registration <https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch3.html>`_
"""
import abc
import enum
import os
import typing

import SimpleITK as sitk

import pymia.filtering.filter as pymia_fltr


class RegistrationType(enum.Enum):
    """Represents the registration transformation type."""
    AFFINE = 1
    SIMILARITY = 2
    RIGID = 3
    BSPLINE = 4


class RegistrationCallback(abc.ABC):

    def __init__(self) -> None:
        """Represents the abstract handler for the registration callbacks."""
        self.registration_method = None
        self.fixed_image = None
        self.moving_image = None
        self.transform = None

    def set_params(self, registration_method: sitk.ImageRegistrationMethod,
                   fixed_image: sitk.Image,
                   moving_image: sitk.Image,
                   transform: sitk.Transform):
        """Sets the parameters that might be used during the callbacks

        Args:
            registration_method (sitk.ImageRegistrationMethod): The registration method.
            fixed_image (sitk.Image): The fixed image.
            moving_image (sitk.Image): The moving image.
            transform (sitk.Transform): The transformation.
        """
        self.registration_method = registration_method
        self.fixed_image = fixed_image
        self.moving_image = moving_image
        self.transform = transform
        # link the callback functions to the events
        self.registration_method.AddCommand(sitk.sitkStartEvent, self.registration_started)
        self.registration_method.AddCommand(sitk.sitkEndEvent, self.registration_ended)
        self.registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                                            self.registration_resolution_changed)
        self.registration_method.AddCommand(sitk.sitkIterationEvent, self.registration_iteration_ended)

    def registration_ended(self):
        """Callback for the EndEvent."""
        pass

    def registration_started(self):
        """Callback for the StartEvent."""
        pass

    def registration_resolution_changed(self):
        """Callback for the MultiResolutionIterationEvent."""
        pass

    def registration_iteration_ended(self):
        """Callback for the IterationEvent."""
        pass


class MultiModalRegistrationParams(pymia_fltr.FilterParams):

    def __init__(self, fixed_image: sitk.Image, fixed_image_mask: sitk.Image = None,
                 callbacks: typing.List[RegistrationCallback] = None):
        """Represents parameters for the multi-modal rigid registration used by the :class:`.MultiModalRegistration` filter.

        Args:
            fixed_image (sitk.Image): The fixed image for the registration.
            fixed_image_mask (sitk.Image): A mask for the fixed image to limit the registration.
            callbacks (t.List[RegistrationCallback]): Path to the directory where to plot the registration
                progress if any. Note that this increases the computational time.
        """

        self.fixed_image = fixed_image
        self.fixed_image_mask = fixed_image_mask
        self.callbacks = callbacks


class MultiModalRegistration(pymia_fltr.Filter):

    def __init__(self,
                 registration_type: RegistrationType = RegistrationType.RIGID,
                 number_of_histogram_bins: int = 200,
                 learning_rate: float = 1.0,
                 step_size: float = 0.001,
                 number_of_iterations: int = 200,
                 relaxation_factor: float = 0.5,
                 shrink_factors: typing.List[int] = (2, 1, 1),
                 smoothing_sigmas: typing.List[float] = (2, 1, 0),
                 sampling_percentage: float = 0.2,
                 sampling_seed: int = sitk.sitkWallClock,
                 resampling_interpolator=sitk.sitkBSpline):
        """Represents a multi-modal image registration filter.

        The filter estimates a 3-dimensional rigid or affine transformation between images of different modalities using
        - Mutual information similarity metric
        - Linear interpolation
        - Gradient descent optimization

        Args:
            registration_type (RegistrationType): The type of the registration ('rigid' or 'affine').
            number_of_histogram_bins (int): The number of histogram bins.
            learning_rate (float): The optimizer's learning rate.
            step_size (float): The optimizer's step size. Each step in the optimizer is at least this large.
            number_of_iterations (int): The maximum number of optimization iterations.
            relaxation_factor (float): The relaxation factor to penalize abrupt changes during optimization.
            shrink_factors (typing.List[int]): The shrink factors at each shrinking level (from high to low).
            smoothing_sigmas (typing.List[int]):  The Gaussian sigmas for smoothing at each shrinking level (in physical units).
            sampling_percentage (float): Fraction of voxel of the fixed image that will be used for registration (0, 1].
                Typical values range from 0.01 (1 %) for low detail images to 0.2 (20 %) for high detail images.
                The higher the fraction, the higher the computational time.
            sampling_seed: The seed for reproducible behavior.
            resampling_interpolator: Interpolation to be applied while resampling the image by the determined
                transformation.

        Examples:

            The following example shows the usage of the MultiModalRegistration class.

            >>> fixed_image = sitk.ReadImage('/path/to/image/fixed.mha')
            >>> moving_image = sitk.ReadImage('/path/to/image/moving.mha')
            >>> registration = MultiModalRegistration()  # specify parameters to your needs
            >>> parameters = MultiModalRegistrationParams(fixed_image)
            >>> registered_image = registration.execute(moving_image, parameters)
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
        self.sampling_seed = sampling_seed
        self.resampling_interpolator = resampling_interpolator

        registration = sitk.ImageRegistrationMethod()

        # similarity metric
        # will compare how well the two images match each other
        # registration.SetMetricAsJointHistogramMutualInformation(self.number_of_histogram_bins, 1.5)
        registration.SetMetricAsMattesMutualInformation(self.number_of_histogram_bins)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(self.sampling_percentage, self.sampling_seed)

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
        if self.registration_type == RegistrationType.BSPLINE:
            registration.SetOptimizerAsLBFGSB()
        else:
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
        self.transform = None

    def execute(self, image: sitk.Image, params: MultiModalRegistrationParams = None) -> sitk.Image:
        """Executes a multi-modal rigid registration.

        Args:
            image (sitk.Image): The moving image to register.
            params (MultiModalRegistrationParams): The parameters, which contain the fixed image.

        Returns:
            sitk.Image: The registered image.
        """

        if params is None:
            raise ValueError("params is not defined")
        dimension = image.GetDimension()
        if dimension not in (2, 3):
            raise ValueError('Image dimension {} is not among the accepted (2, 3)'.format(dimension))

        # set a transform that is applied to the moving image to initialize the registration
        if self.registration_type == RegistrationType.BSPLINE:
            transform_domain_mesh_size = [10] * image.GetDimension()
            initial_transform = sitk.BSplineTransformInitializer(params.fixed_image, transform_domain_mesh_size)
        else:
            if self.registration_type == RegistrationType.RIGID:
                transform_type = sitk.VersorRigid3DTransform() if dimension == 3 else sitk.Euler2DTransform()
            elif self.registration_type == RegistrationType.AFFINE:
                transform_type = sitk.AffineTransform(dimension)
            elif self.registration_type == RegistrationType.SIMILARITY:
                transform_type = sitk.Similarity3DTransform() if dimension == 3 else sitk.Similarity2DTransform()
            else:
                raise ValueError('not supported registration_type')

            initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(params.fixed_image,
                                                                            image.GetPixelIDValue()),
                                                                  image,
                                                                  transform_type,
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)

        self.registration.SetInitialTransform(initial_transform, inPlace=True)

        if params.fixed_image_mask:
            self.registration.SetMetricFixedMask(params.fixed_image_mask)

        if params.callbacks is not None:
            for callback in params.callbacks:
                callback.set_params(self.registration, params.fixed_image, image, initial_transform)

        self.transform = self.registration.Execute(sitk.Cast(params.fixed_image, sitk.sitkFloat32),
                                                   sitk.Cast(image, sitk.sitkFloat32))

        if self.verbose:
            print('MultiModalRegistration:\n Final metric value: {0}'.format(self.registration.GetMetricValue()))
            print(' Optimizer\'s stopping condition, {0}'.format(
                self.registration.GetOptimizerStopConditionDescription()))
        elif self.number_of_iterations == self.registration.GetOptimizerIteration():
            print('MultiModalRegistration: Optimizer terminated at number of iterations and did not converge!')

        return sitk.Resample(image, params.fixed_image, self.transform, self.resampling_interpolator, 0.0,
                             image.GetPixelIDValue())

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
               ' resampling_interpolator:  {self.resampling_interpolator}\n' \
            .format(self=self)


class PlotOnResolutionChangeCallback(RegistrationCallback):

    def __init__(self, plot_dir: str, file_name_prefix: str = '') -> None:
        """Represents a plotter for registrations.

        Saves the moving image on each resolution change and the registration end.

        Args:
            plot_dir (str): Path to the directory where to save the plots.
            file_name_prefix (str): The file name prefix for the plots.
        """
        super().__init__()
        self.plot_dir = plot_dir
        self.file_name_prefix = file_name_prefix

        self.resolution = 0

    def registration_ended(self):
        """Callback for the EndEvent."""
        self._write_image('end')

    def registration_started(self):
        """Callback for the StartEvent."""
        self.resolution = 0

    def registration_resolution_changed(self):
        """Callback for the MultiResolutionIterationEvent."""
        self._write_image('res' + str(self.resolution))
        self.resolution = self.resolution + 1

    def registration_iteration_ended(self):
        """Callback for the IterationEvent."""

    def _write_image(self, file_name_suffix: str):
        """Writes an image."""
        file_name = os.path.join(self.plot_dir, self.file_name_prefix + '_' + file_name_suffix + '.mha')

        moving_transformed = sitk.Resample(self.moving_image, self.fixed_image, self.transform,
                                           sitk.sitkLinear, 0.0, self.moving_image.GetPixelIDValue())
        sitk.WriteImage(moving_transformed, file_name)
