"""Some doc about registration"""
import SimpleITK as sitk
import miapy.filtering.filter as fltr


class RigidMultiModalRegistrationParams(fltr.IFilterParams):

    def __init__(self, fixed_image: sitk.Image):
        """
        Initializes a new instance of the RigidMultiModalRegistrationParams class.

        :param fixed_image: The fixed image for the registration.
        :type fixed_image: sitk.Image
        """
        self.fixed_image = fixed_image


class RigidMultiModalRegistration(fltr.IFilter):
    """
    Represents a multi-modal image registration filter.

    The filter estimates a 3-dimensional rigid transformation between images of different modalities using
    - Mutual information similarity metric
    - Linear interpolation
    - Gradient descent optimization

    Example:

    >>> fixed_image = sitk.ReadImage("/path/to/image/fixed.mha")
    >>> moving_image = sitk.ReadImage("/path/to/image/moving.mha")
    >>> registration = RigidMultiModalRegistration()  # specify parameters to your needs
    >>> parameters = RigidMultiModalRegistrationParams(fixed_image)
    >>> registered_image = reg.execute(moving, parameters)
    """

    def __init__(self,
                 number_of_histogram_bins: int=200,
                 learning_rate: float=1.0,
                 step_size: float=0.001,
                 number_of_iterations: int=200,
                 shrink_factors: [int]=[4, 2, 1],
                 smoothing_sigmas: [float]=[2, 1, 0]):
        """
        Initializes a new instance of the MultiModalRegistration class.

        :param number_of_histogram_bins: The number of histogram bins.
        :type number_of_histogram_bins: int
        :param number_of_iterations: The maximum number of optimization iterations.
        :type number_of_iterations: int
        """

        if len(shrink_factors) != len(smoothing_sigmas):
            raise ValueError("shrink_factors and smoothing_sigmas need to be same length")

        self.number_of_histogram_bins = number_of_histogram_bins
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.number_of_iterations = number_of_iterations
        self.shrink_factors = shrink_factors
        self.smoothing_sigmas = smoothing_sigmas

        registration = sitk.ImageRegistrationMethod()

        # similarity metric
        # will compare how well the two images match each other
        registration.SetMetricAsMattesMutualInformation(self.number_of_histogram_bins)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.1)

        # interpolator
        # will evaluate the intensities of the moving image at non-rigid positions
        registration.SetInterpolator(sitk.sitkLinear)

        # optimizer
        # is required to explore the parameter space of the transform in search of optimal values of the metric
        registration.SetOptimizerAsRegularStepGradientDescent(self.learning_rate,
                                                              self.step_size,
                                                              self.number_of_iterations)

        # setup for the multi-resolution framework
        registration.SetShrinkFactorsPerLevel(self.shrink_factors)
        registration.SetSmoothingSigmasPerLevel(self.smoothing_sigmas)
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        self.registration = registration

    def execute(self, image: sitk.Image, params: RigidMultiModalRegistrationParams=None) -> sitk.Image:
        """
        Executes a rigid multi-modal registration.

        :param image: The moving image.
        :type image: sitk.Image
        :param params: The filter parameters.
        :type params: RigidMultiModalRegistrationParams.
        :return: The registered image.
        :rtype: sitk.Image
        """

        if params is None:
            raise ValueError("params need to be provided")

        initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(params.fixed_image, image.GetPixelIDValue()),
                                                              image,
                                                              sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
        self.registration.SetInitialTransform(initial_transform, inPlace=False)

        fixed_image = sitk.Normalize(params.fixed_image)
        image = sitk.Normalize(image)

        transform = self.registration.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                              sitk.Cast(image, sitk.sitkFloat32))

        if self.number_of_iterations == self.registration.GetOptimizerIteration():
            print("RigidMultiModalRegistration: Optimizer terminated at number of iterations and did not converge!")

        print('Final metric value: {0}'.format(self.registration.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(self.registration.GetOptimizerStopConditionDescription()))

        return sitk.Resample(image, params.fixed_image, transform, sitk.sitkLinear, 0.0, image.GetPixelIDValue())

    def __str__(self):
        """
        Gets a nicely printable string representation.

        :return: String representation.
        """

        return 'RigidMultiModalRegistration:\n' \
               ' number_of_histogram_bins: {self.number_of_histogram_bins}\n' \
               ' learning_rate:            {self.learning_rate}\n' \
               ' step_size:                {self.step_size}\n' \
               ' number_of_iterations:     {self.number_of_iterations}\n' \
               ' shrink_factors:           {self.shrink_factors}\n' \
               ' smoothing_sigmas:         {self.smoothing_sigmas}\n' \
            .format(self=self)
