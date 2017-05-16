import SimpleITK as sitk
from preprocessing.IPreProcessFilter import IPreProcessFilter
from preprocessing.IPreProcessFilterParams import IPreProcessFilterParams


class GradientAnisotropicDiffusion(IPreProcessFilter):
    """
    Represents a gradient anisotropic diffusion filter.
    """
    def __init__(self,
                 time_step: float=0.125,
                 conductance: int=3,
                 conductance_scaling_update_interval: int=1,
                 no_iterations: int=5):
        """
        :param time_step: 
        :param conductance: (the higher the smoother the edges).
        :param conductance_scaling_update_interval: 
        :param no_iterations: 
        """
        self.time_step = time_step
        self.conductance = conductance
        self.conductance_scaling_update_interval = conductance_scaling_update_interval
        self.no_iterations = no_iterations

    def execute(self, image: sitk.Image, params: IPreProcessFilterParams=None) -> sitk.Image:
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
        Get a nicely printable string representation.
        :return: String representation.
        """
        return 'GradientAnisotropicDiffusion:\n' \
               ' time_step:                           {self.time_step}\n' \
               ' conductance:                         {self.conductance}\n' \
               ' conductance_scaling_update_interval: {self.conductance_scaling_update_interval}\n' \
               ' no_iterations:                       {self.no_iterations}\n' \
            .format(self=self)
