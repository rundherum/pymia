"""The post-processing module contains classes for image filtering mostly applied after a classification.

Image post-processing aims to alter images such that they depict a desired representation.
"""
import SimpleITK as sitk

import miapy.filtering.filter as miapy_fltr


class LargestNConnectedComponents(miapy_fltr.IFilter):
    """Represents a largest N connected components filter.

    Extracts the largest N connected components from a label image.
    By default the N components will all have the value 1 in the output image.
    Use the `consecutive_component_labels` option such that the largest has value 1,
    the second largest has value 2, etc. Background is always assumed to be 0.
    """

    def __init__(self, number_of_components: int = 1, consecutive_component_labels: bool = False):
        """Initializes a new instance of the LargestNComponents class.

        Args:
            number_of_components (int): The number of largest components to extract.
            consecutive_component_labels (bool): The largest component has value 1, the second largest has value 2, ect.
                if set to True; otherwise, all components will have value 1.
        """
        super().__init__()

        if not number_of_components >= 1:
            raise ValueError("number_of_components must be larger or equal to 1")

        self.number_of_components = number_of_components
        self.consecutive_component_labels = consecutive_component_labels

    def execute(self, image: sitk.Image, params: miapy_fltr.IFilterParams = None) -> sitk.Image:
        """Executes the largest N connected components filter on an image.

        Args:
            image (sitk.Image): The image.
            params (IFilterParams): The parameters (unused).

        Returns:
            sitk.Image: The filtered image.
        """

        image = sitk.ConnectedComponent(image)
        image = sitk.RelabelComponent(image)

        if self.consecutive_component_labels:
            return sitk.Threshold(image, lower=1, upper=self.number_of_components, outsideValue=0)
        else:
            return sitk.BinaryThreshold(image, lowerThreshold=1, upperThreshold=self.number_of_components,
                                        insideValue=1, outsideValue=0)

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """

        return 'LargestNConnectedComponents:\n' \
               ' number_of_components:         {self.number_of_components}\n' \
               ' consecutive_component_labels: {self.consecutive_component_labels}\n' \
            .format(self=self)
