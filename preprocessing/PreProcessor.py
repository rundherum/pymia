import SimpleITK as sitk
from preprocessing.IPreProcessFilter import IPreProcessFilter


class PreProcessor:
    """
    Represents a pre-processing pipeline, which can be executed on images.
    """

    def __init__(self):
        """
        Initializes a new instance of the PreProcessor class.
        """
        self.filters = []  # holds the IPreProcessFilters
        self.params = []  # holds image-specific parameters

    def add_filter(self, filter: IPreProcessFilter):

        if filter is None:
            raise ValueError("The parameter filter needs to be specified.")

        self.filters.append(filter)
        self.params.append(None)  # params must have the same length as filters

    def set_param(self, params, filter_index):
        """
        Sets an image-specific parameter for a filter.
        :param params: The parameter(s).
        :param filter_index: The filter's index.
        """
        self.params[filter_index] = params

    def execute(self, image: sitk.Image) -> sitk.Image:
        """
        Executes the pre-processing pipeline on an image.
        :param image: The image.
        :return: The pre-processed image.
        """
        for param_index, filter in enumerate(self.filters):
            image = filter.execute(image, self.params[param_index])

        return image

    def __str__(self):
        """
        Get a nicely printable string representation.
        :return: String representation.
        """

        string = 'PreProcessor:\n'

        for filter_no, filter in enumerate(self.filters):
            string += " " + str(filter_no + 1) + ". " + "    ".join(str(filter).splitlines(True))

        return string.format(self=self)
