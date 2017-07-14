"""We provide an easy way to set up a filtering pipeline. All modules in this package implement..."""
from typing import List

import SimpleITK as sitk
from abc import ABCMeta, abstractmethod


class IFilterParams(metaclass=ABCMeta):
    """Represents a filter parameters interface.
    """


class IFilter(metaclass=ABCMeta):
    """Filter base class.
    """

    def __init__(self):
        """Initializes a new instance of the IFilter class.
        """
        self.verbose = False

    @abstractmethod
    def execute(self, image: sitk.Image, params: IFilterParams=None) -> sitk.Image:
        """Executes a filter on an image.

        :param image: The image.
        :param params: The filter parameters.
        :return: The filtered image.
        """
        raise NotImplementedError()


class FilterPipeline:
    """Represents a filter pipeline, which can be executed on images.
    """

    def __init__(self, filters: List[IFilter]=None):
        """Initializes a new instance of the `FilterPipeline` class.

        :param filters: The filters.
        """
        self.params = []  # holds image-specific parameters
        self.filters = []  # holds the `IFilter`s
        if filters is not None:
            for filter_ in filters:
                self.add_filter(filter_)

    def add_filter(self, filter_: IFilter):

        if filter_ is None:
            raise ValueError("The parameter filter needs to be specified.")

        self.filters.append(filter_)
        self.params.append(None)  # params must have the same length as filters

    def set_param(self, params, filter_index):
        """Sets an image-specific parameter for a filter.

        :param params: The parameter(s).
        :param filter_index: The filter's index.
        """
        self.params[filter_index] = params

    def execute(self, image: sitk.Image) -> sitk.Image:
        """Executes the filter pipeline on an image.

        :param image: The image.
        :return: The filtered image.
        """
        for param_index, filter_ in enumerate(self.filters):
            image = filter_.execute(image, self.params[param_index])

        return image

    def __str__(self):
        """Gets a nicely printable string representation.

        :return: String representation.
        """

        string = 'FilterPipeline:\n'

        for filter_no, filter_ in enumerate(self.filters):
            string += " " + str(filter_no + 1) + ". " + "    ".join(str(filter_).splitlines(True))

        return string.format(self=self)
