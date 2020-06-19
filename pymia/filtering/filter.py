"""This module provides classes to set up a filtering pipeline."""
import abc
import typing

import SimpleITK as sitk


class FilterParams(abc.ABC):
    """Represents a filter parameters interface."""


class Filter(abc.ABC):

    def __init__(self):
        """Filter base class."""
        self.verbose = False

    @abc.abstractmethod
    def execute(self, image: sitk.Image, params: FilterParams = None) -> sitk.Image:
        """Executes a filter on an image.

        Args:
            image (sitk.Image): The image to filter.
            params (FilterParams): The filter parameters.

        Returns:
            sitk.Image: The filtered image.
        """
        raise NotImplementedError()


class FilterPipeline:

    def __init__(self, filters: typing.List[Filter] = None):
        """Represents a filter pipeline, which sequentially executes filters (:class:`.Filter`) on an image.

        Args:
            filters (list of Filter): The filters of the pipeline.
        """
        self.filters = []  # holds the `Filter`s
        self.params = []  # holds image-specific parameters

        if filters is not None:
            for filter_ in filters:
                self.add_filter(filter_)

    def add_filter(self, filter_: Filter, params: FilterParams = None):
        """Adds a filter to the pipeline.

        Args:
            filter_ (Filter): A filter.
            params (FilterParams): The filter parameters.
        """
        self.filters.append(filter_)
        self.params.append(params)  # params must have the same length as filters

    def set_param(self, params: FilterParams, filter_index: int):
        """Sets an image-specific parameter for a filter.

        Use this function to update the parameters of a filter to be specific to the image to be filtered.

        Args:
            params (FilterParams): The parameter(s).
            filter_index (int): The filter's index the parameters belong to.
        """
        self.params[filter_index] = params

    def execute(self, image: sitk.Image) -> sitk.Image:
        """Executes the filter pipeline on an image.

        Args:
            image (sitk.Image): The image to filter.

        Returns:
            sitk.Image: The filtered image.
        """
        for param_index, filter_ in enumerate(self.filters):
            image = filter_.execute(image, self.params[param_index])

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """

        string = 'FilterPipeline:\n'

        for filter_no, filter_ in enumerate(self.filters):
            string += ' ' + str(filter_no + 1) + '. ' + '    '.join(str(filter_).splitlines(True))

        return string.format(self=self)
