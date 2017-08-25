"""This module contains classes to set up a filtering pipeline.

All modules in the filter package implement the basic IFilter interface and can be used to set up a pipeline.
"""
from abc import ABCMeta, abstractmethod
from typing import List

import SimpleITK as sitk


class IFilterParams(metaclass=ABCMeta):
    """Represents a filter parameters interface."""


class IFilter(metaclass=ABCMeta):
    """Filter base class."""

    def __init__(self):
        """Initializes a new instance of the IFilter class."""
        self.verbose = False

    @abstractmethod
    def execute(self, image: sitk.Image, params: IFilterParams=None) -> sitk.Image:
        """Executes a filter on an image.

        Args:
            image (sitk.Image): The image.
            params (IFilterParams): The filter parameters.

        Returns:
            sitk.Image: The filtered image.
        """
        raise NotImplementedError()


class FilterPipeline:
    """Represents a filter pipeline, which can be executed on images."""

    def __init__(self, filters: List[IFilter]=None):
        """Initializes a new instance of the `FilterPipeline` class.

        Args:
            filters (list of IFilter): The filters.
        """
        self.params = []  # holds image-specific parameters
        self.filters = []  # holds the `IFilter`s
        if filters is not None:
            for filter_ in filters:
                self.add_filter(filter_)

    def add_filter(self, filter_: IFilter):
        """Add a filter to the pipeline.

        Args:
            filter_ (IFilter): A filter.
        """
        if filter_ is None:
            raise ValueError("The parameter filter needs to be specified.")

        self.filters.append(filter_)
        self.params.append(None)  # params must have the same length as filters

    def set_param(self, params, filter_index: int):
        """Sets an image-specific parameter for a filter.

        Args:
            params (IFilterParams): The parameter(s).
            filter_index (int): The filter's index the parameters belong to.
        """
        self.params[filter_index] = params

    def execute(self, image: sitk.Image) -> sitk.Image:
        """Executes the filter pipeline on an image.

        Args:
            image (sitk.Image): The image.

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
