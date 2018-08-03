"""This module holds classes related to image conversion.

The main purpose of this module is the conversion between SimpleITK images and numpy arrays.
"""
import typing

import SimpleITK as sitk
import numpy as np


class ImageProperties:
    """Represents ITK image properties.

    Holds common ITK image meta-data such as the size, origin, spacing, and direction.

    See Also:
        SimpleITK provides `itk::simple::Image::CopyInformation`_ to copy image information.

    .. _itk::simple::Image::CopyInformation:
        https://itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1Image.html#afa8a4757400c414e809d1767ee616bd0
    """

    def __init__(self, image: sitk.Image):
        """Initializes a new instance of the ImageProperties class.

        Args:
            image (sitk.Image): The image whose properties to hold.
        """
        self.size = image.GetSize()
        self.origin = image.GetOrigin()
        self.spacing = image.GetSpacing()
        self.direction = image.GetDirection()
        self.dimensions = image.GetDimension()
        self.number_of_components_per_pixel = image.GetNumberOfComponentsPerPixel()
        self.pixel_id = image.GetPixelID()

    def is_two_dimensional(self) -> bool:
        """Determines whether the image is two-dimensional.

        Returns:
            bool: True if the image is two-dimensional; otherwise, False.
        """
        return self.dimensions == 2

    def is_three_dimensional(self) -> bool:
        """Determines whether the image is three-dimensional.

        Returns:
            bool: True if the image is three-dimensional; otherwise, False.
        """
        return self.dimensions == 3

    def is_vector_image(self) -> bool:
        """Determines whether the image is a vector image.

        Returns:
            bool: True for vector images; False for scalar images.
        """
        return self.number_of_components_per_pixel > 1

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageProperties:\n' \
               ' size:                           {self.size}\n' \
               ' origin:                         {self.origin}\n' \
               ' spacing:                        {self.spacing}\n' \
               ' direction:                      {self.direction}\n' \
               ' dimensions:                     {self.dimensions}\n' \
               ' number_of_components_per_pixel: {self.number_of_components_per_pixel}\n' \
               ' pixel_id:                       {self.pixel_id}\n' \
            .format(self=self)

    def __eq__(self, other):
        """Determines the equality of two ImageProperties classes.

        Notes
            The equality does not include the number_of_components_per_pixel and pixel_id.

        Args:
            other (object): An ImageProperties instance or any other object.

        Returns:
            bool: True if the ImageProperties are equal; otherwise, False.
        """
        if isinstance(other, self.__class__):
            return self.size == other.size and \
                   self.origin == other.origin and \
                   self.spacing == other.spacing and \
                   self.direction == other.direction and \
                   self.dimensions == other.dimensions
        return NotImplemented

    def __ne__(self, other):
        """Determines the non-equality of two ImageProperties classes.

        Notes
            The non-equality does not include the number_of_components_per_pixel and pixel_id.

        Args:
            other (object): An ImageProperties instance or any other object.

        Returns:
            bool: True if the ImageProperties are non-equal; otherwise, False.
        """
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Gets the hash.

        Returns:
            int: The hash of the object.
        """
        return hash(tuple(sorted(self.__dict__.items())))


class NumpySimpleITKImageBridge:
    """A numpy to SimpleITK bridge, which provides static methods to convert between numpy array and SimpleITK image."""

    @staticmethod
    def convert(array: np.ndarray, properties: ImageProperties) -> sitk.Image:
        """Converts a numpy array to a SimpleITK image.

        Args:
            array (np.ndarray): The image as numpy array. The shape can be either:

                - shape=(n,), where n = total number of voxels
                - shape=(n,v), where n = total number of voxels and v = number of components per pixel (vector image)
                - shape=(<reversed image size>), what you get from sitk.GetArrayFromImage()
                - shape=(<reversed image size>,v), what you get from sitk.GetArrayFromImage()
                    and v = number of components per pixel (vector image)

            properties (ImageProperties): The image properties.

        Returns:
            sitk.Image: The SimpleITK image.
        """

        is_vector = False
        if not array.shape == properties.size[::-1]:
            # we need to reshape the array

            if array.ndim == 1:
                array = array.reshape(properties.size[::-1])
            elif array.ndim == 2:
                is_vector = True
                array = array.reshape((properties.size[::-1] + (array.shape[1],)))
            elif array.ndim == len(properties.size) + 1:
                is_vector = True
                # no need to reshape
            else:
                raise ValueError('array shape {} not supported'.format(array.shape))

        image = sitk.GetImageFromArray(array, is_vector)
        image.SetOrigin(properties.origin)
        image.SetSpacing(properties.spacing)
        image.SetDirection(properties.direction)

        return image


class SimpleITKNumpyImageBridge:
    """A SimpleITK to numpy bridge.

    Converts SimpleITK images to numpy arrays. Use the ``NumpySimpleITKImageBridge`` to convert back.
    """

    @staticmethod
    def convert(image: sitk.Image) -> typing.Tuple[np.ndarray, ImageProperties]:
        """Converts an image to a numpy array and an ImageProperties class.

        Args:
            image (SimpleITK.Image): The image.

        Returns:
            A Tuple[np.ndarray, ImageProperties]: The image as numpy array and the image properties.

        Raises:
            ValueError: If `image` is `None`.
        """

        if image is None:
            raise ValueError('Parameter image can not be None')

        return sitk.GetArrayFromImage(image), ImageProperties(image)
