import unittest

import numpy as np
import SimpleITK as sitk

import pymia.data.conversion as img


class TestImageProperties(unittest.TestCase):
    def test_is_two_dimensional(self):
        x = 10
        y = 10
        image = sitk.Image([x, y], sitk.sitkUInt8)
        dut = img.ImageProperties(image)

        self.assertEqual(dut.is_two_dimensional(), True)
        self.assertEqual(dut.is_three_dimensional(), False)
        self.assertEqual(dut.is_vector_image(), False)

    def test_is_three_dimensional(self):
        x = 10
        y = 10
        z = 3
        image = sitk.Image([x, y, z], sitk.sitkUInt8)
        dut = img.ImageProperties(image)

        self.assertEqual(dut.is_two_dimensional(), False)
        self.assertEqual(dut.is_three_dimensional(), True)
        self.assertEqual(dut.is_vector_image(), False)

    def test_is_vector_image(self):
        x = 10
        y = 10
        number_of_components_per_pixel = 3
        image = sitk.Image([x, y], sitk.sitkVectorUInt8, number_of_components_per_pixel)
        dut = img.ImageProperties(image)

        self.assertEqual(dut.is_two_dimensional(), True)
        self.assertEqual(dut.is_three_dimensional(), False)
        self.assertEqual(dut.is_vector_image(), True)

    def test_properties(self):
        x = 10
        y = 10
        z = 3
        pixel_id = sitk.sitkUInt8
        size = (x, y, z)
        direction = (0, 1, 0, 1, 0, 0, 0, 0, 1)
        image = sitk.Image([x, y, z], pixel_id)
        image.SetOrigin(size)
        image.SetSpacing(size)
        image.SetDirection(direction)
        dut = img.ImageProperties(image)

        self.assertEqual(dut.size, size)
        self.assertEqual(dut.origin, size)
        self.assertEqual(dut.spacing, size)
        self.assertEqual(dut.direction, direction)
        self.assertEqual(dut.dimensions, z)
        self.assertEqual(dut.number_of_components_per_pixel, 1)
        self.assertEqual(dut.pixel_id, pixel_id)

    def test_equality(self):
        x = 10
        y = 10
        z = 3
        pixel_id = sitk.sitkUInt8
        size = (x, y, z)
        direction = (0, 1, 0, 1, 0, 0, 0, 0, 1)
        image = sitk.Image([x, y, z], pixel_id)
        image.SetOrigin(size)
        image.SetSpacing(size)
        image.SetDirection(direction)
        dut1 = img.ImageProperties(image)
        dut2 = img.ImageProperties(image)

        self.assertTrue(dut1 == dut2)
        self.assertFalse(dut1 != dut2)

        image = sitk.Image([x, y, z], sitk.sitkInt8)
        image.SetOrigin(size)
        image.SetSpacing(size)
        image.SetDirection(direction)
        dut1 = img.ImageProperties(image)

        self.assertTrue(dut1 == dut2)
        self.assertFalse(dut1 != dut2)

        image = sitk.Image([x, y, z], sitk.sitkVectorUInt8, 2)
        image.SetOrigin(size)
        image.SetSpacing(size)
        image.SetDirection(direction)
        dut1 = img.ImageProperties(image)

        self.assertTrue(dut1 == dut2)
        self.assertFalse(dut1 != dut2)

    def test_non_equality(self):
        x = 10
        y = 10
        z = 3
        pixel_id = sitk.sitkUInt8
        size = (x, y, z)
        direction = (0, 1, 0, 1, 0, 0, 0, 0, 1)
        image = sitk.Image([x, y, z], pixel_id)
        image.SetOrigin(size)
        image.SetSpacing(size)
        image.SetDirection(direction)
        dut1 = img.ImageProperties(image)

        different_size = (x, y, 2)

        # non-equal size
        image = sitk.Image(different_size, pixel_id)
        image.SetOrigin(size)
        image.SetSpacing(size)
        image.SetDirection(direction)
        dut2 = img.ImageProperties(image)

        self.assertFalse(dut1 == dut2)
        self.assertTrue(dut1 != dut2)

        # non-equal origin
        image = sitk.Image(size, pixel_id)
        image.SetOrigin(different_size)
        image.SetSpacing(size)
        image.SetDirection(direction)
        dut2 = img.ImageProperties(image)

        self.assertFalse(dut1 == dut2)
        self.assertTrue(dut1 != dut2)

        # non-equal spacing
        different_size = (x, y, 2)
        image = sitk.Image(size, pixel_id)
        image.SetOrigin(size)
        image.SetSpacing(different_size)
        image.SetDirection(direction)
        dut2 = img.ImageProperties(image)

        self.assertFalse(dut1 == dut2)
        self.assertTrue(dut1 != dut2)

        # non-equal direction
        different_size = (x, y, 2)
        image = sitk.Image(size, pixel_id)
        image.SetOrigin(size)
        image.SetSpacing(size)
        image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        dut2 = img.ImageProperties(image)

        self.assertFalse(dut1 == dut2)
        self.assertTrue(dut1 != dut2)


class TestNumpySimpleITKImageBridge(unittest.TestCase):

    def setUp(self):
        dim_x = 5
        dim_y = 10
        dim_z = 3
        self.no_vector_components = 4

        # some unordinary origins, spacings and directions
        self.origin_spacing_2d = (dim_x, dim_y)
        self.direction_2d = (0, 1, 1, 0)
        self.origin_spacing_3d = (dim_x, dim_y, dim_z)
        self.direction_3d = (1, 0, 0, 0, 1, 0, 0, 0, 1)

        # set up images
        image = sitk.Image((dim_x, dim_y, dim_z), sitk.sitkUInt8)
        image.SetOrigin(self.origin_spacing_3d)
        image.SetSpacing(self.origin_spacing_3d)
        image.SetDirection(self.direction_3d)
        self.properties_3d = img.ImageProperties(image)

        image = sitk.Image((dim_x, dim_y), sitk.sitkUInt8)
        image.SetOrigin(self.origin_spacing_2d)
        image.SetSpacing(self.origin_spacing_2d)
        image.SetDirection(self.direction_2d)
        self.properties_2d = img.ImageProperties(image)

        # set up numpy arrays (note the inverted order of the dimension)
        self.array_image_shape_2d = np.zeros((dim_y, dim_x), np.uint8)
        self.array_2d_vector = np.zeros((dim_y * dim_x, self.no_vector_components), np.uint8)
        self.array_image_shape_2d_vector = np.zeros((dim_y, dim_x, self.no_vector_components), np.uint8)

        self.array_image_shape_3d = np.zeros((dim_z, dim_y, dim_x), np.uint8)
        self.array_3d_vector = np.zeros((dim_z * dim_y * dim_x, self.no_vector_components), np.uint8)
        self.array_image_shape_3d_vector = np.zeros((dim_z, dim_y, dim_x, self.no_vector_components), np.uint8)

    def test_vector_to_image(self):
        # test array shape (n,) i.e., (x * y) and (x * y * z)
        image = img.NumpySimpleITKImageBridge.convert(self.array_image_shape_3d.flatten(), self.properties_3d)
        self._assert_3d(image)

        image = img.NumpySimpleITKImageBridge.convert(self.array_image_shape_2d.flatten(), self.properties_2d)
        self._assert_2d(image)

    def test_array_to_image(self):
        # test array shape (y, x) and (z, y, x)
        image = img.NumpySimpleITKImageBridge.convert(self.array_image_shape_3d, self.properties_3d)
        self._assert_3d(image)

        image = img.NumpySimpleITKImageBridge.convert(self.array_image_shape_2d, self.properties_2d)
        self._assert_2d(image)

    def test_array_to_vector_image(self):
        # test array shape (y, x, v) and (z, y, x, v), where v = number of vector component
        image = img.NumpySimpleITKImageBridge.convert(self.array_image_shape_3d_vector, self.properties_3d)
        self._assert_3d(image, True)

        image = img.NumpySimpleITKImageBridge.convert(self.array_image_shape_2d_vector, self.properties_2d)
        self._assert_2d(image, True)

    def test_vector_to_vector_image(self):
        # test array shape (y * x, v) and (z * y * x, v), where v = number of vector components
        image = img.NumpySimpleITKImageBridge.convert(self.array_3d_vector, self.properties_3d)
        self._assert_3d(image, True)

        image = img.NumpySimpleITKImageBridge.convert(self.array_2d_vector, self.properties_2d)
        self._assert_2d(image, True)

    def test_convert_unknown_shape(self):
        with self.assertRaises(ValueError):
            img.NumpySimpleITKImageBridge.convert(self.array_image_shape_3d.flatten(), self.properties_2d)

    def _assert_2d(self, image: sitk.Image, is_vector=False):
        self.assertEqual(self.properties_2d.size, image.GetSize())

        if is_vector:
            self.assertEqual(self.no_vector_components, image.GetNumberOfComponentsPerPixel())
        else:
            self.assertEqual(1, image.GetNumberOfComponentsPerPixel())

        self.assertEqual(self.origin_spacing_2d, image.GetOrigin())
        self.assertEqual(self.origin_spacing_2d, image.GetSpacing())
        self.assertEqual(self.direction_2d, image.GetDirection())

    def _assert_3d(self, image: sitk.Image, is_vector=False):
        self.assertEqual(self.properties_3d.size, image.GetSize())

        if is_vector:
            self.assertEqual(self.no_vector_components, image.GetNumberOfComponentsPerPixel())
        else:
            self.assertEqual(1, image.GetNumberOfComponentsPerPixel())

        self.assertEqual(self.origin_spacing_3d, image.GetOrigin())
        self.assertEqual(self.origin_spacing_3d, image.GetSpacing())
        self.assertEqual(self.direction_3d, image.GetDirection())


class TestSimpleITKNumpyImageBridge(unittest.TestCase):
    def test_convert(self):
        x = 10
        y = 10
        z = 3
        size = (x, y, z)
        image = sitk.Image(size, sitk.sitkUInt8)

        array, properties = img.SimpleITKNumpyImageBridge.convert(image)

        self.assertEqual(isinstance(array, np.ndarray), True)
        self.assertEqual(array.shape, size[::-1])
        self.assertEqual(array.dtype, np.uint8)
        self.assertEqual(isinstance(properties, img.ImageProperties), True)
        self.assertEqual(properties.size, size)

    def test_convert_None(self):
        with self.assertRaises(ValueError):
            img.SimpleITKNumpyImageBridge.convert(None)
