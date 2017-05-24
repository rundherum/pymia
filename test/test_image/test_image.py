import SimpleITK as sitk
from unittest import TestCase
from image.image import ImageProperties


class TestImageProperties(TestCase):
    def test_is_two_dimensional(self):
        x = 10
        y = 10
        image = sitk.Image([x, y], sitk.sitkUInt8)
        dut = ImageProperties(image)

        self.assertEqual(dut.is_two_dimensional(), True)
        self.assertEqual(dut.is_three_dimensional(), False)
        self.assertEqual(dut.is_vector_image(), False)

    def test_is_three_dimensional(self):
        x = 10
        y = 10
        z = 3
        image = sitk.Image([x, y, z], sitk.sitkUInt8)
        dut = ImageProperties(image)

        self.assertEqual(dut.is_two_dimensional(), False)
        self.assertEqual(dut.is_three_dimensional(), True)
        self.assertEqual(dut.is_vector_image(), False)

    def test_is_vector_image(self):
        x = 10
        y = 10
        number_of_components_per_pixel = 3
        image = sitk.Image([x, y], sitk.sitkVectorUInt8, number_of_components_per_pixel)
        dut = ImageProperties(image)

        self.assertEqual(dut.is_two_dimensional(), True)
        self.assertEqual(dut.is_three_dimensional(), False)
        self.assertEqual(dut.is_vector_image(), True)

    def test_properties(self):
        x = 10
        y = 10
        z = 3
        size = (x, y, z)
        direction = (0, 1, 0, 1, 0, 0, 0, 0, 1)
        image = sitk.Image([x, y, z], sitk.sitkUInt8)
        image.SetOrigin(size)
        image.SetSpacing(size)
        image.SetDirection(direction)
        dut = ImageProperties(image)

        self.assertEqual(dut.size, size)
        self.assertEqual(dut.origin, size)
        self.assertEqual(dut.spacing, size)
        self.assertEqual(dut.direction, direction)
        self.assertEqual(dut.dimensions, z)
        self.assertEqual(dut.number_of_components_per_pixel, 1)
