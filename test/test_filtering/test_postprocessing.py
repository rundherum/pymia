import unittest

import numpy as np
import SimpleITK as sitk

import pymia.filtering.postprocessing as fltr


class TestLargestNConnectedComponents(unittest.TestCase):

    def setUp(self):
        # set up image with 3 different connected components
        image = sitk.Image((5, 5), sitk.sitkUInt8)
        image.SetPixel((0, 0), 1)

        image.SetPixel((2, 0), 1)
        image.SetPixel((2, 1), 1)

        image.SetPixel((4, 0), 1)
        image.SetPixel((4, 1), 1)
        image.SetPixel((4, 2), 1)

        self.image = image

    def test_zero_components(self):
        with self.assertRaises(ValueError):
            fltr.LargestNConnectedComponents(0, False)

    def test_one_components(self):
        dut = fltr.LargestNConnectedComponents(1, False)
        result = dut.execute(self.image)

        self.assertEqual(result.GetPixel((4, 0)), 1)
        self.assertEqual(result.GetPixel((4, 1)), 1)
        self.assertEqual(result.GetPixel((4, 2)), 1)

        result_array = sitk.GetArrayFromImage(result)
        self.assertEqual(result_array.sum(), 3)

    def test_two_components(self):
        dut = fltr.LargestNConnectedComponents(2, False)
        result = dut.execute(self.image)

        self.assertEqual(result.GetPixel((2, 0)), 1)
        self.assertEqual(result.GetPixel((2, 1)), 1)
        self.assertEqual(result.GetPixel((4, 0)), 1)
        self.assertEqual(result.GetPixel((4, 1)), 1)
        self.assertEqual(result.GetPixel((4, 2)), 1)

        result_array = sitk.GetArrayFromImage(result)
        self.assertEqual(result_array.sum(), 5)

    def test_three_components(self):
        dut = fltr.LargestNConnectedComponents(3, False)
        result = dut.execute(self.image)

        self.assertEqual(result.GetPixel((0, 0)), 1)
        self.assertEqual(result.GetPixel((2, 0)), 1)
        self.assertEqual(result.GetPixel((2, 1)), 1)
        self.assertEqual(result.GetPixel((4, 0)), 1)
        self.assertEqual(result.GetPixel((4, 1)), 1)
        self.assertEqual(result.GetPixel((4, 2)), 1)

        result_array = sitk.GetArrayFromImage(result)
        self.assertEqual(result_array.sum(), 6)

    def test_four_components(self):
        dut = fltr.LargestNConnectedComponents(3, False)
        result = dut.execute(self.image)

        self.assertEqual(result.GetPixel((0, 0)), 1)
        self.assertEqual(result.GetPixel((2, 0)), 1)
        self.assertEqual(result.GetPixel((2, 1)), 1)
        self.assertEqual(result.GetPixel((4, 0)), 1)
        self.assertEqual(result.GetPixel((4, 1)), 1)
        self.assertEqual(result.GetPixel((4, 2)), 1)

        result_array = sitk.GetArrayFromImage(result)
        self.assertEqual(result_array.sum(), 6)

    def test_consecutive_labels(self):
        dut = fltr.LargestNConnectedComponents(3, True)
        result = dut.execute(self.image)

        self.assertEqual(result.GetPixel((0, 0)), 3)
        self.assertEqual(result.GetPixel((2, 0)), 2)
        self.assertEqual(result.GetPixel((2, 1)), 2)
        self.assertEqual(result.GetPixel((4, 0)), 1)
        self.assertEqual(result.GetPixel((4, 1)), 1)
        self.assertEqual(result.GetPixel((4, 2)), 1)

        result_array = sitk.GetArrayFromImage(result)
        self.assertEqual(result_array.sum(), 10)
