import unittest

import numpy as np
import SimpleITK as sitk

import pymia.filtering.filter as pymia_fltr
import pymia.filtering.preprocessing as pymia_fltr_prep


class TestNormalizeZScore(unittest.TestCase):

    def setUp(self):
        # set up image
        image = sitk.Image((4, 1), sitk.sitkUInt8)
        image.SetPixel((0, 0), 1)
        image.SetPixel((1, 0), 2)
        image.SetPixel((2, 0), 3)
        image.SetPixel((3, 0), 4)

        self.image = image

        # test_case = [1, 2, 3, 4]
        # not in R, so tested by using:
        #    (test_case[i] - mean(test_case, axis=0)) / sqrt(var(test_case) * 3/4)
        self.desired = np.array([[-1.3416407864999, -0.44721359549996, 0.44721359549996, 1.3416407864999]], np.float64)

    def test_normalization(self):
        dut = pymia_fltr_prep.NormalizeZScore()
        out = dut.execute(self.image)
        out_arr = sitk.GetArrayFromImage(out)

        np.testing.assert_array_almost_equal(self.desired, out_arr, decimal=12)

    def test_normalization_with_param(self):
        dut = pymia_fltr_prep.NormalizeZScore()
        out = dut.execute(self.image, pymia_fltr.IFilterParams())
        out_arr = sitk.GetArrayFromImage(out)

        np.testing.assert_array_almost_equal(self.desired, out_arr, decimal=12)

    def test_image_properties(self):
        dut = pymia_fltr_prep.NormalizeZScore()
        out = dut.execute(self.image)
        self.assertEqual(self.image.GetSize(), out.GetSize())
        self.assertEqual(self.image.GetOrigin(), out.GetOrigin())
        self.assertEqual(self.image.GetSpacing(), out.GetSpacing())
        self.assertEqual(self.image.GetDirection(), out.GetDirection())
        self.assertEqual(self.image.GetDimension(), out.GetDimension())
        self.assertEqual(self.image.GetNumberOfComponentsPerPixel(), out.GetNumberOfComponentsPerPixel())

        self.assertEqual(sitk.sitkFloat64, out.GetPixelID())
