import unittest
import numpy as np
import SimpleITK as sitk
import pymia.filtering.misc as m


class TestSizeCorrection(unittest.TestCase):

    def test_no_param(self):
        image = sitk.Image(3, 3, sitk.sitkUInt8)
        corrector = m.SizeCorrectionFilter()
        self.assertRaises(ValueError, corrector.execute, image)

    def test_2d(self):
        reference_shape = (4, 6)
        in_shape = (5, 5)
        values = np.random.randint(0, 10, in_shape[::-1])  # since dimesions are switched in itk
        image = sitk.GetImageFromArray(values)

        corrector = m.SizeCorrectionFilter()
        corr_image = corrector.execute(image, params=m.SizeCorrectionParams(reference_shape))

        self.assertEqual(corr_image.GetSize(), reference_shape)
        np_corr_image = sitk.GetArrayFromImage(corr_image)
        self.assertTrue((np_corr_image[:-1] == values[:, :-1]).all())  # dimensions are switched from itk to numpy

    def test_3d(self):
        reference_shape = (4, 4, 3)
        in_shape = (2, 4, 5)
        values = np.random.randint(0, 10, in_shape[::-1])  # since dimesions are switched in itk
        image = sitk.GetImageFromArray(values)

        corrector = m.SizeCorrectionFilter()
        corr_image = corrector.execute(image, params=m.SizeCorrectionParams(reference_shape))

        self.assertEqual(corr_image.GetSize(), reference_shape)
        np_corr_image = sitk.GetArrayFromImage(corr_image)
        self.assertTrue((np_corr_image[:, :, 1:-1] == values[1:-1, :, :]).all())

    def test_crop_and_pad(self):
        reference_shape = (4, 4, 3)
        in_shape = (2, 4, 5)
        values = np.random.randint(0, 10, in_shape[::-1])  # since dimesions are switched in itk
        image = sitk.GetImageFromArray(values)

        corrector = m.SizeCorrectionFilter()
        corr_image = corrector.execute(image, params=m.SizeCorrectionParams(reference_shape))

        self.assertEqual(corr_image.GetSize(), reference_shape)
        np_corr_image = sitk.GetArrayFromImage(corr_image)
        self.assertTrue((np_corr_image[:, :, 1:-1] == values[1:-1, :, :]).all())

    def test_one_sided(self):
        reference_shape = (4, 4, 3)
        in_shape = (2, 4, 5)
        values = np.random.randint(0, 10, in_shape[::-1])  # since dimesions are switched in itk
        image = sitk.GetImageFromArray(values)

        corrector = m.SizeCorrectionFilter(two_sided=False)
        corr_image = corrector.execute(image, params=m.SizeCorrectionParams(reference_shape))

        self.assertEqual(corr_image.GetSize(), reference_shape)
        np_corr_image = sitk.GetArrayFromImage(corr_image)
        self.assertTrue((np_corr_image[:, :, 2:] == values[2:, :, :]).all())

    def test_pad_constant(self):
        reference_shape = (4, 6)
        in_shape = (3, 5)
        values = np.ones(in_shape[::-1])   # since dimesions are switched in itk
        image = sitk.GetImageFromArray(values)

        corrector = m.SizeCorrectionFilter(pad_constant=1)
        corr_image = corrector.execute(image, params=m.SizeCorrectionParams(reference_shape))

        self.assertEqual(corr_image.GetSize(), reference_shape)
        np_corr_image = sitk.GetArrayFromImage(corr_image)
        self.assertTrue((np_corr_image == 1).all())  # dimensions are switched from itk to numpy


class TestRelabel(unittest.TestCase):

    def test_relabel_singles(self):
        image = sitk.Image(4, 4, sitk.sitkUInt8)
        image.SetPixel(1, 1, 5)
        image.SetPixel(1, 2, 5)
        image.SetPixel(3, 3, 1)

        relabel = m.Relabel({3: 5, 2: 1})  # 5 replaced by 3
        re_image = relabel.execute(image)
        np_re_image = sitk.GetArrayFromImage(re_image)
        self.assertFalse((np_re_image == 5).any())
        self.assertEqual(re_image.GetPixel(1, 1), 3)
        self.assertEqual(re_image.GetPixel(1, 2), 3)
        self.assertFalse((np_re_image == 1).any())
        self.assertEqual(re_image.GetPixel(3, 3), 2)

    def test_relabel_multiples(self):
        image = sitk.Image(4, 4, sitk.sitkUInt8)
        image.SetPixel(1, 1, 5)
        image.SetPixel(1, 2, 5)
        image.SetPixel(0, 3, 1)
        image.SetPixel(3, 1, 1)

        relabel = m.Relabel({3: (1, 5)})  # 5 and 1 replaced by 3
        re_image = relabel.execute(image)
        np_re_image = sitk.GetArrayFromImage(re_image)
        self.assertFalse((np_re_image == 5).any())
        self.assertFalse((np_re_image == 1).any())
        self.assertEqual(re_image.GetPixel(1, 1), 3)
        self.assertEqual(re_image.GetPixel(1, 2), 3)
        self.assertEqual(re_image.GetPixel(0, 3), 3)
        self.assertEqual(re_image.GetPixel(3, 1), 3)




