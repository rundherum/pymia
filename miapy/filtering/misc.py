"""The misc (miscellaneous) module contains filters, which don't have a classical purpose."""

import subprocess
import os
import tempfile
from typing import Dict, Union

import numpy as np
import SimpleITK as sitk

import miapy.filtering.filter as miapy_fltr


class Relabel(miapy_fltr.IFilter):
    """Relabels the labels in the file by the provided rule"""

    def __init__(self, label_changes: Dict[int, Union[int, tuple]]) -> None:
        """Initializes a new instance of the LargestNComponents class.

        Args:
            label_changes(Dict[int, Union[int, tuple]]): Label change rule where the key is the new label and
                the value the existing (can be multiple)
        """
        super().__init__()
        self.label_changes = label_changes

    def execute(self, image: sitk.Image, params: miapy_fltr.IFilterParams = None) -> sitk.Image:
        """Executes the relabeling of the label image.

        Args:
            image (sitk.Image): The image.
            params (miapy_fltr.IFilterParams): The parameters (unused).

        Returns:
            sitk.Image: The filtered image.
        """
        np_img = sitk.GetArrayFromImage(image)
        new_np_img = np_img.copy()
        for new_label, old_labels in self.label_changes.items():
            mask = np.in1d(np_img.ravel(), old_labels).reshape(np_img.shape)
            new_np_img[mask] = new_label
        new_img = sitk.GetImageFromArray(new_np_img)
        new_img.CopyInformation(image)
        return new_img

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        str_list = []
        for k, v in self.label_changes.items():
            str_list.append('{}->{}'.format(k, v))
        return 'Relabel:\n' \
               ' label_changes: {label_changes}\n' \
            .format(self=self, label_changes='; '.join(str_list))


class SizeCorrectionParams(miapy_fltr.IFilterParams):
    """Size (shape) correction filter parameters."""

    def __init__(self, reference_shape: tuple) -> None:
        """Initializes a new instance of the SizeCorrectionParams class.

        Args:
            reference_shape (tuple): The reference or target shape.
        """
        self.dims = len(reference_shape)
        self.reference_shape = reference_shape


class SizeCorrectionFilter(miapy_fltr.IFilter):
    """A method to correct shape/size difference."""

    def __init__(self, two_sided=True, pad_constant=0.0) -> None:
        """ Initializes a new instance of the SizeCorrectionFilter class.

        Args:
            two_sided (bool): whether the cropping and padding should be applied on one or both side of the dimension.
            pad_constant (int): constant used for padding
        """
        super().__init__()
        self.two_sided = two_sided
        self.pad_constant = pad_constant

    def execute(self, image: sitk.Image, params: SizeCorrectionParams = None) -> sitk.Image:
        """Executes the shape/size correction by help of padding and cropping.

        Args:
            image (sitk.Image): The image.
            params (SizeCorrectionParams): The parameters containing the reference (target) shape.

        Returns:
            sitk.Image: The filtered image.
        """
        if params is None:
            raise ValueError('ShapeParams argument is missing')
        if image.GetDimension() != params.dims:
            raise ValueError('image dimension {} is not compatible with reference shape dimension {}'
                             .format(image.GetDimension(), params.dims))

        image_shape = image.GetSize()
        crop = [params.dims*[0], params.dims*[0]]
        pad = [params.dims*[0], params.dims*[0]]
        for dim in range(params.dims):
            ref_size = params.reference_shape[dim]
            dim_size = image_shape[dim]
            if dim_size > ref_size:
                if self.two_sided:
                    crop[0][dim] = (dim_size - ref_size) // 2
                    crop[1][dim] = (dim_size - ref_size) // 2 + ((dim_size - ref_size) % 2)
                else:
                    crop[0][dim] = (dim_size - ref_size)
            elif dim_size < ref_size:
                if self.two_sided:
                    pad[0][dim] = (ref_size - dim_size) // 2
                    pad[1][dim] = (ref_size - dim_size) // 2 + ((ref_size - dim_size) % 2)
                else:
                    pad[0][dim] = (ref_size - dim_size)

        crop_needed = any(any(c) for c in crop)
        if crop_needed:
            image = sitk.Crop(image, crop[0], crop[1])

        pad_needed = any(any(p) for p in pad)
        if pad_needed:
            image = sitk.ConstantPad(image, pad[0], pad[1], self.pad_constant)

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SizeCorrectionFilter:\n' \
               ' two_sided: {self.two_sided}\n' \
            .format(self=self)


class CmdlineExecutor(miapy_fltr.IFilter):
    """Represents a command line executable."""

    def __init__(self, executable_path: str):
        """Initializes a new instance of the CmdlineExecutor class.

        Args:
            executable_path (str): path to the executable to run.
        """
        super().__init__()
        self.executable_path = executable_path

    def execute(self, image: sitk.Image, params: miapy_fltr.IFilterParams=None) -> sitk.Image:
        """Executes a command line program.

        Args:
            image (sitk.Image): The image.
            params (miapy_fltr.IFilterParams): The parameters (unused).

        Returns:
            sitk.Image: The filtered image.
        """
        temp_dir = tempfile.gettempdir()
        temp_in = os.path.join(temp_dir, 'in.nii')
        sitk.WriteImage(image, temp_in)
        temp_out = os.path.join(temp_dir, 'out.nii')
        subprocess.run([self.executable_path, temp_in, temp_out], check=True)
        out_image = sitk.ReadImage(temp_out, image.GetPixelID())
        # clean up
        os.remove(temp_in)
        os.remove(temp_out)
        return out_image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'CmdlineExecutor:\n' \
               ' executable_path: {self.executable_path}\n' \
            .format(self=self)
