"""This module holds classes for data augmentation.

The data augmentation bases on the transformation concept (see :class:`pymia.data.transformation.Transform`)
and can easily be incorporated into the data loading process.

Warnings:
The augmentation relies on the random number generator of `numpy`. If you want to obtain reproducible result,
set numpy's seed prior to executing any augmentation:

>>> import numpy as np
>>> your_seed = 0
>>> np.random.seed(your_seed)

See Also:
    https://github.com/MIC-DKFZ/batchgenerators
"""
import typing

import numpy as np

import pymia.data.transformation as tfm
import SimpleITK as sitk


class RandomCrop(tfm.Transform):

    def __init__(self, shape: typing.Union[int, tuple], axis: typing.Union[int, tuple]=None,
                 p: float=1.0, entries=('images', 'labels')):
        """Randomly crops the sample to the specified shape.

        The sample shape must be bigger than the crop shape.

        Notes:
            A probability lower than 1.0 might make not much sense because it results in inconsistent output dimensions.

        Args:
            shape (int, tuple): The shape of the sample after the cropping.
                If axis is not defined, the cropping will be applied from the first dimension onwards of the sample.
                Use None to exclude an axis or define axis to specify the axis/axes to crop.
                E.g.:
                    shape=256 with the default axis parameter results in a shape of 256 x ...
                    shape=(256, 128) with the default axis parameter results in a shape of 256 x 128 x ...
                    shape=(None, 256) with the default axis parameter results in a shape of <as before> x 256 x ...
                    shape=(256, 128) with axis=(1, 0) results in a shape of 128 x 256 x ...
                    shape=(None, 128, 256) with axis=(1, 2, 0) results in a shape of 256 x <as before> x 256 x ...
            axis (int, tuple): Axis or axes to which the shape int or tuple correspond(s) to.
                If defined, must have the same length as shape.
            p (float): The probability of the cropping to be applied.
            entries (tuple): The sample's entries to apply the cropping to.
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape, )

        if axis is None:
            axis = tuple(range(len(shape)))
        if isinstance(axis, int):
            axis = (axis, )

        if len(axis) != len(shape):
            raise ValueError('If specified, the axis parameter must be of the same length as the shape')

        # filter out any axis where shape is None
        self.axis = tuple([a for a, s in zip(axis, shape) if s is not None])
        self.shape = tuple([s for s in shape if s is not None])

        self.p = p
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        if self.p < np.random.random():
            return sample

        for entry in self.entries:
            if entry not in sample:
                raise ValueError(tfm.ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

        anchors = [np.random.randint(0, sample[self.entries[0]].shape[a] - s) for a, s in zip(self.axis, self.shape)]

        for entry in self.entries:
            # todo(fabianbalsiger): replace by slicing (more elegant and faster?)
            for axis, new_axis_size, anchor in zip(self.axis, self.shape, anchors):
                sample[entry] = np.take(sample[entry], range(anchor, anchor + new_axis_size), axis)

        return sample


class RandomElasticDeformation(tfm.Transform):

    def __init__(self, num_control_points: int=4, deformation_sigma=15,
                 interpolators: tuple=(sitk.sitkBSpline, sitk.sitkNearestNeighbor),
                 spatial_rank: int=2, fill_value: float=0.0,
                 p: float=0.5, entries=('images', 'labels')):
        """Randomly transforms the sample elastically.

        Notes:
            The code bases on NiftyNet's RandomElasticDeformationLayer class (version 0.3.0).

        Warnings:
            Always inspect the results of this transform on some samples (especially for 3-D data).

        Args:
            num_control_points (int): The number of control points for the b-spline mesh.
            deformation_sigma: The maximum deformation along the deformation mesh.
            interpolators (tuple): The SimpleITK interpolators to use for each entry in entries.
            spatial_rank (int): The spatial rank (dimension) of the sample.
            fill_value (float): The fill value for the resampling.
            p (float): The probability of the elastic transformation to be applied.
            entries (tuple): The sample's entries to apply the elastic transformation to.
        """
        super().__init__()
        if len(interpolators) != len(entries):
            raise ValueError('interpolators must have the same length as entries')

        self.num_control_points = max(num_control_points, 2)  # need at minimum 2 control points
        self.deformation_sigma = max(deformation_sigma, 1)  # need at minimum 1
        self.spatial_rank = spatial_rank
        self.interpolators = interpolators
        self.fill_value = fill_value

        self.p = p
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        if self.p < np.random.random():
            return sample

        for entry in self.entries:
            if entry not in sample:
                raise ValueError(tfm.ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

        # initialize a SimpleITK image
        shape = sample[self.entries[0]].shape[:self.spatial_rank]
        img = sitk.GetImageFromArray(np.zeros(shape))  # todo(fabianbalsiger): set spacing etc with ImagePropertiesExtractor?

        # initialize B-spline transformation
        transform_mesh_size = [self.num_control_points] * img.GetDimension()  # todo(fabianbalsiger): allow control points to be defined per image dimension. Allow None to do no deformation in the direction (see comment below)
        bspline_transformation = sitk.BSplineTransformInitializer(img, transform_mesh_size)
        params = bspline_transformation.GetParameters()
        params = np.asarray(params, dtype=np.float)
        params += np.random.randn(params.shape[0]) * self.deformation_sigma

        # remove z deformations! The resolution in z is too bad
        # params[0:int(len(params) / 3)] = 0

        params = tuple(params)
        bspline_transformation.SetParameters(tuple(params))

        for interpolator_idx, entry in enumerate(self.entries):
            data = sample[entry]
            for channel in range(data.shape[-1]):
                img = sitk.GetImageFromArray(data[..., channel])
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(img)
                resampler.SetInterpolator(self.interpolators[interpolator_idx])
                resampler.SetDefaultPixelValue(self.fill_value)
                resampler.SetTransform(bspline_transformation)

                img_deformed = resampler.Execute(img)
                sample[entry][..., channel] = sitk.GetArrayFromImage(img_deformed)

        return sample


class RandomMirror(tfm.Transform):

    def __init__(self, axis: int=-2, p: float=1.0, entries=('images', 'labels')):
        """Randomly mirrors the sample along a given axis.

        Args:
            p (float): The probability of the mirroring to be applied.
            axis (int): The axis to apply the mirroring.
            entries (tuple): The sample's entries to apply the mirroring to.
        """
        super().__init__()
        self.axis = axis
        self.p = p
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        if self.p < np.random.random():
            return sample

        for entry in self.entries:
            if entry not in sample:
                raise ValueError(tfm.ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            sample[entry] = np.flip(sample[entry], self.axis).copy()

        return sample


class RandomRotation90(tfm.Transform):

    def __init__(self, axes: typing.Tuple[int]=(-3, -2), p: float=1.0, entries=('images', 'labels')):
        """Randomly rotates the sample 90, 180, or 270 degrees in the plane specified by axes.

        Raises:
            UserWarning: If the plane to rotate is not rectangular.

        Args:
            axes (tuple): The sample is rotated in the plane defined by the axes.
                Axes must be of length two and different.
            p (float): The probability of the rotation to be applied.
            entries (tuple): The sample's entries to apply the rotation to.
        """
        super().__init__()
        if len(axes) !=2:
            raise ValueError('axes must be of length two')

        self.axes = axes
        self.p = p
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        if self.p < np.random.random():
            return sample

        k = np.random.randint(1, 4)

        for entry in self.entries:
            if entry not in sample:
                raise ValueError(tfm.ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            if sample[entry].shape[self.axes[0]] != sample[entry].shape[self.axes[1]]:
                raise UserWarning('entry "{}" has unequal in-plane dimensions ({}, {}). \
                Random 90 degree rotations might produce undesired results.'.format(entry,
                                                                                    sample[entry].shape[self.axes[0]],
                                                                                    sample[entry].shape[self.axes[1]]))
            sample[entry] = np.rot90(sample[entry], k, axes=self.axes).copy()

        return sample


class RandomShift(tfm.Transform):

    def __init__(self, shift: typing.Union[int, tuple], axis: typing.Union[int, tuple]=None,
                 mode: str='mirror', fill: float=0.0,
                 p: float=1.0, entries=('images', 'labels')):
        """Randomly shifts the sample along axes by a value from the interval [-p * size(axis), +p * size(axis)],
        where p is the percentage of shifting and size(axis) is the size along an axis.

        Args:
            shift (int, tuple): The percentage of shifting of the axis' size.
                If axis is not defined, the shifting will be applied from the first dimension onwards of the sample.
                Use None to exclude an axis or define axis to specify the axis/axes to crop.
                E.g.:
                    shift=0.2 with the default axis parameter shifts the sample along the 1st axis.
                    shift=(0.2, 0.1) with the default axis parameter shifts the sample along the 1st and 2nd axes.
                    shift=(None, 0.2) with the default axis parameter shifts the sample along the 2st axis.
                    shift=(0.2, 0.1) with axis=(1, 0) shifts the sample along the 1st and 2nd axes.
                    shift=(None, 0.1, 0.2) with axis=(1, 2, 0) shifts the sample along the 1st and 3rd axes.
            axis (int, tuple): Axis or axes to which the shift int or tuple correspond(s) to.
                If defined, must have the same length as shape.
            p (float): The probability of the shift to be applied.
            entries (tuple): The sample's entries to apply the shifting to.
        """
        super().__init__()
        if isinstance(shift, int):
            shape = (shift, )

        if axis is None:
            axis = tuple(range(len(shift)))
        if isinstance(axis, int):
            axis = (axis, )

        if len(axis) != len(shift):
            raise ValueError('If specified, the axis parameter must be of the same length as the shift')

        # filter out any axis where shift is None
        self.axis = tuple([a for a, s in zip(axis, shift) if s is not None])
        self.shift = tuple([s for s in shift if s is not None])

        self.p = p
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        if self.p < np.random.random():
            return sample

        for entry in self.entries:
            if entry not in sample:
                raise ValueError(tfm.ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

        shifts_maximums = [int(s * sample[self.entries[0]].shape[a]) for a, s in zip(self.axis, self.shift)]
        shifts = [np.random.randint(-s_max, s_max) if s_max is not 0 else 0 for s_max in shifts_maximums]

        for entry in self.entries:
            for axis, shift in zip(self.axis, shifts):
                sample[entry] = np.roll(sample[entry], shift, axis)
                # todo(fabianbalsiger): implement zero filling (as optional "mode" parameter)?

        return sample
