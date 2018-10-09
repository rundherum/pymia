import typing as t

import numpy as np

import pymia.data.transformation as tfm


class RandomCrop(tfm.Transform):

    def __init__(self, shape: t.Union[int, tuple], axis: t.Union[int, tuple] = None, p: float=1.0,
                 entries=('images', 'labels')):
        """Randomly crop the sample to the specified shape. Sample shape must be bigger than crop shape.

        Args:
            shape: Shape of the crop as int or tuple of ints.
                E.g. for image data HxW:
                    shape=(256, 128) with the default axis parameter results in an image of 256x128.
                    shape=256 with the default axis parameter results in an image of Hx256.
                E.g. for volumetric data DxHxW:
                    shape=(16, 32, 64) with the default axis parameter results in a volume of 16x32x64.
                    shape=(1, 128, 128) with the default axis parameter results in a volume/image of 1x128x128.
                    shape=(50, 5) with axis=(-2, -4) in a volume of 5xHx50 for channels-last data.
                    shape=(50, 5) with axis=(-1, -3) in a volume of 5xHx50 for channels-second data.
                    shape=64 with the default axis parameter results in a volume of DxHx64.
                    shape=32 with axis=-4 (axis=-3 for channels-second data) results in a volume of 32xHxW.
            axis: Axis to which the shape sizes correspond to.
                Must have the same number of dimensions as shape when defined.
            p (float): probability of this transform to be applied. Default is 1.0.
            entries: entries on which this transform is applied to.
        """

        super().__init__()

        if isinstance(shape, int):
            shape = (shape,)
        self.crop_shape = shape

        self.crop_axis = tuple(i for i in range(-len(shape), 0))
        if axis is None:
            rng = range(-len(shape) - 1, -1)
            axis = tuple(i for i in rng)
        if isinstance(axis, int):
            axis = (axis,)

        if len(axis) != len(shape):
            raise ValueError('If specified, the axis parameter must have the same number of dimensions as the shape!')

        self.shape_axis = axis

        self.p = p

        self.entries = entries

    def __call__(self, sample: dict) -> dict:

        if not any(self.entries & sample.keys()):
            return sample

        if self.p < np.random.random():
            return sample

        if any([sample[self.entries[0]].shape[s_a] < self.crop_shape[c_a] for s_a, c_a in
                zip(self.shape_axis, self.crop_axis)]):
            raise ValueError('Crop shape must not be bigger than sample shape!')

        a_maxs = [sample[self.entries[0]].shape[s_a] - self.crop_shape[c_a] for s_a, c_a in
                  zip(self.shape_axis, self.crop_axis)]
        anchors = [np.random.randint(0, a_max) for a_max in a_maxs]

        for entry in self.entries:
            if entry not in sample:
                continue
            for s_a, c_a, anc in zip(self.shape_axis, self.crop_axis, anchors):
                if sample[entry].shape[s_a] == 1:
                    continue
                sample[entry] = np.take(sample[entry], range(anc, anc + self.crop_shape[c_a]), s_a)
        return sample
