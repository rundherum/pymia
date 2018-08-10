import abc
import typing as t

import numpy as np
import torch


ENTRY_NOT_EXTRACTED_ERR_MSG = 'Transform can not be applied because entry "{}" was not extracted'


# follows the principle of torchvision transform
class Transform(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, sample: dict) -> dict:
        pass


class ComposeTransform(Transform):

    def __init__(self, transforms: t.Iterable[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


class IntensityRescale(Transform):

    def __init__(self, lower, upper, loop_axis=None, entries=('images',)) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.loop_axis = loop_axis
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            np_entry = check_and_return(sample[entry], np.ndarray)
            if self.loop_axis is None:
                np_entry = self._normalize(np_entry, self.lower, self.upper)
            else:
                slicing = [slice(None) for _ in range(np_entry.ndim)]
                for i in range(np_entry.shape[self.loop_axis]):
                    slicing[self.loop_axis] = i
                    np_entry[slicing] = self._normalize(np_entry[slicing], self.lower, self.upper)

            sample[entry] = np_entry
        return sample

    @staticmethod
    def _normalize(arr: np.ndarray, lower, upper):
        dtype = arr.dtype
        min_, max_ = arr.min(), arr.max()
        if min_ == max_:
            raise ValueError('cannot normalize when min == max')
        arr = (arr - min_) / (max_ - min_) * (upper - lower) + lower
        return arr.astype(dtype)


class IntensityNormalization(Transform):

    def __init__(self, loop_axis=None, entries=('images',)) -> None:
        super().__init__()
        self.loop_axis = loop_axis
        self.entries = entries
        self.normalize_fn = self._normalize

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            np_entry = check_and_return(sample[entry], np.ndarray)
            if not np.issubdtype(np_entry.dtype, np.floating):
                raise ValueError('Array must be floating type')

            if self.loop_axis is None:
                np_entry = self.normalize_fn(np_entry)
            else:
                slicing = [slice(None) for _ in range(np_entry.ndim)]
                for i in range(np_entry.shape[self.loop_axis]):
                    slicing[self.loop_axis] = i
                    np_entry[slicing] = self.normalize_fn(np_entry[slicing])
            sample[entry] = np_entry
        return sample

    @staticmethod
    def _normalize(arr: np.ndarray):
        return (arr - arr.mean()) / arr.std()


class LambdaTransform(Transform):

    def __init__(self, lambda_fn, loop_axis=None, entries=('images',)) -> None:
        super().__init__()
        self.lambda_fn = lambda_fn
        self.loop_axis = loop_axis
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            if self.loop_axis is None:
                np_entry = self.lambda_fn(sample[entry])
            else:
                np_entry = check_and_return(sample[entry], np.ndarray)
                slicing = [slice(None) for _ in range(np_entry.ndim)]
                for i in range(np_entry.shape[self.loop_axis]):
                    slicing[self.loop_axis] = i
                    np_entry[slicing] = self.lambda_fn(np_entry[slicing])
            sample[entry] = np_entry
        return sample


class ClipPercentile(Transform):

    def __init__(self, upper_percentile: float, lower_percentile: float=None,
                 loop_axis=None, entries=('images',)) -> None:
        super().__init__()
        self.upper_percentile = upper_percentile
        if lower_percentile is None:
            lower_percentile = 100 - upper_percentile
        self.lower_percentile = lower_percentile
        self.loop_axis = loop_axis
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            np_entry = check_and_return(sample[entry], np.ndarray)

            if self.loop_axis is None:
                np_entry = self._clip(np_entry)
            else:
                slicing = [slice(None) for _ in range(np_entry.ndim)]
                for i in range(np_entry.shape[self.loop_axis]):
                    slicing[self.loop_axis] = i
                    np_entry[slicing] = self._clip(np_entry[slicing])

            sample[entry] = np_entry
        return sample

    def _clip(self, arr: np.ndarray):
        upper_max = np.percentile(arr, self.upper_percentile)
        arr[arr > upper_max] = upper_max
        lower_max = np.percentile(arr, self.lower_percentile)
        arr[arr < lower_max] = lower_max
        return arr


class Relabel(Transform):

    def __init__(self, label_changes: t.Dict[int, int], entries=('labels',)) -> None:
        super().__init__()
        self.label_changes = label_changes
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            np_entry = check_and_return(sample[entry], np.ndarray)
            for new_label, old_label in self.label_changes.items():
                np_entry[np_entry == old_label] = new_label
            sample[entry] = np_entry
        return sample


class Reshape(Transform):

    def __init__(self, shapes: dict) -> None:
        """Initializes a new instance of the Reshape class.

        Args:
            shapes (dict): A dict with keys being the entries and the values the new shapes of the entries.
                E.g. shapes = {'images': (-1, 4), 'labels' : (-1, 1)}
        """
        super().__init__()
        self.shapes = shapes

    def __call__(self, sample: dict) -> dict:
        for entry in self.shapes:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            np_entry = check_and_return(sample[entry], np.ndarray)
            sample[entry] = np.reshape(np_entry, self.shapes[entry])
        return sample


class ToTorchTensor(Transform):

    def __init__(self, entries=('images', 'labels')) -> None:
        super().__init__()
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            np_entry = check_and_return(sample[entry], np.ndarray)
            sample[entry] = torch.from_numpy(np_entry)
        return sample


class Permute(Transform):

    def __init__(self, permutation: tuple, entries=('images', 'labels')) -> None:
        super().__init__()
        self.permutation = permutation
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            np_entry = check_and_return(sample[entry], np.ndarray)
            sample[entry] = np.transpose(np_entry, self.permutation)
        return sample


class Squeeze(Transform):

    def __init__(self, entries=('images', 'labels')) -> None:
        super().__init__()
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            np_entry = check_and_return(sample[entry], np.ndarray)
            sample[entry] = np_entry.squeeze()
        return sample


class UnSqueeze(Transform):

    def __init__(self, entries=('images', 'labels')) -> None:
        super().__init__()
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            np_entry = check_and_return(sample[entry], np.ndarray)
            sample[entry] = np.expand_dims(np_entry, -1)
        return sample


class SizeCorrection(Transform):
    """Size correction transformation.

    Corrects the size, i.e. shape, of an array to a given reference shape.
    """

    def __init__(self, shape: t.Tuple[t.Union[None, int], ...], pad_value: int=0, entries=('images', 'labels')) -> None:
        """Initializes a new instance of the SizeCorrection class.

        Args:
            shape (tuple of ints): The reference shape in NumPy format, i.e. z-, y-, x-order. To not correct an axis
                dimension, set the axis value to None.
            pad_value (int): The value to set the padded values of the array.
            entries ():
        """
        super().__init__()
        self.entries = entries
        self.shape = shape
        self.pad_value = pad_value

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            np_entry = check_and_return(sample[entry], np.ndarray)
            if not len(self.shape) <= np_entry.ndim:
                raise ValueError('Shape dimension needs to be less or equal to {}'.format(np_entry.ndim))

            for idx, size in enumerate(self.shape):
                if size is not None and size < np_entry.shape[idx]:
                    # crop current dimension
                    before = (np_entry.shape[idx] - size) // 2
                    after = np_entry.shape[idx] - (np_entry.shape[idx] - size) // 2 - ((np_entry.shape[idx] - size) % 2)
                    slicing = [slice(None)] * np_entry.ndim
                    slicing[idx] = slice(before, after)
                    np_entry = np_entry[slicing]
                elif size is not None and size > np_entry.shape[idx]:
                    # pad current dimension
                    before = (size - np_entry.shape[idx]) // 2
                    after = (size - np_entry.shape[idx]) // 2 + ((size - np_entry.shape[idx]) % 2)
                    pad_width = [(0, 0)] * np_entry.ndim
                    pad_width[idx] = (before, after)
                    np_entry = np.pad(np_entry, pad_width, mode='constant', constant_values=self.pad_value)

            sample[entry] = np_entry

        return sample


class Mask(Transform):

    def __init__(self, mask_key: str, mask_value: int=0, masking_value: float=0,
                 loop_axis=None, entries=('images', 'labels')) -> None:
        super().__init__()
        self.mask_key = mask_key
        self.mask_value = mask_value
        self.masking_value = masking_value
        self.loop_axis = loop_axis
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        np_mask = check_and_return(sample[self.mask_key], np.ndarray)

        for entry in self.entries:
            if entry not in sample:
                raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            np_entry = check_and_return(sample[entry], np.ndarray)

            if np_mask.shape == np_entry.shape:
                np_entry[np_mask == self.mask_value] = self.masking_value
            else:
                mask_for_np_entry = np.repeat(np.expand_dims(np_mask, self.loop_axis),
                                          np_entry.shape[self.loop_axis], axis=self.loop_axis)
                np_entry[mask_for_np_entry == self.mask_value] = self.masking_value

            sample[entry] = np_entry
        return sample


class RandomCrop(Transform):

    def __init__(self, shape: t.Union[int, tuple], axis: t.Union[int, tuple] = None, channels_last=True, p=1.0,
                 entries=('images', 'labels')):
        """
        Randomly crop the sample to the specified shape. Sample shape must be bigger than crop shape.

        :param shape: shape of the crop as int or tuple of ints.
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
        :param axis: axis to which the shape sizes correspond to. Must have the same number of dimensions as shape when defined.
        :param channels_last: data is expected to be channels-last by default. Otherwise channels-second is expected.
        :param p: probability of this transform to be applied. Default is 1.0.
        :param entries: entries on which this transform is applied to.
        """

        super().__init__()

        if isinstance(shape, int):
            shape = (shape,)
        self.crop_shape = shape

        self.crop_axis = tuple(i for i in range(-len(shape), 0))
        if axis is None:
            rng = range(-len(shape) - 1, -1) if channels_last else range(-len(shape), 0)
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


def check_and_return(obj, type_):
    if not isinstance(obj, type_):
        raise ValueError("entry must be '{}'".format(type_.__name__))
    return obj
