import abc
import typing

import numpy as np
import torch


ENTRY_NOT_EXTRACTED_ERR_MSG = 'Transform can not be applied because entry "{}" was not extracted'

raise_error_if_entry_not_extracted = True


# follows the principle of torchvision transform
class Transform(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, sample: dict) -> dict:
        pass


class ComposeTransform(Transform):

    def __init__(self, transforms: typing.Iterable[Transform]) -> None:
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
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

            np_entry = check_and_return(sample[entry], np.ndarray)
            if self.loop_axis is None:
                np_entry = self._normalize(np_entry, self.lower, self.upper)
            else:
                slicing = [slice(None) for _ in range(np_entry.ndim)]
                for i in range(np_entry.shape[self.loop_axis]):
                    slicing[self.loop_axis] = i
                    np_entry[tuple(slicing)] = self._normalize(np_entry[tuple(slicing)], self.lower, self.upper)

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
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

            np_entry = check_and_return(sample[entry], np.ndarray)
            if not np.issubdtype(np_entry.dtype, np.floating):
                raise ValueError('Array must be floating type')

            if self.loop_axis is None:
                np_entry = self.normalize_fn(np_entry)
            else:
                slicing = [slice(None) for _ in range(np_entry.ndim)]
                for i in range(np_entry.shape[self.loop_axis]):
                    slicing[self.loop_axis] = i
                    np_entry[tuple(slicing)] = self.normalize_fn(np_entry[tuple(slicing)])
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
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

            if self.loop_axis is None:
                np_entry = self.lambda_fn(sample[entry])
            else:
                np_entry = check_and_return(sample[entry], np.ndarray)
                slicing = [slice(None) for _ in range(np_entry.ndim)]
                for i in range(np_entry.shape[self.loop_axis]):
                    slicing[self.loop_axis] = i
                    np_entry[tuple(slicing)] = self.lambda_fn(np_entry[tuple(slicing)])
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
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

            np_entry = check_and_return(sample[entry], np.ndarray)

            if self.loop_axis is None:
                np_entry = self._clip(np_entry)
            else:
                slicing = [slice(None) for _ in range(np_entry.ndim)]
                for i in range(np_entry.shape[self.loop_axis]):
                    slicing[self.loop_axis] = i
                    np_entry[tuple(slicing)] = self._clip(np_entry[tuple(slicing)])

            sample[entry] = np_entry
        return sample

    def _clip(self, arr: np.ndarray):
        upper_max = np.percentile(arr, self.upper_percentile)
        arr[arr > upper_max] = upper_max
        lower_max = np.percentile(arr, self.lower_percentile)
        arr[arr < lower_max] = lower_max
        return arr


class Relabel(Transform):

    def __init__(self, label_changes: typing.Dict[int, int], entries=('labels',)) -> None:
        super().__init__()
        self.label_changes = label_changes
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

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
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

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
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

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
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

            np_entry = check_and_return(sample[entry], np.ndarray)
            sample[entry] = np.transpose(np_entry, self.permutation)
        return sample


class Squeeze(Transform):

    def __init__(self, entries=('images', 'labels'), squeeze_axis=None) -> None:
        super().__init__()
        self.entries = entries
        self.squeeze_axis = squeeze_axis

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

            np_entry = check_and_return(sample[entry], np.ndarray)
            sample[entry] = np_entry.squeeze(self.squeeze_axis)
        return sample


class UnSqueeze(Transform):

    def __init__(self, entries=('images', 'labels')) -> None:
        super().__init__()
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

            np_entry = check_and_return(sample[entry], np.ndarray)
            sample[entry] = np.expand_dims(np_entry, -1)
        return sample


class SizeCorrection(Transform):
    """Size correction transformation.

    Corrects the size, i.e. shape, of an array to a given reference shape.
    """

    def __init__(self, shape: typing.Tuple[typing.Union[None, int], ...], pad_value: int=0,
                 entries=('images', 'labels')) -> None:
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
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

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
                    np_entry = np_entry[tuple(slicing)]
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
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

            np_entry = check_and_return(sample[entry], np.ndarray)

            if np_mask.shape == np_entry.shape:
                np_entry[np_mask == self.mask_value] = self.masking_value
            else:
                mask_for_np_entry = np.repeat(np.expand_dims(np_mask, self.loop_axis),
                                              np_entry.shape[self.loop_axis], axis=self.loop_axis)
                np_entry[mask_for_np_entry == self.mask_value] = self.masking_value

            sample[entry] = np_entry
        return sample


def check_and_return(obj, type_):
    if not isinstance(obj, type_):
        raise ValueError("entry must be '{}'".format(type_.__name__))
    return obj
