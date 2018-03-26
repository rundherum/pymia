import abc
import typing as t

import numpy as np
import torch


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
                continue

            np_entry = _check_and_return(sample[entry], np.ndarray)
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
        min_, max_ = arr.min(), arr.max()
        if min_ == max_:
            raise ValueError('cannot normalize when min == max')
        arr = (arr - min_) / (max_ - min_) * (upper - lower) + lower
        return arr


class IntensityNormalization(Transform):

    def __init__(self, loop_axis=None, entries=('images',)) -> None:
        super().__init__()
        self.loop_axis = loop_axis
        self.entries = entries
        self.normalize_fn = self._normalize

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                continue

            np_entry = _check_and_return(sample[entry], np.ndarray)
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
                continue

            np_entry = _check_and_return(sample[entry], np.ndarray)

            if self.loop_axis is None:
                np_entry = self.lambda_fn(np_entry)
            else:
                slicing = [slice(None) for _ in range(np_entry.ndim)]
                for i in range(np_entry.shape[self.loop_axis]):
                    slicing[self.loop_axis] = i
                    np_entry[slicing] = self.lambda_fn(np_entry[slicing])
            sample[entry] = np_entry
        return sample


class ClipPercentile(Transform):

    def __init__(self, upper_percentile: float, lower_percentile: float = None, entries=('images',)) -> None:
        super().__init__()
        self.upper_percentile = upper_percentile
        self.lower_percentile = lower_percentile
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                continue

            np_entry = _check_and_return(sample[entry], np.ndarray)

            upper_max = np.percentile(np_entry, self.upper_percentile)
            np_entry[np_entry > upper_max] = upper_max
            lower_max = np.percentile(np_entry, self.lower_percentile)
            np_entry[np_entry < lower_max] = lower_max

            sample[entry] = np_entry
        return sample


class Relabel(Transform):

    def __init__(self, label_changes: t.Dict[int, int], entries=('labels',)) -> None:
        super().__init__()
        self.label_changes = label_changes
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                continue
            np_entry = _check_and_return(sample[entry], np.ndarray)
            for new_label, old_label in self.label_changes.items():
                np_entry[np_entry == old_label] = new_label
            sample[entry] = np_entry
        return sample


class ToTorchTensor(Transform):

    def __init__(self, entries=('images', 'labels')) -> None:
        super().__init__()
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                continue

            np_entry = _check_and_return(sample[entry], np.ndarray)
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
                continue
            np_entry = _check_and_return(sample[entry], np.ndarray)
            sample[entry] = np.transpose(np_entry, self.permutation)
        return sample


class Squeeze(Transform):

    def __init__(self, entries=('images', 'labels')) -> None:
        super().__init__()
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                continue

            np_entry = _check_and_return(sample[entry], np.ndarray)
            sample[entry] = np_entry.squeeze()
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
                continue

            np_entry = _check_and_return(sample[entry], np.ndarray)
            if not len(self.shape) <= len(np_entry.shape):
                raise ValueError('Shape dimension needs to be less or equal to {}'.format(len(np_entry.shape)))

            for idx, size in enumerate(self.shape):
                if size is not None and size < np_entry.shape[idx]:
                    # crop current dimension
                    before = (np_entry.shape[idx] - size) // 2
                    after = np_entry.shape[idx] - (np_entry.shape[idx] - size) // 2 + ((np_entry.shape[idx] - size) % 2)
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


def _check_and_return(obj, type_):
    if not isinstance(obj, type_):
        raise ValueError("entry must be '{}'".format(type_.__name__))
    return obj
