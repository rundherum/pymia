import abc
import typing as t

import numpy as np
import torch


# follows the principle of torchvision transform
class Transform(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, sample: dict) -> dict:
        pass


class Compose(Transform):

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

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                continue

            np_entry = _check_and_return(sample[entry], np.ndarray)

            if self.loop_axis is None:
                np_entry = self._normalize(np_entry)
            else:
                slicing = [slice(None) for _ in range(np_entry.ndim)]
                for i in range(np_entry.shape[self.loop_axis]):
                    slicing[self.loop_axis] = i
                    np_entry[slicing] = self._normalize(np_entry[slicing])
            sample[entry] = np_entry
        return sample

    @staticmethod
    def _normalize(arr: np.ndarray):
        return (arr - arr.mean()) / arr.std()


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


class ToTensor(Transform):

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


def _check_and_return(obj, type_):
    if not isinstance(obj, type_):
        raise ValueError("entry must be '{}'".format(type_.__name__))
    return obj
