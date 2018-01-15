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

    def __init__(self, lower, upper, loop_axis=None) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.loop_axis = loop_axis

    def __call__(self, sample: dict) -> dict:
        np_image = _check_and_return(sample['images'], np.ndarray)

        if self.loop_axis is None:
            np_image = self._normalize(np_image, self.lower, self.upper)
        else:
            slicing = [slice(None) for _ in range(np_image.ndim)]
            for i in range(np_image.shape[self.loop_axis]):
                slicing[self.loop_axis] = i
                np_image[slicing] = self._normalize(np_image[slicing], self.lower, self.upper)

        sample['images'] = np_image
        return sample

    @staticmethod
    def _normalize(arr: np.ndarray, lower, upper):
        min_, max_ = arr.min(), arr.max()
        if min_ == max_:
            raise ValueError('cannot normalize when min == max')
        arr = (arr - min_) / (max_ - min_) * (upper - lower) + lower
        return arr


class IntensityNormalization(Transform):

    def __init__(self, loop_axis=None) -> None:
        super().__init__()
        self.loop_axis = loop_axis

    def __call__(self, sample: dict) -> dict:
        np_image = _check_and_return(sample['images'], np.ndarray)

        if self.loop_axis is None:
            np_image = self._normalize(np_image)
        else:
            slicing = [slice(None) for _ in range(np_image.ndim)]
            for i in range(np_image.shape[self.loop_axis]):
                slicing[self.loop_axis] = i
                np_image[slicing] = self._normalize(np_image[slicing])

        sample['images'] = np_image
        return sample

    @staticmethod
    def _normalize(arr: np.ndarray):
        return (arr - arr.mean()) / arr.std()


class ClipPercentile(Transform):

    def __init__(self, upper_percentile: float, lower_percentile: float = None) -> None:
        super().__init__()
        self.upper_percentile = upper_percentile
        self.lower_percentile = lower_percentile

    def __call__(self, sample: dict) -> dict:
        np_image = _check_and_return(sample['images'], np.ndarray)

        upper_max = np.percentile(np_image, self.upper_percentile)
        np_image[np_image > upper_max] = upper_max
        lower_max = np.percentile(np_image, self.lower_percentile)
        np_image[np_image < lower_max] = lower_max

        sample['images'] = np_image
        return sample


class Relabel(Transform):

    def __init__(self, label_changes: t.Dict[int, int]) -> None:
        super().__init__()
        self.label_changes = label_changes

    def __call__(self, sample: dict) -> dict:
        np_label = _check_and_return(sample['labels'], np.ndarray)
        for new_label, old_label in self.label_changes.items():
            np_label[np_label == old_label] = new_label

        sample['labels'] = np_label
        return sample


class ToTensor(Transform):

    def __init__(self, image_dims: int=3) -> None:
        super().__init__()
        self.image_dims = image_dims

    def __call__(self, sample: dict) -> dict:
        np_image = _check_and_return(sample['images'], np.ndarray)
        sample['images'] = torch.from_numpy(np_image)
        if 'labels' in sample:
            np_label = _check_and_return(sample['labels'], np.ndarray)
            sample['labels'] = torch.from_numpy(np_label)
        return sample


class Permute(Transform):

    def __init__(self, permutation: tuple, label_permutation=None) -> None:
        super().__init__()
        self.permutation = permutation
        if label_permutation is None:
            label_permutation = self.permutation
        self.label_permutation = label_permutation

    def __call__(self, sample: dict) -> dict:
        np_image = _check_and_return(sample['images'], np.ndarray)
        sample['images'] = np.transpose(np_image, self.permutation)

        if 'labels' in sample:
            np_label = _check_and_return(sample['labels'], np.ndarray)
            sample['labels'] = np.transpose(np_label, self.label_permutation)
        return sample


class Squeeze(Transform):

    def __call__(self, sample: dict) -> dict:
        np_image = _check_and_return(sample['images'], np.ndarray)
        sample['images'] = np_image.squeeze()

        if 'labels' in sample:
            np_label = _check_and_return(sample['labels'], np.ndarray)
            sample['labels'] = np_label.squeeze()
        return sample


def _check_and_return(obj, type_):
    if not isinstance(obj, type_):
        raise ValueError("entry must be '{}'".format(type_.__name__))
    return obj