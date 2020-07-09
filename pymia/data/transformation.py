import abc
import typing

import numpy as np

import pymia.data.definition as defs


ENTRY_NOT_EXTRACTED_ERR_MSG = 'Transform can not be applied because entry "{}" was not extracted'

raise_error_if_entry_not_extracted = True


# follows the principle of torchvision transform
class Transform(abc.ABC):
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


class LoopEntryTransform(Transform, abc.ABC):

    def __init__(self, loop_axis=None, entries=()) -> None:
        super().__init__()
        self.loop_axis = loop_axis
        self.entries = entries

    @staticmethod
    def loop_entries(sample: dict, fn, entries, loop_axis=None):
        for entry in entries:
            if entry not in sample:
                if raise_error_if_entry_not_extracted:
                    raise ValueError(ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))
                continue

            np_entry = check_and_return(sample[entry], np.ndarray)

            if loop_axis is None:
                np_entry = fn(np_entry, entry)
            else:
                slicing = [slice(None) for _ in range(np_entry.ndim)]
                for i in range(np_entry.shape[loop_axis]):
                    slicing[loop_axis] = i
                    np_entry[tuple(slicing)] = fn(np_entry[tuple(slicing)], entry, i)
            sample[entry] = np_entry
        return sample

    def __call__(self, sample: dict) -> dict:
        return self.loop_entries(sample, self.transform_entry, self.entries, self.loop_axis)

    @abc.abstractmethod
    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        pass


class IntensityRescale(LoopEntryTransform):

    def __init__(self, lower, upper, loop_axis=None, entries=(defs.KEY_IMAGES, )) -> None:
        super().__init__(loop_axis=loop_axis, entries=entries)
        self.lower = lower
        self.upper = upper

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        return self._normalize(np_entry, self.lower, self.upper)

    @staticmethod
    def _normalize(arr: np.ndarray, lower, upper):
        dtype = arr.dtype
        min_, max_ = arr.min(), arr.max()
        if min_ == max_:
            raise ValueError('cannot normalize when min == max')
        arr = (arr - min_) / (max_ - min_) * (upper - lower) + lower
        return arr.astype(dtype)


class IntensityNormalization(LoopEntryTransform):

    def __init__(self, loop_axis=None, entries=(defs.KEY_IMAGES, )) -> None:
        super().__init__(loop_axis=loop_axis, entries=entries)
        self.normalize_fn = self._normalize

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        if not np.issubdtype(np_entry.dtype, np.floating):
            raise ValueError('Array must be floating type')
        return self._normalize(np_entry)

    @staticmethod
    def _normalize(arr: np.ndarray):
        return (arr - arr.mean()) / arr.std()


class LambdaTransform(LoopEntryTransform):

    def __init__(self, lambda_fn, loop_axis=None, entries=(defs.KEY_IMAGES, )) -> None:
        super().__init__(loop_axis=loop_axis, entries=entries)
        self.lambda_fn = lambda_fn

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        return self.lambda_fn(np_entry)


class ClipPercentile(LoopEntryTransform):

    def __init__(self, upper_percentile: float, lower_percentile: float = None,
                 loop_axis=None, entries=(defs.KEY_IMAGES, )) -> None:
        super().__init__(loop_axis=loop_axis, entries=entries)
        self.upper_percentile = upper_percentile
        if lower_percentile is None:
            lower_percentile = 100 - upper_percentile
        self.lower_percentile = lower_percentile

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        return self._clip(np_entry)

    def _clip(self, arr: np.ndarray):
        upper_max = np.percentile(arr, self.upper_percentile)
        arr[arr > upper_max] = upper_max
        lower_max = np.percentile(arr, self.lower_percentile)
        arr[arr < lower_max] = lower_max
        return arr


class Relabel(LoopEntryTransform):

    def __init__(self, label_changes: typing.Dict[int, int], entries=(defs.KEY_LABELS, )) -> None:
        super().__init__(loop_axis=None, entries=entries)
        self.label_changes = label_changes

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        for new_label, old_label in self.label_changes.items():
            np_entry[np_entry == old_label] = new_label
        return np_entry


class Reshape(LoopEntryTransform):

    def __init__(self, shapes: dict) -> None:
        """Initializes a new instance of the Reshape class.

        Args:
            shapes (dict): A dict with keys being the entries and the values the new shapes of the entries.
                E.g. shapes = {defs.KEY_IMAGES: (-1, 4), defs.KEY_LABELS : (-1, 1)}
        """
        super().__init__(loop_axis=None, entries=tuple(shapes.keys()))
        self.shapes = shapes

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        return np.reshape(np_entry, self.shapes[entry])


class Permute(LoopEntryTransform):

    def __init__(self, permutation: tuple, entries=(defs.KEY_IMAGES, defs.KEY_LABELS)) -> None:
        super().__init__(loop_axis=None, entries=entries)
        self.permutation = permutation

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        return np.transpose(np_entry, self.permutation)


class Squeeze(LoopEntryTransform):

    def __init__(self, entries=(defs.KEY_IMAGES, defs.KEY_LABELS), squeeze_axis=None) -> None:
        super().__init__(loop_axis=None, entries=entries)
        self.squeeze_axis = squeeze_axis

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        return np_entry.squeeze(self.squeeze_axis)


class UnSqueeze(LoopEntryTransform):

    def __init__(self, axis=-1, entries=(defs.KEY_IMAGES, defs.KEY_LABELS)) -> None:
        super().__init__(loop_axis=None, entries=entries)
        self.axis = axis

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        return np.expand_dims(np_entry, self.axis)


class SizeCorrection(Transform):
    """Size correction transformation.

    Corrects the size, i.e. shape, of an array to a given reference shape.
    """

    def __init__(self, shape: typing.Tuple[typing.Union[None, int], ...], pad_value: int = 0,
                 entries=(defs.KEY_IMAGES, defs.KEY_LABELS)) -> None:
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

    def __init__(self, mask_key: str, mask_value: int = 0, masking_value: float = 0.0,
                 loop_axis=None, entries=(defs.KEY_IMAGES, defs.KEY_LABELS)) -> None:
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


class RandomCrop(LoopEntryTransform):

    def __init__(self, size: tuple, loop_axis=None, entries=(defs.KEY_IMAGES, defs.KEY_LABELS)) -> None:
        super().__init__(loop_axis, entries)
        self.size = size
        self.slices = None

    def transform_entry(self, np_entry, entry, loop_i=None) -> np.ndarray:
        current_size = np_entry.shape[-len(self.size):]
        dim_diff = np_entry.ndim - len(self.size)

        if entry == self.entries[0]:
            # define offset for first of the entries -> apply on the others
            mins = np.zeros(3, dtype=np.int)
            maxs = np.subtract(current_size, self.size)
            if (maxs < 0).any():
                raise ValueError('current size is not large enough for crop')

            offset = np.random.randint(mins, maxs + 1, len(self.size))
            self.slices = tuple([slice(offset[i], offset[i]+self.size[i]) for i in range(len(offset))])

        slices = dim_diff*(slice(None),) + self.slices
        return np_entry[slices]


def check_and_return(obj, type_):
    if not isinstance(obj, type_):
        raise ValueError("entry must be '{}'".format(type_.__name__))
    return obj
