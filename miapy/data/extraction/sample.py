import abc
import typing as t

import numpy as np
import torch.utils.data as data
import torch.utils.data.sampler as smplr


class SelectionStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, sample) -> bool:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__


class NonConstantSelection(SelectionStrategy):

    def __init__(self, loop_axis=None) -> None:
        super().__init__()
        self.loop_axis = loop_axis

    def __call__(self, sample) -> bool:
        image_data = sample['images']

        if self.loop_axis is None:
            return not self._all_equal(image_data)

        slicing = [slice(None) for _ in range(image_data.ndim)]
        for i in range(image_data.shape[self.loop_axis]):
            slicing[self.loop_axis] = i
            slice_data = image_data[slicing]
            if not self._all_equal(slice_data):
                return True
        return False

    @staticmethod
    def _all_equal(image_data):
        return np.all(image_data == image_data.ravel()[0])

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, self.loop_axis)


class NonBlackSelection(SelectionStrategy):

    def __init__(self, black_value: float=.0) -> None:
        # todo: add something to do this for one sequence only, too -> loop over one dim
        self.black_value = black_value

    def __call__(self, sample) -> bool:
        return (sample['images'] > self.black_value).any()

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, self.black_value)


class PercentileSelection(SelectionStrategy):

    def __init__(self, percentile: float) -> None:
        # todo: add something to do this for one sequence only, too -> loop over one dim
        self.percentile = percentile

    def __call__(self, sample) -> bool:
        image_data = sample['images']

        percentile_value = np.percentile(image_data, self.percentile)
        return (image_data >= percentile_value).all()

    def __repr__(self) -> str:
        return '{} ({})'.format(self.__class__.__name__, self.percentile)


class WithForegroundSelection(SelectionStrategy):

    def __call__(self, sample) -> bool:
        return (sample['labels']).any()


class SubjectSelection(SelectionStrategy):
    """Select subjects by their name or index."""

    def __init__(self, subjects) -> None:
        if isinstance(subjects, int):
            subjects = (subjects, )
        if isinstance(subjects, str):
            subjects = (subjects, )
        self.subjects = subjects

    def __call__(self, sample) -> bool:
        return sample['subject'] in self.subjects or sample['subject_index'] in self.subjects

    def __repr__(self) -> str:
        return '{} ({})'.format(self.__class__.__name__, ','.join(self.subjects))


class ComposeSelection(SelectionStrategy):

    def __init__(self, strategies) -> None:
        self.strategies = strategies

    def __call__(self, sample) -> bool:
        return all(strategy(sample) for strategy in self.strategies)

    def __repr__(self) -> str:
        return '|'.join(repr(s) for s in self.strategies)


def select_indices(data_source: data.Dataset, selection_strategy: SelectionStrategy):
    selected_indices = []
    for i, sample in enumerate(data_source):
        if selection_strategy(sample):
            selected_indices.append(i)
    return selected_indices


class SubsetSequentialSampler(smplr.Sampler):
    """Samples elements sequential from a given list of indices, without replacement."""

    def __init__(self, indices):
        super().__init__(None)
        self.indices = indices

    def __iter__(self):
        return (idx for idx in self.indices)

    def __len__(self):
        return len(self.indices)
