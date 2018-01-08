import abc

import numpy as np
import torch.utils.data as data
import torch.utils.data.sampler as smplr


class SelectionStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, sample):
        pass


class NonBlackSelection(SelectionStrategy):

    def __init__(self, black_value: float=.0) -> None:
        self.black_value = black_value

    def __call__(self, sample):
        return np.any(sample['images'] > self.black_value)


class WithForegroundSelection(SelectionStrategy):

    def __call__(self, sample):
        return np.any(sample['labels'])


class SubjectSelection(SelectionStrategy):

    def __init__(self, subjects) -> None:
        self.subjects = subjects

    def __call__(self, sample):
        return sample['subject'] in self.subjects


class ComposeSelection(SelectionStrategy):

    def __init__(self, strategies) -> None:
        self.strategies = strategies

    def __call__(self, sample):
        return all(strategy(sample) for strategy in self.strategies)


def select_indices(data_source: data.Dataset, selection_strategy: SelectionStrategy):
    selected_indices = []
    for i, sample in enumerate(data_source):
        if selection_strategy(sample):
            selected_indices.append(i)
    return selected_indices


class SubsetSequentialSampler(smplr.Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
