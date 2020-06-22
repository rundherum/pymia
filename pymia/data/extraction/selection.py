import abc

import numpy as np

import pymia.data.definition as defs
from . import datasource as ds


class SelectionStrategy(abc.ABC):
    """Interface for selecting indices according some rule.

    .. automethod:: __call__
    .. automethod:: __repr__
    """

    @abc.abstractmethod
    def __call__(self, sample: dict) -> bool:
        """

        Args:
            sample (dict): An extracted from :class:`.PymiaDatasource`.

        Returns:
            bool: Whether or not the sample should be considered.

        """
        pass

    def __repr__(self) -> str:
        """
        Returns:
            str: Representation of the strategy. Should include attributes such that it uniquely defines the strategy.
        """
        return self.__class__.__name__


class NonConstantSelection(SelectionStrategy):

    def __init__(self, loop_axis=None) -> None:
        super().__init__()
        self.loop_axis = loop_axis

    def __call__(self, sample) -> bool:
        image_data = sample[defs.KEY_IMAGES]

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

    def __init__(self, black_value: float = 0.0) -> None:
        # todo: add something to do this for one sequence only, too -> loop over one dim
        self.black_value = black_value

    def __call__(self, sample) -> bool:
        return (sample[defs.KEY_IMAGES] > self.black_value).any()

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, self.black_value)


class PercentileSelection(SelectionStrategy):

    def __init__(self, percentile: float) -> None:
        # todo: add something to do this for one sequence only, too -> loop over one dim
        self.percentile = percentile

    def __call__(self, sample) -> bool:
        image_data = sample[defs.KEY_IMAGES]

        percentile_value = np.percentile(image_data, self.percentile)
        return (image_data >= percentile_value).all()

    def __repr__(self) -> str:
        return '{} ({})'.format(self.__class__.__name__, self.percentile)


class WithForegroundSelection(SelectionStrategy):

    def __call__(self, sample) -> bool:
        return (sample[defs.KEY_LABELS]).any()


class SubjectSelection(SelectionStrategy):
    """Select subjects by their name or index."""

    def __init__(self, subjects) -> None:
        if isinstance(subjects, int):
            subjects = (subjects, )
        if isinstance(subjects, str):
            subjects = (subjects, )
        self.subjects = subjects

    def __call__(self, sample) -> bool:
        return sample[defs.KEY_SUBJECT] in self.subjects or sample[defs.KEY_SUBJECT_INDEX] in self.subjects

    def __repr__(self) -> str:
        return '{} ({})'.format(self.__class__.__name__, ','.join(self.subjects))


class ComposeSelection(SelectionStrategy):

    def __init__(self, strategies) -> None:
        self.strategies = strategies

    def __call__(self, sample) -> bool:
        return all(strategy(sample) for strategy in self.strategies)

    def __repr__(self) -> str:
        return '|'.join(repr(s) for s in self.strategies)


def select_indices(data_source: ds.PymiaDatasource, selection_strategy: SelectionStrategy):
    selected_indices = []
    for i, sample in enumerate(data_source):
        if selection_strategy(sample):
            selected_indices.append(i)
    return selected_indices
