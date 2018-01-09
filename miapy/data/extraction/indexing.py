import abc
import typing as t

import miapy.data.indexexpression as util


class IndexingStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, shape, offset) -> t.List[util.IndexExpression]:
        # return list of indexes by giving shape
        pass


class SliceIndexing(IndexingStrategy):

    def __init__(self, slice_axis=0) -> None:
        self.slice_axis = slice_axis

    def __call__(self, shape, offset=0) -> t.List[util.IndexExpression]:
        count = shape[self.slice_axis]
        return [util.IndexExpression(i) for i in range(offset, offset + count)]
