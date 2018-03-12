import abc
import typing as t

import numpy as np

import miapy.data.indexexpression as expr


class IndexingStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, shape, offset) -> t.List[expr.IndexExpression]:
        # return list of indexes by giving shape
        pass


class SliceIndexing(IndexingStrategy):

    def __init__(self, slice_axis=0) -> None:
        self.slice_axis = slice_axis

    def __call__(self, shape, offset=0) -> t.List[expr.IndexExpression]:
        count = shape[self.slice_axis]
        return [expr.IndexExpression(i, self.slice_axis) for i in range(offset, offset + count)]


class VoxelWiseIndexing(IndexingStrategy):

    def __init__(self):
        self.shape = None
        self.indexing = None

    def __call__(self, shape, offset=0) -> t.List[expr.IndexExpression]:
        if self.shape == shape:
            return self.indexing

        self.shape = shape  # save for later comparison
        shape_without_voxel = shape[:-1]
        indices = np.indices(shape_without_voxel)
        indices = indices.reshape((indices.shape[0], np.prod(indices.shape[1:])))
        indices = indices.transpose()
        self.indexing = [expr.IndexExpression(idx.tolist()) for idx in indices]
        return self.indexing
