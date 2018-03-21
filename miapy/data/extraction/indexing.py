import abc
import typing as t

import numpy as np

import miapy.data.indexexpression as expr


class IndexingStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, shape) -> t.List[expr.IndexExpression]:
        # return list of indexes by giving shape
        pass


class SliceIndexing(IndexingStrategy):

    def __init__(self, slice_axis: t.Union[int, tuple]=0) -> None:
        if isinstance(slice_axis, int):
            slice_axis = (slice_axis,)
        self.slice_axis = slice_axis

    def __call__(self, shape) -> t.List[expr.IndexExpression]:
        indexing = []
        for axis in self.slice_axis:
            indexing.extend(expr.IndexExpression(i, axis) for i in range(shape[axis]))
        return indexing


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
