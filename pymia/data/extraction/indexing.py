import abc
import typing

import numpy as np

import pymia.data.indexexpression as expr


class IndexingStrategy(abc.ABC):
    """Interface for indexing strategies that can be applied to images.

    .. automethod:: __call__
    .. automethod:: __repr__
    """

    @abc.abstractmethod
    def __call__(self, shape: tuple) -> typing.List[expr.IndexExpression]:
        """Calculate the indexes for a given shape

        Args:
            shape (tuple): The shape to determine the indexes for.

        Returns:
            list: The list of :class:`.IndexExpression` instances defining the indexes for an image shape.
        """
        pass

    def __repr__(self) -> str:
        """
        Returns:
            str: Representation of the strategy. Should include attributes such that it uniquely defines the strategy.
        """
        return self.__class__.__name__


class EmptyIndexing(IndexingStrategy):
    """An empty indexing strategy. This is useful when a strategy is required but entire images should be extracted."""

    def __call__(self, shape) -> typing.List[expr.IndexExpression]:
        return [expr.IndexExpression()]


class SliceIndexing(IndexingStrategy):

    def __init__(self, slice_axis: typing.Union[int, tuple] = 0) -> None:
        """Strategy to generate a slice-wise indexing.

        Args:
            slice_axis (int, tuple): The axis to be sliced. Multi-axis slicing can be achieved by providing a tuple of axes.
        """
        if isinstance(slice_axis, int):
            slice_axis = (slice_axis, )
        self.slice_axis = slice_axis

    def __call__(self, shape) -> typing.List[expr.IndexExpression]:
        indexing = []
        for axis in self.slice_axis:
            indexing.extend(expr.IndexExpression(i, axis) for i in range(shape[axis]))
        return indexing

    def __repr__(self) -> str:
        return '{} ({})'.format(self.__class__.__name__, self.slice_axis)


class VoxelWiseIndexing(IndexingStrategy):

    def __init__(self, image_dimension: int = 3):
        """Strategy to generate indices for every voxel of an image.

        Args:
            image_dimension (int): The image dimension without the dimension of the voxels itself.
        """
        self.shape = None
        self.indexing = None
        self.image_dimension = image_dimension

    def __call__(self, shape) -> typing.List[expr.IndexExpression]:
        if self.shape == shape:
            return self.indexing

        self.shape = shape  # save for later comparison to avoid calculating indices if the shape is equal
        shape_without_voxel = shape[0:self.image_dimension]
        indices = np.indices(shape_without_voxel)
        indices = indices.reshape((indices.shape[0], np.prod(indices.shape[1:])))
        indices = indices.transpose()
        self.indexing = [expr.IndexExpression(idx.tolist()) for idx in indices]
        return self.indexing


class PatchWiseIndexing(IndexingStrategy):

    def __init__(self, patch_shape: tuple, ignore_incomplete=True) -> None:
        """Strategy to generate indices for patches (sub-volumes) of an image.

        Args:
            patch_shape (tuple): The patch shape.
            ignore_incomplete (bool): If even division of image by patch shape ignore incomplete patch on True.
                Boundary condition.
        """
        super().__init__()
        self.patch_shape = patch_shape
        self.image_dimension = len(patch_shape)
        self.ignore_incomplete = ignore_incomplete
        self.prev_shape = None
        self.prev_indexing = None

    def __call__(self, shape) -> typing.List[expr.IndexExpression]:
        if shape == self.prev_shape:
            return self.prev_indexing

        shape_without_voxel = shape[:self.image_dimension]
        index_count = np.divide(shape_without_voxel, self.patch_shape)
        index_count = np.floor(index_count) if self.ignore_incomplete else np.ceil(index_count)
        index_count = index_count.astype('int')

        indices = np.indices(index_count).reshape(index_count.size, -1).T
        index_ranges = np.stack([indices, indices + 1], axis=-1)
        index_ranges *= np.asarray(self.patch_shape)[np.newaxis, :, np.newaxis]
        indexing = [expr.IndexExpression(idx.tolist()) for idx in index_ranges]

        self.prev_indexing = indexing
        self.prev_shape = shape
        return indexing

    def __repr__(self) -> str:
        return '{} (patch shape={}, ignore incomplete={})'.format(self.__class__.__name__,
                                                                  self.patch_shape,
                                                                  self.ignore_incomplete)
