import typing as t


# note: might be replaced with or wrap numpy.s_
class IndexExpression:

    def __init__(self, indexing: t.Union[int, tuple, t.List[int], t.List[tuple], t.List[list]] = None,
                 axis: t.Union[int, tuple] = None) -> None:
        """Defines the indexing of a chunk of raw data in the dataset.

        Args:
            indexing (int, tuple, list): The indexing. If :obj:`int` or list of :obj:`int`, individual entries of and axis
                are indexed. If :obj:`tuple` or list of :obj:`tuple`, the axis should be sliced.
            axis (int, tuple): The axis/axes to the corresponding indexing. If :obj:`tuple`, the length has to be equal to the
                list length of :obj:`indexing`
        """
        self.expression = None
        """list of :obj:`slice` objects defining the slicing each axis"""
        self.set_indexing(indexing, axis)

    def set_indexing(self, indexing: t.Union[int, tuple, slice, t.List[int], t.List[tuple], t.List[list]],
                     axis: t.Union[int, tuple] = None):
        if indexing is None:
            self.expression = slice(None)
            return

        if isinstance(indexing, int) or isinstance(indexing, tuple):
            indexing = [indexing]

        if axis is None:
            axis = tuple(range(len(indexing)))
        if isinstance(axis, int):
            axis = (axis,)

        expr = [slice(None) for _ in range(max(axis) + 1)]
        for a, index in zip(axis, indexing):
            if isinstance(index, int):
                expr[a] = index
            elif isinstance(index, (tuple, list)):
                start, stop = index
                expr[a] = slice(start, stop)
            elif isinstance(index, slice):
                expr[a] = index
            else:
                raise ValueError('Unknown type "{}" of index'.format(type(index)))

        # needs to be tuple otherwise exception from h5py while slicing
        self.expression = tuple(expr)

    def get_indexing(self):
        """
        Returns:
            list: a list tuples defining the indexing (i.e., None, index, (start, stop)) at each axis. Can be used to generate
            a new index expression.
        """
        indexing = []
        # todo(alainjungo): handle case when self.expression is of type slice
        for index in self.expression:
            if index is None:
                indexing.append(None)
            elif isinstance(index, slice):
                indexing.append((index.start, index.stop))
            elif isinstance(index, int):
                indexing.append(index)
            else:
                raise ValueError("only 'int', 'slice', and 'None' types possible in expression")
        return indexing
