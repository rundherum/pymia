import typing as t


# could maybe be replaced or wrapper with numpy.s_
class IndexExpression:

    def __init__(self, indexing: t.Union[int, tuple, t.List[int], t.List[tuple]]=None,
                 axis: t.Union[int, tuple]=None) -> None:
        self.expression = None
        self.set_indexing(indexing, axis)

    def set_indexing(self, indexing: t.Union[int, tuple, t.List[int], t.List[tuple]], axis: t.Union[int, tuple]=None):
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
            elif isinstance(index, tuple):
                start, stop = index
                expr[a] = slice(start, stop)

        # needs to be tuple otherwise exception from h5py while slicing
        self.expression = tuple(expr)

    def get_indexing(self):
        indexing = []
        for index in self.expression:
            if index is None:
                indexing.append(None)
            elif isinstance(index, slice):
                indexing.append((index.start, index.stop))
            elif isinstance(index, int):
                indexing.append(index)
            else:
                raise ValueError("only 'int', 'slice', and 'None' types possible in expression")
        return indexing if len(indexing) > 1 else indexing[0]

