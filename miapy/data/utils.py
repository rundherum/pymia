import typing as t
import os


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

        expr = [None for _ in range(max(axis) + 1)]
        for a, index in zip(axis, indexing):
            if isinstance(index, int):
                expr[a] = index
            else:
                start, stop = index
                expr[a] = slice(start, stop)

        # needs to be tuple otherwise exception from h5py while slicing
        self.expression = expr[0] if len(expr) == 1 else tuple(expr)


def create_dir_if_not_exists(path_name, is_file=False) -> None:
    if is_file:
        path_name = os.path.dirname(path_name)

    if not os.path.exists(path_name):
        os.makedirs(path_name)