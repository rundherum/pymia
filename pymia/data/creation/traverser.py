import abc
import typing as t

import numpy as np

from pymia.data import subjectfile as subj
import pymia.data.transformation as tfm
import pymia.data.conversion as conv
from . import callback as cb
from . import fileloader as load


class Traverser(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def traverse(self, files: list, loader: load.Load, callbacks: t.List[cb.Callback]=None,
                 transform: tfm.Transform=None):
        pass


def default_concat(data: t.List[np.ndarray]) -> np.ndarray:
    return np.stack(data, axis=-1)


class SubjectFileTraverser(Traverser):

    def __init__(self, categories: t.Union[str, t.Tuple[str, ...]]=None):
        """Initializes a new instance of the SubjectFileTraverser class.

        Args:
            categories (str or tuple of str): The categories to traverse. If None, then all categories of a SubjectFile
                will be traversed.
        """
        if isinstance(categories, str):
            categories = (categories, )
        self.categories = categories

    def traverse(self, subject_files: t.List[subj.SubjectFile], load=load.LoadDefault(), callback: cb.Callback=None,
                 transform: tfm.Transform=None, concat_fn=default_concat):
        if len(subject_files) == 0:
            raise ValueError('No files')
        if not isinstance(subject_files[0], subj.SubjectFile):
            raise ValueError('files must be of type {}'.format(subj.SubjectFile.__class__.__name__))
        if callback is None:
            raise ValueError('callback can not be None')

        if self.categories is None:
            self.categories = subject_files[0].categories

        callback_params = {'subject_files': subject_files}
        for category in self.categories:
            callback_params.setdefault('categories', []).append(category)
            callback_params['{}_names'.format(category)] = self._get_names(subject_files, category)
        callback.on_start(callback_params)

        # looping over the subject files and calling callbacks
        for subject_index, subject_file in enumerate(subject_files):
            transform_params = {'subject_index': subject_index}
            for category in self.categories:

                category_list = []
                category_property = None  # type: conv.ImageProperties
                for id_, file_path in subject_file.categories[category].entries.items():
                    np_data, data_property = load(file_path, id_, category, subject_file.subject)
                    category_list.append(np_data)
                    if category_property is None:  # only required once
                        category_property = data_property

                category_data = concat_fn(category_list)
                transform_params[category] = category_data
                transform_params['{}_properties'.format(category)] = category_property

            if transform:
                transform_params = transform(transform_params)

            callback.on_subject({**transform_params, **callback_params})

        callback.on_end(callback_params)

    @staticmethod
    def _get_names(subject_files: t.List[subj.SubjectFile], category: str) -> list:
        names = subject_files[0].categories[category].entries.keys()
        if not all(s.categories[category].entries.keys() == names for s in subject_files):
            raise ValueError('Inconsistent {} identifiers in the subject list'.format(category))
        return list(names)
