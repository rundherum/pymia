import typing

import numpy as np

from pymia.data import subjectfile as subj
import pymia.data.transformation as tfm
import pymia.data.conversion as conv
import pymia.data.definition as defs
from . import callback as cb
from . import fileloader as load


def default_concat(data: typing.List[np.ndarray]) -> np.ndarray:
    """Default concatenation function used to combine all entries from a category (e.g. T1, T2 data from "images" category)
    in :meth:`.Traverser.traverse`

    Args:
        data (list): List of numpy.ndarray entries to be concatenated.

    Returns:
        numpy.ndarray: Concatenated entry.
    """
    return np.stack(data, axis=-1)


class Traverser:

    def __init__(self, categories: typing.Union[str, typing.Tuple[str, ...]] = None):
        """Class managing the dataset creation process.

        Args:
            categories (str or tuple of str): The categories to traverse. If None, then all categories of a
                :class:`.SubjectFile` will be traversed.
        """
        if isinstance(categories, str):
            categories = (categories, )
        self.categories = categories

    def traverse(self, subject_files: typing.List[subj.SubjectFile], load=load.LoadDefault(),
                 callback: cb.Callback = None,
                 transform: tfm.Transform = None, concat_fn=default_concat):
        """Controls the actual dataset creation. It goes through the file list, loads the files,
        applies transformation to the data, and calls the callbacks to do the storing (or other stuff).

        Args:
            subject_files (list): list of :class:`SubjectFile` to be processes.
            load (callable): A load function or :class:`.Load` instance that performs the data loading
            callback (.Callback): A callback or composed (:class:`.ComposeCallback`) callback performing the storage of the
                loaded data (and other things such as logging).
            transform (.Transform): Transformation to be applied to the data after loading
                and before :meth:`Callback.on_subject` is called
            concat_fn (callable): Function that concatenates all the entries of a category
                (e.g. T1, T2 data from "images" category). Default is :func:`default_concat`.
        """
        if len(subject_files) == 0:
            raise ValueError('No files')
        if not isinstance(subject_files[0], subj.SubjectFile):
            raise ValueError('files must be of type {}'.format(subj.SubjectFile.__class__.__name__))
        if callback is None:
            raise ValueError('callback can not be None')

        if self.categories is None:
            self.categories = subject_files[0].categories

        callback_params = {defs.KEY_SUBJECT_FILES: subject_files}
        for category in self.categories:
            callback_params.setdefault(defs.KEY_CATEGORIES, []).append(category)
            callback_params[defs.KEY_PLACEHOLDER_NAMES.format(category)] = self._get_names(subject_files, category)
        callback.on_start(callback_params)

        # looping over the subject files and calling callbacks
        for subject_index, subject_file in enumerate(subject_files):
            transform_params = {defs.KEY_SUBJECT_INDEX: subject_index}
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
                transform_params[defs.KEY_PLACEHOLDER_PROPERTIES.format(category)] = category_property

            if transform:
                transform_params = transform(transform_params)

            callback.on_subject({**transform_params, **callback_params})

        callback.on_end(callback_params)

    @staticmethod
    def _get_names(subject_files: typing.List[subj.SubjectFile], category: str) -> list:
        names = subject_files[0].categories[category].entries.keys()
        if not all(s.categories[category].entries.keys() == names for s in subject_files):
            raise ValueError('Inconsistent {} identifiers in the subject list'.format(category))
        return list(names)
