import os
import typing

import numpy as np

import pymia.data.conversion as conv
import pymia.data.definition as defs
import pymia.data.indexexpression as expr
import pymia.data.subjectfile as subj
from . import writer as wr


class Callback:
    """Base class for the interaction with the dataset creation.

    Implementations of the :class:`.Callback` class can be provided to :meth:`.SubjectFileTraverser.traverse` in order to
    write/process specific information of the original data.
    """

    def on_start(self, params: dict):
        """Called at the beginning of :meth:`.SubjectFileTraverser.traverse`.

        Args:
            params (dict): Parameters provided by the :class:`.SubjectFileTraverser`. The provided parameters will differ from
                :meth:`.Callback.on_subject`.
        """
        pass

    def on_end(self, params: dict):
        """Called at the end of :meth:`.SubjectFileTraverser.traverse`.

        Args:
            params (dict): Parameters provided by the :class:`.SubjectFileTraverser`. The provided parameters will differ from
                :meth:`.Callback.on_subject`.
        """
        pass

    def on_subject(self, params: dict):
        """Called for each subject of :meth:`.SubjectFileTraverser.traverse`.


        Args:
            params (dict): Parameters provided by the :class:`.SubjectFileTraverser` containing subject specific information
                and data.
        """
        pass


class ComposeCallback(Callback):

    def __init__(self, callbacks: typing.List[Callback]) -> None:
        """Composes many :class:`.Callback` instances and behaves like an single :class:`.Callback` instance.

        This class allows passing multiple :class:`.Callback` to :meth:`.SubjectFileTraverser.traverse`.

        Args:
            callbacks (list): A list of :class:`.Callback` instances.
        """
        self.callbacks = callbacks

    def on_start(self, params: dict):
        """see :meth:`.Callback.on_start`."""
        for c in self.callbacks:
            c.on_start(params)

    def on_end(self, params: dict):
        """see :meth:`.Callback.on_end`."""
        for c in self.callbacks:
            c.on_end(params)

    def on_subject(self, params: dict):
        """see :meth:`.Callback.on_subject`."""
        for c in self.callbacks:
            c.on_subject(params)


class MonitoringCallback(Callback):

    def on_start(self, params: dict):
        """see :meth:`.Callback.on_start`."""
        print('start dataset creation')

    def on_subject(self, params: dict):
        """see :meth:`.Callback.on_subject`."""
        index = params[defs.KEY_SUBJECT_INDEX]
        subject_files = params[defs.KEY_SUBJECT_FILES]
        print('[{}/{}] {}'.format(index + 1, len(subject_files), subject_files[index].subject))

    def on_end(self, params: dict):
        """see :meth:`.Callback.on_end`."""
        print('dataset creation finished')


class WriteDataCallback(Callback):

    def __init__(self, writer: wr.Writer) -> None:
        """Callback that writes the raw data to the dataset.

        Args:
            writer (.creation.writer.Writer): The writer used to write the data.
        """
        self.writer = writer

    def on_subject(self, params: dict):
        """see :meth:`.Callback.on_subject`."""
        subject_files = params[defs.KEY_SUBJECT_FILES]
        subject_index = params[defs.KEY_SUBJECT_INDEX]

        max_digits = len(str(len(subject_files)))
        index_str = '{{:0{}}}'.format(max_digits).format(subject_index)

        for category in params[defs.KEY_CATEGORIES]:
            data = params[category]
            self.writer.write('{}/{}'.format(defs.LOC_DATA_PLACEHOLDER.format(category), index_str), data, dtype=data.dtype)


class WriteSubjectCallback(Callback):

    def __init__(self, writer: wr.Writer) -> None:
        """Callback that writes the subject information to the dataset.

        Args:
            writer (.creation.writer.Writer): The writer used to write the data.
        """
        self.writer = writer

    def on_start(self, params: dict):
        """see :meth:`.Callback.on_start`."""
        subject_count = len(params[defs.KEY_SUBJECT_FILES])
        self.writer.reserve(defs.LOC_SUBJECT, (subject_count,), str)

    def on_subject(self, params: dict):
        """see :meth:`.Callback.on_subject`."""
        subject_files = params[defs.KEY_SUBJECT_FILES]
        subject_index = params[defs.KEY_SUBJECT_INDEX]

        subject = subject_files[subject_index].subject
        self.writer.fill(defs.LOC_SUBJECT, subject, expr.IndexExpression(subject_index))


class WriteImageInformationCallback(Callback):

    def __init__(self, writer: wr.Writer, category=defs.KEY_IMAGES) -> None:
        """Callback that writes the image information (shape, origin, direction, spacing) to the dataset.

        Args:
            writer (.creation.writer.Writer): The writer used to write the data.
            category (str): The category from which to extract the information from.
        """
        self.writer = writer
        self.category = category
        self.new_subject = False

    def on_start(self, params: dict):
        """see :meth:`.Callback.on_start`."""
        subject_count = len(params[defs.KEY_SUBJECT_FILES])
        self.writer.reserve(defs.LOC_INFO_SHAPE, (subject_count, 3), dtype=np.uint16)
        self.writer.reserve(defs.LOC_INFO_ORIGIN, (subject_count, 3), dtype=np.float)
        self.writer.reserve(defs.LOC_INFO_DIRECTION, (subject_count, 9), dtype=np.float)
        self.writer.reserve(defs.LOC_INFO_SPACING, (subject_count, 3), dtype=np.float)

    def on_subject(self, params: dict):
        """see :meth:`.Callback.on_subject`."""
        subject_index = params[defs.KEY_SUBJECT_INDEX]
        properties = params[defs.KEY_PLACEHOLDER_PROPERTIES.format(self.category)]  # type: conv.ImageProperties

        self.writer.fill(defs.LOC_INFO_SHAPE, properties.size, expr.IndexExpression(subject_index))
        self.writer.fill(defs.LOC_INFO_ORIGIN, properties.origin, expr.IndexExpression(subject_index))
        self.writer.fill(defs.LOC_INFO_DIRECTION, properties.direction, expr.IndexExpression(subject_index))
        self.writer.fill(defs.LOC_INFO_SPACING, properties.spacing, expr.IndexExpression(subject_index))


class WriteNamesCallback(Callback):

    def __init__(self, writer: wr.Writer) -> None:
        """Callback that writes the names of the category entries to the dataset.

        Args:
            writer (.creation.writer.Writer): The writer used to write the data.
        """
        self.writer = writer

    def on_start(self, params: dict):
        """see :meth:`.Callback.on_start`."""
        for category in params[defs.KEY_CATEGORIES]:
            self.writer.write(defs.LOC_NAMES_PLACEHOLDER.format(category),
                              params[defs.KEY_PLACEHOLDER_NAMES.format(category)], dtype='str')


class WriteFilesCallback(Callback):

    def __init__(self, writer: wr.Writer) -> None:
        """Callback that writes the file names to the dataset.

        Args:
            writer (.creation.writer.Writer): The writer used to write the data.
        """
        self.writer = writer
        self.file_root = None

    @staticmethod
    def _get_common_path(subject_files):
        def get_subject_common(subject_file: subj.SubjectFile):
            return os.path.commonpath(list(subject_file.get_all_files().values()))
        return os.path.commonpath([get_subject_common(sf) for sf in subject_files])

    def on_start(self, params: dict):
        """see :meth:`.Callback.on_start`."""
        subject_files = params[defs.KEY_SUBJECT_FILES]
        self.file_root = self._get_common_path(subject_files)
        self.writer.write(defs.LOC_FILES_ROOT, self.file_root, dtype='str')

        for category in params[defs.KEY_CATEGORIES]:
            self.writer.reserve(defs.LOC_FILES_PLACEHOLDER.format(category),
                                (len(subject_files), len(params[defs.KEY_PLACEHOLDER_NAMES.format(category)])), dtype='str')

    def on_subject(self, params: dict):
        """see :meth:`.Callback.on_subject`."""
        subject_index = params[defs.KEY_SUBJECT_INDEX]
        subject_files = params[defs.KEY_SUBJECT_FILES]

        subject_file = subject_files[subject_index]  # type: subj.SubjectFile

        for category in params[defs.KEY_CATEGORIES]:
            for index, file_name in enumerate(subject_file.categories[category].entries.values()):
                relative_path = os.path.relpath(file_name, self.file_root)
                index_expr = expr.IndexExpression(indexing=[subject_index, index], axis=(0, 1))
                self.writer.fill(defs.LOC_FILES_PLACEHOLDER.format(category), relative_path, index_expr)


def get_default_callbacks(writer: wr.Writer) -> ComposeCallback:
    """Provides a selection of commonly used callbacks to write the most important information to the dataset.

    Args:
        writer (.creation.writer.Writer): The writer used to write the data.

    Returns:
        Callback: The composed selection of common callbacks.

    """
    return ComposeCallback([MonitoringCallback(),
                            WriteDataCallback(writer),
                            WriteFilesCallback(writer),
                            WriteNamesCallback(writer),
                            WriteImageInformationCallback(writer),
                            WriteSubjectCallback(writer)])
