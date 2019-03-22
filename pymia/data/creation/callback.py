import os
import typing

import numpy as np

import pymia.data.indexexpression as expr
import pymia.data.conversion as conv
import pymia.data.subjectfile as subj
import pymia.data.definition as df
from . import writer as wr


class Callback:

    def on_start(self, params: dict):
        pass

    def on_end(self, params: dict):
        pass

    def on_subject(self, params: dict):
        pass


class ComposeCallback(Callback):

    def __init__(self, callbacks: typing.List[Callback]) -> None:
        self.callbacks = callbacks

    def on_start(self, params: dict):
        for c in self.callbacks:
            c.on_start(params)

    def on_end(self, params: dict):
        for c in self.callbacks:
            c.on_end(params)

    def on_subject(self, params: dict):
        for c in self.callbacks:
            c.on_subject(params)


class MonitoringCallback(Callback):

    def on_start(self, params: dict):
        print('start dataset creation')

    def on_subject(self, params: dict):
        index = params['subject_index']
        subject_files = params['subject_files']
        print('[{}/{}] {}'.format(index + 1, len(subject_files), subject_files[index].subject))

    def on_end(self, params: dict):
        print('dataset creation finished')


class WriteDataCallback(Callback):

    def __init__(self, writer: wr.Writer) -> None:
        self.writer = writer

    def on_subject(self, params: dict):
        subject_files = params['subject_files']
        subject_index = params['subject_index']

        max_digits = len(str(len(subject_files)))
        index_str = '{{:0{}}}'.format(max_digits).format(subject_index)

        for category in params['categories']:
            data = params[category]
            self.writer.write('{}/{}'.format(df.DATA_PLACEHOLDER.format(category), index_str), data, dtype=data.dtype)


class WriteSubjectCallback(Callback):

    def __init__(self, writer: wr.Writer) -> None:
        self.writer = writer

    def on_start(self, params: dict):
        subject_count = len(params['subject_files'])
        self.writer.reserve(df.SUBJECT, (subject_count,), str)

    def on_subject(self, params: dict):
        subject_files = params['subject_files']
        subject_index = params['subject_index']

        subject = subject_files[subject_index].subject
        self.writer.fill(df.SUBJECT, subject, expr.IndexExpression(subject_index))


class WriteImageInformationCallback(Callback):

    def __init__(self, writer: wr.Writer, category='images') -> None:
        self.writer = writer
        self.category = category
        self.new_subject = False

    def on_start(self, params: dict):
        subject_count = len(params['subject_files'])
        self.writer.reserve(df.INFO_SHAPE, (subject_count, 3), dtype=np.uint16)
        self.writer.reserve(df.INFO_ORIGIN, (subject_count, 3), dtype=np.float)
        self.writer.reserve(df.INFO_DIRECTION, (subject_count, 9), dtype=np.float)
        self.writer.reserve(df.INFO_SPACING, (subject_count, 3), dtype=np.float)

    def on_subject(self, params: dict):
        subject_index = params['subject_index']
        properties = params['{}_properties'.format(self.category)]  # type: conv.ImageProperties

        self.writer.fill(df.INFO_SHAPE, properties.size, expr.IndexExpression(subject_index))
        self.writer.fill(df.INFO_ORIGIN, properties.origin, expr.IndexExpression(subject_index))
        self.writer.fill(df.INFO_DIRECTION, properties.direction, expr.IndexExpression(subject_index))
        self.writer.fill(df.INFO_SPACING, properties.spacing, expr.IndexExpression(subject_index))


class WriteNamesCallback(Callback):

    def __init__(self, writer: wr.Writer) -> None:
        self.writer = writer

    def on_start(self, params: dict):
        for category in params['categories']:
            self.writer.write(df.NAMES_PLACEHOLDER.format(category), params['{}_names'.format(category)], dtype='str')


class WriteFilesCallback(Callback):

    def __init__(self, writer: wr.Writer) -> None:
        self.writer = writer
        self.file_root = None

    @staticmethod
    def _get_common_path(subject_files):
        def get_subject_common(subject_file: subj.SubjectFile):
            return os.path.commonpath(list(subject_file.get_all_files().values()))
        return os.path.commonpath([get_subject_common(sf) for sf in subject_files])

    def on_start(self, params: dict):
        subject_files = params['subject_files']
        self.file_root = self._get_common_path(subject_files)
        self.writer.write(df.FILES_ROOT, self.file_root, dtype='str')

        for category in params['categories']:
            self.writer.reserve(df.FILES_PLACEHOLDER.format(category),
                                (len(subject_files), len(params['{}_names'.format(category)])), dtype='str')

    def on_subject(self, params: dict):
        subject_index = params['subject_index']
        subject_files = params['subject_files']

        subject_file = subject_files[subject_index]  # type: subj.SubjectFile

        for category in params['categories']:
            for index, file_name in enumerate(subject_file.categories[category].entries.values()):
                relative_path = os.path.relpath(file_name, self.file_root)
                index_expr = expr.IndexExpression(indexing=[subject_index, index], axis=(0, 1))
                self.writer.fill(df.FILES_PLACEHOLDER.format(category), relative_path, index_expr)


def get_default_callbacks(writer: wr.Writer) -> ComposeCallback:
    return ComposeCallback([MonitoringCallback(),
                            WriteDataCallback(writer),
                            WriteFilesCallback(writer),
                            WriteNamesCallback(writer),
                            WriteImageInformationCallback(writer),
                            WriteSubjectCallback(writer)])
