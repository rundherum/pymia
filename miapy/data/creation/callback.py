import os

import numpy as np

import miapy.data.indexexpression as expr
import miapy.data.conversion as conv
import miapy.data.subjectfile as subj
import miapy.data.definition as df
from . import writer as wr


class Callback:

    def on_start(self, params: dict):
        pass

    def on_end(self, params: dict):
        pass

    def on_subject(self, params: dict):
        pass


class ComposeCallback(Callback):

    def __init__(self, callbacks) -> None:
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


class WriteDataCallback(Callback):

    def __init__(self, writer: wr.Writer) -> None:
        self.writer = writer

    def on_subject(self, params: dict):
        subject_files = params['subject_files']
        subject_index = params['subject_index']

        max_digits = len(str(len(subject_files)))
        index_str = '{{:0{}}}'.format(max_digits).format(subject_index)

        images = params['images']
        self.writer.write('{}/{}'.format(df.DATA_IMAGE, index_str), images)
        if params['has_labels']:
            labels = params['labels']
            self.writer.write('{}/{}'.format(df.DATA_LABEL, index_str), labels)
        if params['has_supplementaries']:
            supplementaries = params['supplementaries']
            for key, value in supplementaries.items():
                self.writer.write('{}/{}/{}'.format(df.DATA, key, index_str), value, value.dtype)


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

    def __init__(self, writer: wr.Writer) -> None:
        self.writer = writer
        self.new_subject = False

    def on_start(self, params: dict):
        subject_count = len(params['subject_files'])
        self.writer.reserve(df.INFO_SHAPE, (subject_count, 3), dtype=np.uint16)
        self.writer.reserve(df.INFO_ORIGIN, (subject_count, 3), dtype=np.float)
        self.writer.reserve(df.INFO_DIRECTION, (subject_count, 9), dtype=np.float)
        self.writer.reserve(df.INFO_SPACING, (subject_count, 3), dtype=np.float)

    def on_subject(self, params: dict):
        subject_index = params['subject_index']
        image_properties = params['image_properties']  # type: conv.ImageProperties

        self.writer.fill(df.INFO_SHAPE, image_properties.size, expr.IndexExpression(subject_index))
        self.writer.fill(df.INFO_ORIGIN, image_properties.origin, expr.IndexExpression(subject_index))
        self.writer.fill(df.INFO_DIRECTION, image_properties.direction, expr.IndexExpression(subject_index))
        self.writer.fill(df.INFO_SPACING, image_properties.spacing, expr.IndexExpression(subject_index))


class WriteNamesCallback(Callback):

    def __init__(self, writer: wr.Writer) -> None:
        self.writer = writer

    def on_start(self, params: dict):
        image_names = params['image_names']
        self.writer.write(df.NAMES_IMAGE, image_names, dtype='str')

        if params['has_labels']:
            label_names = params['label_names']
            self.writer.write(df.NAMES_LABEL, label_names, dtype='str')

        if params['has_supplementaries']:
            supplementary_names = params['supplementary_names']
            self.writer.write(df.NAMES_SUPPL, supplementary_names, dtype='str')


class WriteFilesCallback(Callback):

    def __init__(self, writer: wr.Writer) -> None:
        self.writer = writer
        self.file_root = None

    @staticmethod
    def _get_common_path(subject_files):
        def get_subject_common(subject_file: subj.SubjectFile):
            return os.path.commonpath(list(subject_file.get_all_files().values()))
        return os.path.commonpath(get_subject_common(sf) for sf in subject_files)

    def on_start(self, params: dict):
        subject_files = params['subject_files']
        self.file_root = self._get_common_path(subject_files)

        self.writer.write(df.FILES_ROOT, self.file_root, dtype='str')
        image_names = params['image_names']
        self.writer.reserve(df.FILES_IMAGE, (len(subject_files), len(image_names)), dtype='str')

        if params['has_labels']:
            label_names = params['label_names']
            self.writer.reserve(df.FILES_LABEL, (len(subject_files), len(label_names)), dtype='str')

        if params['has_supplementaries']:
            supplementary_names = params['supplementary_names']
            self.writer.reserve(df.FILES_SUPPL, (len(subject_files), len(supplementary_names)), dtype='str')

    def on_subject(self, params: dict):
        subject_index = params['subject_index']
        subject_files = params['subject_files']

        subject_file = subject_files[subject_index]  # type: subj.SubjectFile

        for image_index, key in enumerate(subject_file.images):
            relative_path = os.path.relpath(subject_file.images[key], self.file_root)
            index_expr = expr.IndexExpression(indexing=[subject_index, image_index], axis=(0, 1))
            self.writer.fill(df.FILES_IMAGE, relative_path, index_expr)

        if subject_file.label_images is not None:
            for label_index, key in enumerate(subject_file.label_images):
                relative_path = os.path.relpath(subject_file.label_images[key], self.file_root)
                index_expr = expr.IndexExpression(indexing=[subject_index, label_index], axis=(0, 1))
                self.writer.fill(df.FILES_LABEL, relative_path, index_expr)

        if subject_file.supplementaries is not None:
            for supplementary_index, key in enumerate(subject_file.supplementaries):
                relative_path = os.path.relpath(subject_file.supplementaries[key], self.file_root)
                index_expr = expr.IndexExpression(indexing=[subject_index, supplementary_index], axis=(0, 1))
                self.writer.fill(df.FILES_SUPPL, relative_path, index_expr)


def get_default_callbacks(writer: wr.Writer):
    return ComposeCallback([WriteDataCallback(writer),
                            WriteFilesCallback(writer),
                            WriteNamesCallback(writer),
                            WriteImageInformationCallback(writer),
                            WriteSubjectCallback(writer)])
