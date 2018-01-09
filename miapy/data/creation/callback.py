import os

import miapy.data.utils as util
from . import writer as wr


class Callback:

    def on_start(self, params: dict):
        pass

    def on_end(self, params: dict):
        pass

    def on_subject_start(self, params: dict):
        pass

    def on_subject_end(self, params: dict):
        pass

    def on_gt_file(self, params: dict):
        pass

    def on_image_file(self, params: dict):
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

    def on_subject_start(self, params: dict):
        for c in self.callbacks:
            c.on_subject_start(params)

    def on_subject_end(self, params: dict):
        for c in self.callbacks:
            c.on_subject_end(params)

    def on_gt_file(self, params: dict):
        for c in self.callbacks:
            c.on_gt_file(params)

    def on_image_file(self, params: dict):
        for c in self.callbacks:
            c.on_image_file(params)


# todo: name passed to writer is path (for hdf5 dataset entries). This is not really generic -> other solution?
class WriteDataCallback(Callback):
    SEQ_PATH = 'data/sequences'
    GT_PATH = 'data/gts'

    def __init__(self, writer: wr.Writer) -> None:
        self.writer = writer

    def on_subject_end(self, params: dict):
        subject_files = params['subject_files']
        subject_index = params['subject_index']

        max_digits = len(str(len(subject_files)))
        index_str = '{{:0{}}}'.format(max_digits).format(subject_index)

        np_images = params['images']
        self.writer.write('{}/{}'.format(self.SEQ_PATH, index_str), np_images)
        if params['has_gt']:
            np_gts = params['labels']
            self.writer.write('{}/{}'.format(self.GT_PATH, index_str), np_gts)


class WriteSubjectCallback(Callback):
    SUBJECT = 'meta/subjects'

    def __init__(self, writer: wr.Writer) -> None:
        self.writer = writer

    def on_start(self, params: dict):
        subject_count = len(params['subject_files'])
        self.writer.reserve(self.SUBJECT, (subject_count,), str)

    def on_subject_start(self, params: dict):
        subject = params['subject']
        subject_index = params['subject_index']
        self.writer.fill(self.SUBJECT, subject, util.IndexExpression(subject_index))


class WriteNamesCallback(Callback):
    SEQ_NAMES = 'meta/sequence_names'
    GT_NAMES = 'meta/gt_names'

    def __init__(self, writer: wr.Writer) -> None:
        self.writer = writer

    def on_start(self, params: dict):
        sequence_names = params['sequence_names']
        self.writer.write(self.SEQ_NAMES, sequence_names, dtype='str')

        if params['has_gt']:
            gt_names = params['gt_names']
            self.writer.write(self.GT_NAMES, gt_names, dtype='str')


class WriteFilesCallback(Callback):
    SEQ_FILES = 'meta/sequence_files'
    GT_FILES = 'meta/gt_files'
    FILE_ROOT = 'meta/file_root'

    def __init__(self, writer: wr.Writer) -> None:
        self.writer = writer
        self.file_root = None

    @staticmethod
    def _get_common_path(subject_files):
        def get_subject_common(subject_file):
            all_files = list(subject_file.get_sequences().values())
            gts = subject_file.get_gts()
            if gts is not None:
                all_files.extend(list(gts.values()))
            return os.path.commonpath(all_files)
        return os.path.commonpath(get_subject_common(sf) for sf in subject_files)

    def on_start(self, params: dict):
        subject_files = params['subject_files']
        self.file_root = self._get_common_path(subject_files)

        self.writer.write(self.FILE_ROOT, self.file_root, dtype='str')
        sequence_names = params['sequence_names']
        self.writer.reserve(self.SEQ_FILES, (len(subject_files), len(sequence_names)), dtype='str')

        if params['has_gt']:
            gt_names = params['gt_names']
            self.writer.reserve(self.GT_FILES, (len(subject_files), len(gt_names)), dtype='str')

    def on_image_file(self, params: dict):
        sequence_file = params['file']
        subject_index = params['subject_index']
        sequence_index = params['sequence_index']

        relative_path = os.path.relpath(sequence_file, self.file_root)
        index_expr = util.IndexExpression(indexing=[subject_index, sequence_index], axis=(0, 1))
        self.writer.fill(self.SEQ_FILES, relative_path, index_expr)

    def on_gt_file(self, params: dict):
        gt_file = params['file']
        subject_index = params['subject_index']
        gt_index = params['gt_index']

        relative_path = os.path.relpath(gt_file, self.file_root)
        index_expr = util.IndexExpression(indexing=[subject_index, gt_index], axis=(0, 1))
        self.writer.fill(self.GT_FILES, relative_path, index_expr)
