import os
import shutil

import SimpleITK as sitk
import numpy as np

import pymia.data.subjectfile as subj
import pymia.data.creation as crt
import pymia.data.conversion as conv


def main():
    source_file = '../examples/example-data/Subject_1/Subject_1_T1.mha'
    root_dir = 'benchmark_data/'

    entries_per_subject = 2
    nb_subjects_list = [5, 10, 25, 50, 100, 200]
    methods = ['dataset', 'file-numpy', 'file-standard', 'file-uncompressed']

    for nb_subjects in nb_subjects_list:
        out_dir = os.path.join(root_dir, f'subjects-{nb_subjects}_each-{entries_per_subject}')
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        for method_idx, method in enumerate(methods):
            print(f'[{method_idx + 1}/{len(methods)}] {method}')
            img = sitk.ReadImage(source_file)
            out_file = save_by_method(method, out_dir, source_file, img)

            subject_files = []
            for subject_idx in range(nb_subjects):
                # always the same file, no need to have different ones
                entries = {f'T1_{i}': out_file for i in range(entries_per_subject)}
                subject_file = subj.SubjectFile(f'{subject_idx:03}', images=entries)
                subject_files.append(subject_file)

            out_dataset = os.path.join(out_dir, f'benchmark_{method}.h5')
            with crt.get_writer(out_dataset) as writer:
                callbacks = [crt.WriteFilesCallback(writer), crt.MonitoringCallback(), crt.WriteNamesCallback(writer),
                             crt.WriteEssentialCallback(writer), crt.WriteImageInformationCallback(writer)]
                if method == 'dataset':
                    callbacks.append(crt.WriteDataCallback(writer))
                callbacks = crt.ComposeCallback(callbacks)

                traverser = crt.SubjectFileTraverser()
                traverser.traverse(subject_files, load=CacheLoad(), callback=callbacks)


def save_by_method(method: str, out_dir, in_file_path, img):

    path, ext = os.path.splitext(in_file_path)
    out_file = os.path.join(out_dir, f'{os.path.basename(path)}_{method}{ext}')

    if method == 'file-numpy':
        out_file = out_file.replace('.mha', '.npy')
        np.save(out_file, sitk.GetArrayFromImage(img))
    elif method == 'file-standard' or method == 'dataset':
        sitk.WriteImage(img, out_file, True)
    elif method == 'file-uncompressed':
        sitk.WriteImage(img, out_file, False)
    else:
        raise ValueError(f'unknown method "{method}"')

    return out_file


class CacheLoad(crt.Load):

    def __init__(self) -> None:
        super().__init__()
        self.cached_filename = None
        self.cached_entry = None

    def __call__(self, file_name: str, id_: str, category: str, subject_id: str):
        if (self.cached_entry is not None) and (self.cached_entry == file_name):
            return self.cached_entry

        if file_name.endswith('npy'):
            np_img = np.load(file_name)
            self.cached_entry = np_img, conv.ImageProperties(sitk.GetImageFromArray(np_img))
        else:
            img = sitk.ReadImage(file_name)
            self.cached_entry = sitk.GetArrayFromImage(img), conv.ImageProperties(img)
        return self.cached_entry


if __name__ == '__main__':
    main()
