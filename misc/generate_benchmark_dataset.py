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
    file_methods = ['file-numpy', 'file-standard', 'file-uncompressed']
    dataset_methods = {'dataset': 'file-uncompressed'}  # value is where to get the data from to build the dataset

    if os.path.isdir(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)

    # create all the required data
    img = sitk.ReadImage(source_file)
    methods_to_subject_files = create_files(root_dir, file_methods, img, max(nb_subjects_list), entries_per_subject)

    # also add the files required for the dataset methods
    for dataset_method, corr_file_method in dataset_methods.items():
        methods_to_subject_files[dataset_method] = methods_to_subject_files[corr_file_method]

    # create the .h5 datasets for all the methods
    for nb_subjects in nb_subjects_list:
        out_dir = os.path.join(root_dir, f'datasets_subjects-{nb_subjects}_each-{entries_per_subject}')
        os.makedirs(out_dir)

        for method_idx, method in enumerate(methods_to_subject_files):
            print(f'[{method_idx + 1}/{len(methods_to_subject_files)}] Load and build dataset for: {method}')

            out_dataset = os.path.join(out_dir, f'benchmark_{method}.h5')
            with crt.get_writer(out_dataset) as writer:
                callbacks = [crt.WriteFilesCallback(writer), crt.MonitoringCallback(), crt.WriteNamesCallback(writer),
                             crt.WriteEssentialCallback(writer), crt.WriteImageInformationCallback(writer)]
                if method in dataset_methods:
                    callbacks.append(crt.WriteDataCallback(writer))
                callbacks = crt.ComposeCallback(callbacks)

                subject_files = methods_to_subject_files[method][:nb_subjects]  # only select the nb_subject firsts
                traverser = crt.SubjectFileTraverser()
                traverser.traverse(subject_files, load=CacheLoad(), callback=callbacks)


def create_files(root_dir, file_methods, source_img, max_subjects, entries_per_subject):
    # create the data (build max)
    print(f'create the {max_subjects} subjects per method')
    out_dir = os.path.join(root_dir, f'rawdata_subjects-{max_subjects}_each-{entries_per_subject}')

    method_to_subject_files = {}
    for method_idx, method in enumerate(file_methods):
        print(f'[{method_idx + 1}/{len(file_methods)}] Create data for: {method}')

        method_dir = os.path.join(out_dir, method)
        os.makedirs(method_dir)

        subject_files = []
        existing_entries = {}
        for subject_idx in range(max_subjects):
            print(f'[{subject_idx + 1}/{max_subjects}]')
            entries = {}
            subject = f'{subject_idx:03}'
            for entry_idx in range(entries_per_subject):
                entry = f'T1-{entry_idx}'
                if entry in existing_entries:
                    out_file = copy(existing_entries[entry], method_dir, f'{subject}_{entry}', method)
                else:
                    out_file = save_by_method(method, method_dir, f'{subject}_{entry}', source_img)
                    existing_entries[entry] = out_file
                entries[entry] = out_file
            subject_file = subj.SubjectFile(f'{subject_idx:03}', images=entries)
            subject_files.append(subject_file)
        method_to_subject_files[method] = subject_files
    return method_to_subject_files


def copy(existing_file, out_dir, subject_id, method):
    _, ext = os.path.splitext(existing_file)
    out_file = os.path.join(out_dir, f'{subject_id}_{method}{ext}')
    shutil.copy(existing_file, out_file)
    return out_file


def save_by_method(method: str, out_dir, subject_id, img):
    out_file = os.path.join(out_dir, f'{subject_id}_{method}.mha')

    if method == 'file-numpy':
        out_file = out_file.replace('.mha', '.npy')
        np.save(out_file, sitk.GetArrayFromImage(img))
    elif method == 'file-standard':
        sitk.WriteImage(img, out_file, True)
    elif method == 'file-uncompressed' or method == 'dataset':
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
