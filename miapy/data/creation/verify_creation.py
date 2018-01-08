import glob
import os

import data.creation.subjectfile as subj
import data.creation.writer as wr
import data.creation.traverser as tv
import data.creation.callback as cb
import libs.util.filehelper as fh


def main():
    out_file = '../../in/datasets/ds2.h5'

    subject_files = get_subject_files()
    # sort the subject files according to the subject name (identifier)
    subject_files.sort(key=lambda sf: sf.subject)

    fh.create_dir_if_not_exists(out_file, is_file=True)
    fh.remove_if_exists(out_file)

    with wr.Hdf5Writer(out_file) as writer:
        callbacks = [cb.WriteNamesCallback(writer), cb.WriteFilesCallback(writer), cb.WriteDataCallback(writer)]

        traverser = tv.SubjectFileTraverser()
        traverser.traverse(subject_files, callbacks=callbacks)


def get_subject_files():
    # in_dir = '/home/a/Desktop/cavity_atlas_space'
    in_dir = '/media/data/datasets/cavity/atlas_space'

    childs = glob.glob(in_dir + '/*')
    subject_files = []

    # collect the files
    for child in childs:
        subject = os.path.basename(child)
        sequence_files = {'flair': os.path.join(child, 'Flair-registered.nii.gz'),
                          't1': os.path.join(child, 'T1-registered.nii.gz'),
                          't2': os.path.join(child, 'T2-registered.nii.gz'),
                          't1c': os.path.join(child, 'T1c-registered.nii.gz')}
        gt_files = {'rater-EE': os.path.join(child, 'cavity_label_EE-registered.nii.gz'),
                    'rater-EH': os.path.join(child, 'cavity_label_EH-registered.nii.gz'),
                    'rater-MB': os.path.join(child, 'cavity_label_MB-registered.nii.gz'),
                    'gt': os.path.join(child, 'union_EEEHMB-registered.nii.gz')}
        assert all(os.path.isfile(file_) for file_ in sequence_files.values())
        assert all(os.path.isfile(file_) for file_ in gt_files.values())

        sf = subj.BaseSubjectFile(subject, sequence_files, gt_files)
        subject_files.append(sf)

    return subject_files


if __name__ == '__main__':
    main()
