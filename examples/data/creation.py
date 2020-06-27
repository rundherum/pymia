import argparse
import enum
import glob
import os
import typing

import SimpleITK as sitk
import numpy as np

import pymia.data as data
import pymia.data.conversion as conv
import pymia.data.definition as defs
import pymia.data.creation as crt
import pymia.data.transformation as tfm
import pymia.data.creation.fileloader as file_load


class FileTypes(enum.Enum):
    T1 = 1  # The T1-weighted image
    T2 = 2  # The T2-weighted image
    GT = 3  # The label (ground truth) image
    MASK = 4  # The foreground mask
    AGE = 5  # The age
    GPA = 6  # The GPA
    GENDER = 7  # The gender


class LoadData(file_load.Load):

    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) -> \
            typing.Tuple[np.ndarray, typing.Union[conv.ImageProperties, None]]:
        if id_ == FileTypes.AGE.name:
            with open(file_name, 'r') as f:
                value = np.asarray([int(f.readline().split(':')[1].strip())])
                return value, None
        if id_ == FileTypes.GPA.name:
            with open(file_name, 'r') as f:
                value = np.asarray([float(f.readlines()[1].split(':')[1].strip())])
                return value, None
        if id_ == FileTypes.GENDER.name:
            with open(file_name, 'r') as f:
                value = np.array(f.readlines()[2].split(':')[1].strip())
                return value, None

        if category == defs.KEY_IMAGES:
            img = sitk.ReadImage(file_name, sitk.sitkFloat32)
        else:
            # this is the ground truth (defs.KEY_LABELS), which will be loaded as unsigned integer
            img = sitk.ReadImage(file_name, sitk.sitkUInt8)

        return sitk.GetArrayFromImage(img), conv.ImageProperties(img)


class Subject(data.SubjectFile):

    def __init__(self, subject: str, files: dict):
        super().__init__(subject,
                         images={FileTypes.T1.name: files[FileTypes.T1], FileTypes.T2.name: files[FileTypes.T2]},
                         labels={FileTypes.GT.name: files[FileTypes.GT]},
                         mask={FileTypes.MASK.name: files[FileTypes.MASK]},
                         numerical={FileTypes.AGE.name: files[FileTypes.AGE], FileTypes.GPA.name: files[FileTypes.GPA]},
                         gender={FileTypes.GENDER.name: files[FileTypes.GENDER]})
        self.subject_path = files.get(subject, '')


def main(hdf_file: str, data_dir: str):

    # collect the files for each subject
    subjects = get_subject_files(data_dir)

    # remove the "old" dataset if it exists
    if os.path.exists(hdf_file):
        os.remove(hdf_file)

    with crt.get_writer(hdf_file) as writer:
        # initialize the callbacks that will actually write the data to the dataset file
        callbacks = crt.get_default_callbacks(writer)

        # add a transform to normalize the images
        transform = tfm.IntensityNormalization(loop_axis=3, entries=(defs.KEY_IMAGES, ))

        # run through the subject files (loads them, applies transformations, and calls the callback for writing them)
        traverser = crt.SubjectFileTraverser()
        traverser.traverse(subjects, callback=callbacks, load=LoadData(), transform=transform)


def get_subject_files(data_dir: str) -> typing.List[Subject]:
    """Collects the files for all the subjects in the data directory.

    Args:
        data_dir (str): The data directory.

    Returns:
        typing.List[data.SubjectFile]: The list of the collected subject files.

    """
    subject_dirs = [subject_dir for subject_dir in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(subject_dir) and
                    os.path.basename(subject_dir).startswith('Subject')]
    sorted(subject_dirs)

    keys = [FileTypes.T1, FileTypes.T2, FileTypes.GT, FileTypes.MASK, FileTypes.AGE, FileTypes.GPA, FileTypes.GENDER]
    subjects = []
    # for each subject on file system, initialize a Subject object
    for subject_dir in subject_dirs:
        id_ = os.path.basename(subject_dir)

        file_dict = {id_: subject_dir}  # init dict with id_ pointing to the path of the subject
        for key in keys:
            file_path = get_full_file_path(id_, subject_dir, key)
            file_dict[key] = file_path

        subjects.append(Subject(id_, file_dict))
    return subjects


def get_full_file_path(id_: str, root_dir: str, file_key) -> str:
    """Gets the full file path for an image.
    Args:
        id_ (str): The image identification.
        root_dir (str): The image' root directory.
        file_key (object): A human readable identifier used to identify the image.
    Returns:
        str: The images' full file path.
    """

    if file_key == FileTypes.T1:
        file_name = f'{id_}_T1.mha'
    elif file_key == FileTypes.T2:
        file_name = f'{id_}_T2.mha'
    elif file_key == FileTypes.GT:
        file_name = f'{id_}_GT.mha'
    elif file_key == FileTypes.MASK:
        file_name = f'{id_}_MASK.nii.gz'
    elif file_key == FileTypes.AGE or file_key == FileTypes.GPA or file_key == FileTypes.GENDER:
        file_name = f'{id_}_demographic.txt'
    else:
        raise ValueError('Unknown key')

    return os.path.join(root_dir, file_name)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Creation')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../example-data/example-dataset.h5',
        help='Path to the dataset file.'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../example-data',
        help='Path to the data directory.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.data_dir)
