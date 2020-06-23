"""This module contains global definitions for the :mod:`pymia.data` package."""


def subject_index_to_str(subject_index, nb_subjects):
    max_digits = len(str(nb_subjects))
    index_str = '{{:0{}}}'.format(max_digits).format(subject_index)
    return index_str


# location strings for the database
LOC_NAMES_PLACEHOLDER = 'meta/names/{}_names'
LOC_IMGPROP_SHAPE = 'meta/image_props/shapes'
LOC_IMGPROP_ORIGIN = 'meta/image_props/origins'
LOC_IMGPROP_DIRECTION = 'meta/image_props/directions'
LOC_IMGPROP_SPACING = 'meta/image_props/spacing'
LOC_FILES_PLACEHOLDER = 'meta/files/{}_files'
LOC_FILES_ROOT = 'meta/files/file_root'
LOC_SUBJECT = 'meta/subjects'
LOC_SHAPE_PLACEHOLDER = 'meta/shapes/{}_shapes'
LOC_DATA_PLACEHOLDER = 'data/{}'

# keys for a (batch) dictionary
# Do not remove the '#:', they are relevant for the documentation
KEY_SUBJECT_FILES = 'subject_files'  #:
KEY_CATEGORIES = 'categories'  #:
KEY_PLACEHOLDER_NAMES = '{}_names'  #:
KEY_PLACEHOLDER_PROPERTIES = '{}_properties'  #:
KEY_PLACEHOLDER_FILES = '{}_files'  #:
KEY_FILE_ROOT = 'file_root'  #:
KEY_IMAGES = 'images'  #:
KEY_LABELS = 'labels'  #:
KEY_PROPERTIES = 'properties'  #:
KEY_SHAPE = 'shape'  #:
KEY_SUBJECT = 'subject'  #:
KEY_SUBJECT_INDEX = 'subject_index'  #:
KEY_INDEX_EXPR = 'index_expr'  #:
KEY_SAMPLE_INDEX = 'sample_index'  #:
