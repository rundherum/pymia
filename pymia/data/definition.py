"""This module contains global definitions for the :mod:`pymia.data` package."""

# location strings for the database
LOC_NAMES_PLACEHOLDER = 'meta/names/{}_names'
LOC_INFO_SHAPE = 'meta/info/shapes'
LOC_INFO_ORIGIN = 'meta/info/origins'
LOC_INFO_DIRECTION = 'meta/info/directions'
LOC_INFO_SPACING = 'meta/info/spacing'
LOC_FILES_PLACEHOLDER = 'meta/files/{}_files'
LOC_FILES_ROOT = 'meta/files/file_root'
LOC_SUBJECT = 'meta/subjects'
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
