"""This module contains global definitions for the :mod:`pymia.data` package.
"""

NAMES_PLACEHOLDER = 'meta/names/{}_names'
# NAMES_IMAGE = 'meta/names/image_names'
# NAMES_LABEL = 'meta/names/label_names'
# NAMES_SUPPL = 'meta/names/supplementary_names'

INFO_SHAPE = 'meta/info/shapes'
INFO_ORIGIN = 'meta/info/origins'
INFO_DIRECTION = 'meta/info/directions'
INFO_SPACING = 'meta/info/spacing'

FILES_PLACEHOLDER = 'meta/files/{}_files'
# FILES_IMAGE = 'meta/files/image_files'
# FILES_LABEL = 'meta/files/label_files'
# FILES_SUPPL = 'meta/files/supplementary_files'
FILES_ROOT = 'meta/files/file_root'

SUBJECT = 'meta/subjects'

# DATA = 'data'
DATA_PLACEHOLDER = 'data/{}'
# DATA_IMAGE = '{}/images'.format(DATA)
# DATA_LABEL = '{}/labels'.format(DATA)

# keys for a batch dictionary
KEY_FILE_ROOT = 'file_root'
KEY_IMAGES = 'images'
KEY_LABELS = 'labels'
KEY_PROPERTIES = 'properties'
KEY_SHAPE = 'shape'
KEY_SUBJECT = 'subject'
KEY_SUBJECT_INDEX = 'subject_index'
