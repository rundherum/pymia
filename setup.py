import os
import sys
from setuptools import find_packages, setup

if sys.version_info < (3, 6):
    sys.exit("Requires Python 3.6 or higher")

directory = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(directory, 'pymia', '__version__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), about)

with open(os.path.join('README.rst'), 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRED_PACKAGES = [
    'h5py >= 2.10.0',
    'numpy >= 1.12.1',
    'scipy >= 1.3.2',
    'SimpleITK >= 1.2.3'
]

TEST_PACKAGES = [
    'pytest >= 5',
    'pytest-cov >= 2.10.0'
]

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/x-rst',
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
    license=about['__license__'],
    python_requires='>=3.6',
    packages=find_packages(exclude=['docs', 'examples', 'test']),
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries'
    ],
    keywords=[
        'medical image analysis',
        'deep learning',
        'data handling',
        'evaluation',
        'metrics'
    ]
)
