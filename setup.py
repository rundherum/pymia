import os
import sys
from setuptools import find_packages, setup

if sys.version_info < (3, 6):
    sys.exit("Requires Python 3.6 or higher")

directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join('README.rst'), encoding='utf-8') as f:
    readme = f.read()

REQUIRED_PACKAGES = [
    'h5py',
    'matplotlib >= 2.0.2',
    'numpy >= 1.12.1',
    'SimpleITK >= 1.0.0',
    'vtk >= 7.1.0',
    'tensorboardX',
    'tensorflow',
    'torch',
    'pyyaml',
    'scipy'
]

TEST_PACKAGES = [

]

setup(
    name='pymia',
    version='0.2.0',
    description='pymia facilitates medical image analysis',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Fabian Balsiger and Alain Jungo',
    author_email='fabian.balsiger@artorg.unibe.ch',
    url='https://github.com/rundherum/pymia',
    license='Apache License',
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
