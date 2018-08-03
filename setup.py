import sys
from setuptools import find_packages, setup

if sys.version_info < (3, 6):
    sys.exit("Requires Python 3.6 or higher")

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license_ = f.read()

REQUIRED_PACKAGES = [
    'matplotlib >= 2.0.2',
    'numpy >= 1.12.1',
    'SimpleITK >= 1.0.0',
    'vtk >= 7.1.0',
    'h5py',
    'torch'
]

TEST_PACKAGES = [

]

setup(
    name='pymia',
    version='0.1.0',
    description='pymia facilitates medical image analysis',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Medical Image Analysis Group, University of Bern',
    author_email='fabian.balsiger@istb.unibe.ch',
    url='https://github.com/rundherum/pymia',
    license=license_,
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
        'ITK'
        'SimpleITK'
    ]
)
