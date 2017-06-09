import sys
from setuptools import find_packages, setup


if not sys.hexversion < 0x03060000:
    sys.exit("Requires Python 3.6 or higher")

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license_ = f.read()

REQUIRED_PACKAGES = [
    'matplotlib >= 2.0.2',
    'numpy >= 1.12.1',
    'SimpleITK >= 1.0.0',
    'sphinx >= 1.6',
    'sphinx_rtd_theme >= 0.2.4'
]

TEST_PACKAGES = [

]

setup(
    name='miapy',
    version='0.1.0',
    description='miapy facilitates medical image analysis',
    long_description=readme,
    author='Medical Image Analysis Group',
    author_email='fabian.balsiger@istb.unibe.ch',
    url='https://github.com/istb-mia/miapy',
    license=license_,
    packages=find_packages(exclude=['test', 'docs']),
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
