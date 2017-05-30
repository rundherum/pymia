from setuptools import find_packages, setup


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

REQUIRED_PACKAGES = [
    'numpy >= 1.12.1',
    'SimpleITK >= 1.0.0'
]

TEST_PACKAGES = [

]

setup(
    name='PythonCommon',
    version='0.0.0',
    description='PythonCommon helps MIA',
    long_description=readme,
    author='Medical Image Analysis Group',
    author_email='fabian.balsiger@istb.unibe.ch',
    url='https://github.com/istb-mia/PythonCommon',
    license=license,
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
    keywords='medical image analysis'
)
