from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='tf.rf01',
    version='0.0.0',
    description='Python common code',
    long_description=readme,
    author='MIA',
    author_email='fabian.balsiger@istb.unibe.ch',
    url='https://github.com/istb-mia/merlin/tree/tf.rf01',
    license=license,
    packages=find_packages(exclude=['test', 'docs'], install_requires=['SimpleITK'])
)
