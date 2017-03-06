"""setup.py."""

import os

from setuptools import setup

import versioneer


filename = os.path.join(os.path.dirname(__file__), 'requirements.txt')
requirements = open(filename).read().splitlines()

setup(
    name='dasem',
    author='Finn Aarup Nielsen',
    author_email='faan@dtu.dk',
    cmdclass=versioneer.get_cmdclass(),
    description='Danish semantic analysis',
    license='Apache',
    keywords='text',
    url='https://github.com/fnielsen/dasem',
    packages=['dasem'],
    package_data={
        'dasem': [
            'data/four_words.csv',
            'data/compounds.txt',
            'data/README.rst',
        ]
    },
    install_requires=requirements,
    long_description='',
    classifiers=[
        'Programming Language :: Python :: 2.7',
    ],
    tests_require=['pytest', 'flake8'],
)
