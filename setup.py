#!/usr/bin/env python
from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as req_fp:
    install_requires = req_fp.readlines()

setup(
    name='mowgli',
    version='1.0',
    description='Minimalist NMT for educational purposes',
    author='David Stap',
    url='https://github.com/davidstap/mowgli',
    license='Apache License',
    install_requires=install_requires,
    packages=find_packages(exclude=[]),
    python_requires='>=3.5',
    project_urls={
        'Documentation': 'http://mowgli.readthedocs.io/en/latest/',
        'Source': 'https://github.com/mowgli/mowgli',
        'Tracker': 'https://github.com/mowgli/mowgli/issues',
    },
    entry_points={
        'console_scripts': [
        ],
    }
)
