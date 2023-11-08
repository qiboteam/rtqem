#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="rtqem",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'numpy>=1.23.5',
        'matplotlib>=3.7.2',
        'qibo>=0.2.1',
        'seaborn>=0.13.0',
        'scipy>=1.10.1',
    ],
    zip_safe=False,
)
