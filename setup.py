import os
from glob import glob
from setuptools import setup, find_packages

exec(open("chemspace/version.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="joint-chem-space",
    version=__version__,
    description="A joint chemical space for different chemical representations",
    author="Mayk Caldas",
    author_email="andrew.white@rochester.edu",
    url="https://github.com/ur-whitelab/joint-chem-space",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'ipykernel',
        'datasets',
        'transformers',
        'pytorch',
    ],
    test_suite="tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
