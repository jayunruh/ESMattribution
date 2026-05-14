#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os

from setuptools import setup, find_packages

def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding="utf-8").read()

setup(
    name='ESMattribution',
    version='0.0.1',
    author='Jay Unruh',
    description='Notebooks and scripts to calculate and train models on ESM2 representations.',
    url='https://github.com/jayunruh/ESMattribution',
    license='GNU GPLv3',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=["matplotlib","numpy","pandas","scipy","sklearn","torch","torchvision"],
    py_modules=['model_utils']
)
