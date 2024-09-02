""" Script to see available models"""
from .utils import ROOTDIR
import pathlib

directory = ROOTDIR/'output/models/'
for path in directory.glob('*'):
    print(str(path).split('/')[-1])
    for file in path.glob('*.pkl'):
        print(file)
#



