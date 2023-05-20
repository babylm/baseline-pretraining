import os
import sys
from os.path import expanduser

home = expanduser("~")
ROOT_DIR = os.environ.get(
        'BABYLM_ROOT_DIR',
        home)
ROOT_DIR_FREQ = os.environ.get(
        'BABYLM_ROOT_DIR_FREQ',
        ROOT_DIR)
DATASET_ROOT_DIR = os.environ.get(
        'BABYLM_DATASET_ROOT_DIR',
        os.path.join(ROOT_DIR, 'datasets'))
DEBUG = int(os.environ.get(
        'DEBUG',
        '0')) == 1
