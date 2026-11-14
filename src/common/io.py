"""Filesystem and log helpers."""

import os


def check_dir(path, make_dir=True):
    if not os.path.exists(path): 
        os.makedirs(path)


def write_log(log, path):
    with open(path, 'a') as f:
        f.writelines(log + '\n')
