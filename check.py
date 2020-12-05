from pathlib import Path
import os
import numpy as np
import torch
import pickle
import scipy.io as sio

# lines = Path('./train.configs/train_aug_120x120.list.train').read_text().strip().split('\n')

# print(len(lines))

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

params = _load('./train.configs/param_all_norm.pkl')
print(params[3])  # (636252, 62)