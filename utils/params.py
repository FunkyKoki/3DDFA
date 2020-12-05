#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
import numpy as np
from .io import _load


def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)


d = make_abs_path('../train.configs')
keypoints = _load(osp.join(d, 'keypoints_sim.npy'))  # shape (204,)
# print(keypoints.shape)
w_shp = _load(osp.join(d, 'w_shp_sim.npy'))  # shape (159645, 40)
w_exp = _load(osp.join(d, 'w_exp_sim.npy'))  # shape (159645, 10)  # simplified version
# print(w_shp.shape)
# print(w_exp.shape)
meta = _load(osp.join(d, 'param_whitening.pkl'))
# param_mean and param_std are used for re-whitening
param_mean = meta.get('param_mean')  # shape (62,)
param_std = meta.get('param_std')    # shape (62,)
# print(param_mean)
# print(param_std.shape)
u_shp = _load(osp.join(d, 'u_shp.npy'))  # shape (159645, 1)
u_exp = _load(osp.join(d, 'u_exp.npy'))  # shape (159645, 1)
# print(u_shp.shape)
# print(u_exp.shape)
u = u_shp + u_exp  # (159645, 1)
# print(u.shape)
w = np.concatenate((w_shp, w_exp), axis=1)  # (159645, 50)
# print(w.shape)
w_base = w[keypoints]  # (204, 50)
# print(w_base.shape)
w_norm = np.linalg.norm(w, axis=0)  # (50,)
# print(w_norm.shape)
w_base_norm = np.linalg.norm(w_base, axis=0)  # (50,)
# print(w_base_norm.shape)

# for inference
dim = w_shp.shape[0] // 3
u_base = u[keypoints].reshape(-1, 1)
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]
std_size = 120

# for paf (pac)
paf = _load(osp.join(d, 'Model_PAF.pkl'))
u_filter = paf.get('mu_filter')  # (12288, 1)
# print(u_filter.shape)
w_filter = paf.get('w_filter')  # (12288, 40)
# print(w_filter.shape)
w_exp_filter = paf.get('w_exp_filter')  # (12288, 10)
# print(w_exp_filter.shape)

# pncc code (mean shape)
pncc_code = _load(osp.join(d, 'pncc_code.npy')) # shape (3, 53215)
# print(pncc_code.shape)
