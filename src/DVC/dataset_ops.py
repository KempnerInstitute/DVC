###############################################
# src/torch_vine/dataset_ops.py
###############################################

import torch
import numpy as np
from sklearn.model_selection import KFold

def kfold(data: np.ndarray, n_splits: int):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)
    train_ind_list = []
    test_ind_list = []
    for train_index, test_index in kf.split(data):
        train_ind_list.append(train_index)
        test_ind_list.append(test_index)
    return train_ind_list, test_ind_list


def data_split(data: torch.Tensor, indices_list):
    out_list = []
    for inds in indices_list:
        subset = data[inds]
        out_list.append(subset)
    return torch.stack(out_list, dim=-1)


def create_bins(data: np.ndarray, n_bin: int):
    data_sorted = np.sort(data)
    length = len(data_sorted)
    step = length // n_bin
    bins = []
    bins.append(data_sorted[0] - 1e-15)
    for i in range(1,n_bin):
        bins.append(data_sorted[step*i])
    bins.append(data_sorted[-1] + 1e-15)
    return np.array(bins)


def check_bins(data: np.ndarray, bins: np.ndarray):
    val_to_bin = np.digitize(data, bins) -1
    val_to_bin = np.clip(val_to_bin, 0, len(bins)-2)
    return val_to_bin