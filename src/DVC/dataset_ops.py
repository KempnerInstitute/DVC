##################################################
# src/DVC/dataset_ops.py
##################################################
import torch
import numpy as np

try:
    from sklearn.model_selection import KFold  # type: ignore
except ImportError:
    class KFold:
        def __init__(self, n_splits=5, shuffle=True, random_state=None):
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state
        def split(self, X):
            n = len(X)
            indices = np.arange(n)
            if self.shuffle:
                rng = np.random.RandomState(self.random_state)
                rng.shuffle(indices)
            fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
            fold_sizes[: n % self.n_splits] += 1
            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                test_index = indices[start:stop]
                train_index = np.concatenate([indices[:start], indices[stop:]])
                yield train_index, test_index
                current = stop

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
    bins = [data_sorted[0] - 1e-15]
    for i in range(1, n_bin):
        bins.append(data_sorted[step * i])
    bins.append(data_sorted[-1] + 1e-15)
    return np.array(bins)

def check_bins(data: np.ndarray, bins: np.ndarray):
    n_bin = bins.size - 1
    val_to_bin = np.digitize(data, bins) - 1
    val_to_bin = np.clip(val_to_bin, 0, n_bin - 1)
    sorted_indices = np.argsort(data)
    length = len(data)
    chunk_size = length // n_bin
    val_to_bin2 = val_to_bin.copy()
    for bb in range(n_bin):
        start_idx = bb * chunk_size
        end_idx = (bb + 1) * chunk_size
        if bb == n_bin - 1:
            end_idx = length
        these_inds = sorted_indices[start_idx:end_idx]
        val_to_bin2[these_inds] = bb
    return val_to_bin2