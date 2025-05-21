###############################################
# src/DVC/dataset_ops.py
###############################################

import torch
import numpy as np

# Optional dependency: scikit-learn provides KFold for data splitting.
# Some environments may not have it installed, so we provide a very small
# fallback with the same interface.  This keeps the public API unchanged and
# avoids import errors during testing when sklearn is unavailable.
try:  # pragma: no cover - behaviour tested via absence of sklearn
    from sklearn.model_selection import KFold  # type: ignore
except Exception:  # pragma: no cover - executed when sklearn missing
    class KFold:  # minimal drop-in replacement
        def __init__(self, n_splits=5, *, shuffle=True, random_state=None):
            self.n_splits = int(n_splits)
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
    """
    Perform K-fold splitting on 'data' (numpy array),
    returning lists of train/test indices for each fold.

    Args:
      data: np.ndarray of shape [N, ...]
      n_splits: number of folds
    Returns:
      train_ind_list, test_ind_list: each is a list of arrays of indices.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)
    train_ind_list = []
    test_ind_list = []
    for train_index, test_index in kf.split(data):
        train_ind_list.append(train_index)
        test_ind_list.append(test_index)
    return train_ind_list, test_ind_list


def data_split(data: torch.Tensor, indices_list):
    """
    Gather subsets of 'data' by the index arrays in 'indices_list',
    then stack them along a new last dimension.

    Args:
      data: torch.Tensor shape [N, ...]
      indices_list: list of 1D arrays (e.g. from kfold)
    Returns:
      A torch.Tensor stacking each subset in the last dimension.
    """
    out_list = []
    for inds in indices_list:
        subset = data[inds]
        out_list.append(subset)
    return torch.stack(out_list, dim=-1)


def create_bins(data: np.ndarray, n_bin: int):
    """
    Partition 'data' (1D) into 'n_bin' bins, producing a bin boundary array.

    Steps:
      1) sort 'data'
      2) pick equally spaced cut points => step*i
      3) add small offsets at the extremes
      4) return an array 'bins' of length n_bin+1

    Args:
      data: shape [N,], 1D
      n_bin: number of bins
    Returns:
      bins: shape [n_bin+1], ascending boundary array
    """
    data_sorted = np.sort(data)
    length = len(data_sorted)
    step = length // n_bin
    bins = []
    # the very first boundary
    bins.append(data_sorted[0] - 1e-15)
    # intermediate cut points
    for i in range(1, n_bin):
        bins.append(data_sorted[step * i])
    # final boundary
    bins.append(data_sorted[-1] + 1e-15)
    return np.array(bins)


def check_bins(data: np.ndarray, bins: np.ndarray):
    """
    Assign each value in 'data' to a bin index in [0..n_bin-1],
    forcibly ensuring that each bin has a roughly equal # of points
    (mirroring logic from the original code).

    Steps:
      1) val_to_bin = np.digitize(...) - 1  => a preliminary bin index
      2) sort data's indices => chunk them in 'n_bin' groups => reassign
         each chunk to a single bin #, ensuring uniform distribution.

    This matches the code where we do e.g.:
       val_to_bin2 = val_to_bin
       for bb in range(n_bin):
          sorted_indices[bb*len_bin : (bb+1)*len_bin] => bin=bb
       ...

    Args:
      data: shape [N]
      bins: shape [n_bin+1], from create_bins
    Returns:
      val_to_bin2: shape [N], each in [0..n_bin-1]
    """
    n_bin = bins.size - 1
    # preliminary
    val_to_bin = np.digitize(data, bins) - 1
    # clip in case any out-of-range
    val_to_bin = np.clip(val_to_bin, 0, n_bin - 1)

    # forcibly reassign to ensure each bin has the same count
    sorted_indices = np.argsort(data)
    length = len(data)
    chunk_size = length // n_bin

    val_to_bin2 = val_to_bin.copy()
    for bb in range(n_bin):
        start_idx = bb * chunk_size
        end_idx = (bb + 1) * chunk_size
        if bb == n_bin - 1:  # last bin => take remainder
            end_idx = length
        these_inds = sorted_indices[start_idx:end_idx]
        val_to_bin2[these_inds] = bb

    return val_to_bin2