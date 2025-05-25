import torch
import numpy as np
from sklearn.model_selection import KFold

def kfold(data, n_splits):
    """Create k-fold cross validation indices"""
    train_indices = []
    test_indices = []
    
    kf = KFold(n_splits=n_splits, shuffle=True)
    
    # Convert to numpy if needed for sklearn
    if torch.is_tensor(data):
        data_np = data.cpu().numpy()
    else:
        data_np = data
    
    for train_index, test_index in kf.split(data_np):
        train_indices.append(torch.tensor(train_index, dtype=torch.long))
        test_indices.append(torch.tensor(test_index, dtype=torch.long))
    
    train_ind = torch.stack(train_indices)
    test_ind = torch.stack(test_indices)
    
    return train_ind, test_ind

def data_split(data, ind):
    """Divide data in training and test set"""
    # Handle 2D data by adding a dimension
    if data.dim() == 2:
        data = data.unsqueeze(-1)
    
    n_splits = ind.shape[0]
    data_splits = []
    
    for j in range(n_splits):
        # Gather data for this split
        data_split = data[ind[j]]
        data_splits.append(data_split)
    
    # Stack all splits
    data_new = torch.stack(data_splits)
    # Transpose to match expected output shape
    data_new = data_new.permute(1, 2, 3, 0)
    
    return data_new

############### BIN FUNCTIONS

def create_bins(data, n_bin):
    """Create bins for data discretization"""
    if torch.is_tensor(data):
        data_np = data.cpu().numpy()
    else:
        data_np = data
    
    len_bin = len(data_np) // n_bin
    data_sorted = np.sort(data_np)
    
    bins = [data_sorted[0] - 1e-15]
    for i in range(1, n_bin):
        bins.append(data_sorted[len_bin * i])
    bins.append(data_sorted[-1] + 1e-15)
    
    return bins

def check_bins(data, bins):
    """Check and correct bin assignments"""
    n_bin = len(bins) - 1
    len_bin = len(data) // n_bin
    
    if torch.is_tensor(data):
        data_np = data.cpu().numpy()
    else:
        data_np = data
    
    val_to_bin = np.digitize(data_np, bins) - 1
    
    ind_sort = np.argsort(data_np)
    
    val_to_bin2 = val_to_bin.copy()
    for bb in range(n_bin):
        val_to_bin2[ind_sort[bb*len_bin:(bb+1)*len_bin]] = bb
    
    return val_to_bin2 