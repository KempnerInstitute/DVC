# src/utils/tree_op.py
import numpy as np
import random
import math

def prepare_c_vine(dim: int) -> np.ndarray:
    """
    Build a c-vine r_matrix: lower-triangular matrix with decreasing integers.
    """
    r_matrix = np.tril(np.tile(np.arange(dim, 0, -1), (dim, 1)).T)
    return r_matrix

def prepare_d_vine(dim: int) -> np.ndarray:
    """
    Build a d-vine r_matrix.
    """
    r_matrix = np.zeros((dim, dim), dtype=np.int32)
    for i in range(dim):
        r_matrix[i, i] = dim - i
    for j in range(dim - 1):
        c = 1
        for i in range(j + 1, dim):
            r_matrix[i, j] = c
            c += 1
    return r_matrix

def random_r_matrix(dim: int) -> np.ndarray:
    """
    Generate a random r-vine matrix.
    """
    perm = np.random.permutation(dim)
    r_matrix = np.zeros((dim, dim), dtype=np.int32)
    for i in range(dim):
        r_matrix[i, i] = perm[i] + 1
    return r_matrix

def edges_index(r_matrix: np.ndarray, tr: int):
    """
    Compute edge index pairs for tree level 'tr'.
    For tr=0, we return pairs (0,j) for j in 1..(dim-1).
    For higher trees, a simple sequential pairing is used.
    """
    dim = r_matrix.shape[0]
    edges = []
    if tr == 0:
        for j in range(1, dim):
            edges.append((0, j))
    else:
        for j in range(1, dim - tr):
            edges.append((j - 1, j))
    return edges