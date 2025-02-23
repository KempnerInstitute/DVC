###############################################
# src/DVC/preparation.py
###############################################

import torch
import numpy as np
from .objects import margin_obj
from .vine_tree import prepare_vine, prepare_regular, random_r_matrix_gen

def define_copulas(vine_type: str,
                   method: str,
                   binning: bool,
                   n_bin: int,
                   dim: int):
    """
    Build the vine structure (r_matrix, edges, etc.) plus default margin objects
    and the placeholder 'cop_vine' that indicates which family is used at each edge.

    Args:
      vine_type: 'r-vine', 'c-vine', or 'd-vine'
      method:    'matrix' => user-supplied r_matrix (example) or random, or empty
      binning:   bool => if True, we build nested bin families
      n_bin:     number of bins
      dim:       dimension d

    Returns:
      r_matrix, cop_vine, ind_vine, nodes, matrix_edges, margin_vine
    """
    # 1) Build the r_matrix / structure
    if vine_type=='r-vine':
        if method=='matrix':
            # use a typical example
            if dim==3:
                r_matrix = np.array([[3,0,0],
                                     [2,2,0],
                                     [1,1,1]], dtype=int)
            elif dim==4:
                r_matrix = np.array([[3,0,0,0],
                                     [1,4,0,0],
                                     [2,1,2,0],
                                     [4,2,1,1]], dtype=int)
            else:
                r_matrix = np.eye(dim, dtype=int)
            E, ind_vine, nodes, matrix_edges = prepare_regular(r_matrix)

        elif method=='random':
            # randomly build an r_matrix
            r_matrix, ind_vine_, nodes_, E_ = random_r_matrix_gen(dim)
            ind_vine = ind_vine_
            nodes = nodes_
            matrix_edges = []
            E = E_
        else:
            # fallback => identity
            r_matrix = np.eye(dim, dtype=int)
            ind_vine = []
            nodes = []
            matrix_edges = []
            E = []
    else:
        # c-vine or d-vine
        r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, dim)
        E = []

    # 2) Build margin objects => e.g. 'norm', [0,1], True
    margin_vine = []
    for i in range(dim):
        margin_vine.append(margin_obj('norm', [0,1], True))

    # 3) Build cop_vine => which family is used at each edge
    # if binning => for tr>0, store an array of n_bin families
    # else => store a single family
    cop_vine = []
    for tr in range(dim-1):
        cop_vine_lvl = []
        n_edges = dim -1 - tr
        for col in range(n_edges):
            if (not binning) or (tr==0):
                cop_vine_lvl.append("gaussian")  # or "kercop" in nonparam
            else:
                bin_list = []
                for _ in range(n_bin):
                    bin_list.append("gaussian")
                cop_vine_lvl.append(bin_list)
        cop_vine.append(cop_vine_lvl)

    return r_matrix, cop_vine, ind_vine, nodes, matrix_edges, margin_vine


def prep_cop(x: np.ndarray,
             vine1,
             sort_n: str = 'sort'):
    """
    Preprocess data 'x' => compute ranks in [0,1], store them in vine1.margin[i].ker.

    Additionally, if sort_n == 'rand', we add a small random shift to each tie or rank
    so that no two points are exactly the same.

    This matches the original code's approach for 'prep_copula'.

    Args:
      x: shape [N,d], raw data
      vine1: a vine_obj_bin with 'margin'
      sort_n: 'sort' => standard rank. 'rand' => add small uniform offsets
    Returns:
      e => shape [N,d], the final rank-based data in [0,1].
    """
    e = x.copy()
    N = e.shape[0]
    d = e.shape[1]

    for i in range(d):
        col = e[:, i]
        # sort approach
        srtd = np.sort(col)
        # ranks
        ranks = np.searchsorted(srtd, col)
        # => [0..N-1], shift to [1..N]
        if sort_n=='sort':
            # standard => (ranks+1)/(N+1)
            e[:, i] = (ranks+1)/(N+1)
        elif sort_n=='rand':
            # add small random offset => ranks + uniform(0,1) => / (N+1)
            offsets = np.random.rand(N)
            e[:, i] = (ranks + 1 + offsets*0.999999) / float(N+1)
        else:
            # fallback => do sort approach
            e[:, i] = (ranks+1)/(N+1)

        # store in margin
        vine1.margin[i].ker = e[:, i]

    return e