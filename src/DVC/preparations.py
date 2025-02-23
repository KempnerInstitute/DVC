###############################################
# src/torch_vine/preparation.py
###############################################

import torch
import numpy as np
from .objects import margin_obj
from .vine_tree import prepare_vine, prepare_regular, random_r_matrix_gen

def define_copulas(vine_type: str, method: str, binning: bool,
                   n_bin: int, dim: int):
    if vine_type=='r-vine':
        if method=='matrix':
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
            r_matrix, ind_vine_, nodes_, E_ = random_r_matrix_gen(dim)
            ind_vine = ind_vine_
            nodes = nodes_
            matrix_edges = []
            E = E_
        else:
            r_matrix = np.eye(dim, dtype=int)
            ind_vine = []
            nodes = []
            matrix_edges = []
            E=[]
    else:
        r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, dim)
        E=[]

    margin_vine = []
    for i in range(dim):
        margin_vine.append(margin_obj('norm', [0,1], True))

    cop_vine = []
    for tr in range(dim-1):
        cop_vine_lvl = []
        for col in range(dim-1-tr):
            if not binning or tr==0:
                cop_vine_lvl.append("gaussian")
            else:
                bin_list = []
                for _ in range(n_bin):
                    bin_list.append("gaussian")
                cop_vine_lvl.append(bin_list)
        cop_vine.append(cop_vine_lvl)

    return r_matrix, cop_vine, ind_vine, nodes, matrix_edges, margin_vine


def prep_cop(x: np.ndarray, vine1, sort_n: str):
    e = x.copy()
    n = e.shape[0]
    d = e.shape[1]
    for i in range(d):
        col = e[:, i]
        srtd = np.sort(col)
        ranks = np.searchsorted(srtd, col)
        e[:, i] = (ranks+1)/(n+1)
        vine1.margin[i].ker = e[:, i]
    return e