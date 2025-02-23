###############################################
# src/torch_vine/vine_tree.py
###############################################

import math
import numpy as np
import random
from scipy.stats import kendalltau

def parent_var(k, ind_vine, edge):
    """
    Return the 'parent' variable from the sets of previous edges in the vine.
    """
    if k ==0:
        return None, None, None
    e1, e2 = edge[0], edge[1]
    uprev1 = set(ind_vine[k-1][e1])
    uprev2 = set(ind_vine[k-1][e2])
    inter = uprev1.intersection(uprev2)
    parent = None
    if len(inter)>0:
        parent = inter.pop()
    return parent, uprev1, uprev2


def optimal_tree(data, data_flip, ind_vine, tr, rand_flag=False):
    """
    Attempt a MST approach using |kendalltau|.
    """
    dimension = data.shape[1]
    V = set(range(dimension))
    Q = set()
    edges = []
    weights = []
    start_ = random.randint(0, dimension-1)
    Q.add(start_)
    V.remove(start_)
    while V:
        best_abs_tau = -999.0
        best_u, best_v = None, None
        for i in Q:
            for j in V:
                if rand_flag:
                    tau_val = random.uniform(-1.,1.)
                else:
                    tau_val,_ = kendalltau(data[:,i], data[:,j])
                if abs(tau_val)>abs(best_abs_tau):
                    best_abs_tau = tau_val
                    best_u = i
                    best_v = j
        Q.add(best_v)
        V.remove(best_v)
        edges.append([best_u, best_v])
        weights.append(best_abs_tau)
    return edges, weights


def random_tree(vine_depth, ind_vine, tr):
    dimension = vine_depth - tr
    V = set(range(dimension))
    Q = set()
    edges = []
    weights = []
    start_ = random.randint(0, dimension-1)
    Q.add(start_)
    V.remove(start_)
    while V:
        best_ = random.uniform(-1,1)
        best_u, best_v = None, None
        for i in Q:
            for j in V:
                w = random.uniform(-1,1)
                if abs(w)>abs(best_):
                    best_ = w
                    best_u = i
                    best_v = j
        Q.add(best_v)
        V.remove(best_v)
        edges.append([best_u, best_v])
        weights.append(best_)
    return edges, weights


def prepare_optimal(d, ind_vine):
    """
    Build an R-matrix from the set of edges in ind_vine for an R-vine.
    """
    r_matrix = np.zeros((d,d), dtype=int)
    for i in range(d):
        r_matrix[i,i] = d - i
    E = []
    for tr in range(d-1):
        E.append(ind_vine[tr] if tr<len(ind_vine) else [])
    nodes = r_matrix.diagonal()[::-1]
    return r_matrix, E, nodes


def prepare_regular(r_matrix):
    """
    from a user-supplied r_matrix, build E, ind_vine, nodes, matrix_edges 
    for an R-vine.
    """
    d = r_matrix.shape[0]
    n = d-1
    E = []
    ind_vine = []
    for tr in range(n):
        E.append([])
        ind_vine.append([])
    nodes = r_matrix.diagonal()[::-1]
    matrix_edges = []
    for i in range(n,0,-1):
        edges_level = []
        for j in range(i-1,-1,-1):
            e_str = '('+str(r_matrix[i,j])+','+str(r_matrix[j,j])
            c=0
            for ii in range(i+1,n+1):
                if c==0:
                    e_str += '|'+str(r_matrix[ii,j])
                else:
                    e_str += ','+str(r_matrix[ii,j])
                c+=1
            e_str += ')'
            edges_level.append(e_str)
        matrix_edges.append(edges_level)
    return E, ind_vine, nodes, matrix_edges


def prepare_vine(vine_family, dim):
    if vine_family=='c-vine':
        r_matrix = np.tril(np.tile(np.arange(dim,0,-1),(dim,1)).T)
    elif vine_family=='d-vine':
        r_matrix = np.zeros((dim,dim), dtype=int)
        for i in range(dim):
            r_matrix[i,i] = dim-i
        for j in range(dim-1):
            c=1
            for i in range(j+1, dim):
                r_matrix[i,j] = c
                c+=1
    else:
        r_matrix = np.eye(dim, dtype=int)
    E, ind_vine, nodes, matrix_edges = prepare_regular(r_matrix)
    return r_matrix, ind_vine, nodes, matrix_edges


def flip_check_all(ind_vine, tr, binning, n_bin):
    flip_flag1 = []
    ind_edge_rel1 = []
    parent_all = []
    edges_now = ind_vine[tr] if tr<len(ind_vine) else []
    for j,edge in enumerate(edges_now):
        flip_flag1.append(False)
        ind_edge_rel1.append(j)
        parent_all.append([])
    return flip_flag1, ind_edge_rel1, parent_all