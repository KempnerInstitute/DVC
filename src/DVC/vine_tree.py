###############################################
# src/DVC/vine_tree.py
###############################################

import math
import numpy as np
import random
from scipy.stats import kendalltau

def parent_var(k, ind_vine, edge):
    """
    For a given edge at level k in the vine, find the 'parent' variable
    from the previous level k-1's edges.

    In a vine, each edge e might come from two edges e1,e2 in the prior level,
    we gather their sets, find intersection => the 'parent'.

    Args:
      k: int, the vine level
      ind_vine: a list of lists => ind_vine[level], each containing edges
      edge: [e1, e2], indices in the prior level
    Returns:
      (parent, uprev1, uprev2)
        parent: the variable in intersection
        uprev1,uprev2: sets from ind_vine[k-1][e1], ind_vine[k-1][e2] for reference
    """
    if k == 0:
        # no prior level => no parent
        return None, None, None
    e1, e2 = edge[0], edge[1]
    uprev1 = set(ind_vine[k-1][e1]) if (k-1<len(ind_vine) and e1<len(ind_vine[k-1])) else set()
    uprev2 = set(ind_vine[k-1][e2]) if (k-1<len(ind_vine) and e2<len(ind_vine[k-1])) else set()
    inter = uprev1.intersection(uprev2)
    parent = None
    if len(inter)>0:
        # pick the first from intersection
        parent = next(iter(inter))
    return parent, uprev1, uprev2


def optimal_tree(data, data_flip, ind_vine, tr, rand_flag=False):
    """
    Attempt a MST approach using |kendalltau| as "distance" (though we pick the max correlation).

    data, data_flip: shape [N, dimension], used if we do flipping logic
    ind_vine: structure to check if we must flip or not
    tr: current level in the vine
    rand_flag: if True => randomly pick correlation
    Returns:
      edges: list of [u,v]
      weights: list of correlation / tau
    """
    dimension = data.shape[1]
    V = set(range(dimension))
    Q = set()
    edges = []
    weights = []
    if dimension<1:
        return edges, weights
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
                    # compute Kendall's tau
                    # optionally see if we do data_flip
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
    """
    Random approach to build a "tree" (like a MST but purely random).
    vine_depth = dimension
    ind_vine: not heavily used, but might be references
    tr: current level
    """
    dimension = vine_depth - tr
    if dimension<1:
        return [], []
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
    Build an R-matrix from the set of edges in ind_vine for an R-vine
    after an 'optimal' or 'random' build approach.

    We do:
      r_matrix: shape [d,d], set diag => d..1
      fill edges in some order => minimal usage
      E: just a reference to edges
      nodes: the diagonal
    """
    r_matrix = np.zeros((d,d), dtype=int)
    for i in range(d):
        r_matrix[i,i] = d - i
    # E => each level => from ind_vine
    E = []
    for tr in range(d-1):
        if tr<len(ind_vine):
            E.append(ind_vine[tr])
        else:
            E.append([])

    nodes = r_matrix.diagonal()[::-1]
    return r_matrix, E, nodes


def prepare_regular(r_matrix):
    """
    From a user-supplied r_matrix, build E,ind_vine,nodes,matrix_edges for an R-vine.
    Typically used when method=='matrix'.
    We interpret r_matrix shape => [d,d].

    Steps:
      1) build empty E, ind_vine
      2) fill 'nodes' from diag
      3) build matrix_edges => string representation
    """
    d = r_matrix.shape[0]
    n = d-1
    E = []
    ind_vine = []
    for tr in range(n):
        E.append([])
        ind_vine.append([])
    nodes = r_matrix.diagonal()[::-1]

    # build matrix_edges => a list of levels
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
    """
    Build a c-vine or d-vine r_matrix. 
    c-vine => a lower-tri matrix with diag => dim..1
    d-vine => diag => dim..1 plus a pattern in below diag
    Then we pass to prepare_regular.
    """
    if vine_family=='c-vine':
        # shape => np.tril
        arr_ = np.arange(dim,0,-1)
        mat_ = np.tile(arr_, (dim,1))
        r_matrix = np.tril(mat_.T)
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
        # fallback => identity
        r_matrix = np.eye(dim, dtype=int)
    E, ind_vine, nodes, matrix_edges = prepare_regular(r_matrix)
    return r_matrix, ind_vine, nodes, matrix_edges


def flip_check_all(ind_vine, tr, binning, n_bin):
    """
    Check which edges should be 'flipped' for the next level in an R-vine.
    In your original code, we compared parents to see if we need to flip one variable.
    For now, we do a minimal approach returning all false.

    If you want a full approach:
      - if tr < len(ind_vine)-1:
          check the next level's edges
          see if intersection => if parent's not the left => flip
      else => no flipping

    We'll do a minimal: return no flips so code won't break.

    Returns:
      flip_flag1: list of bool => if we flip that edge
      ind_edge_rel1: same length => the edge index
      parent_all: if we store parents or not
    """
    flip_flag1 = []
    ind_edge_rel1 = []
    parent_all = []
    if tr<len(ind_vine):
        edges_now = ind_vine[tr]
        for j,edge in enumerate(edges_now):
            flip_flag1.append(False)
            ind_edge_rel1.append(j)
            parent_all.append([])
    return flip_flag1, ind_edge_rel1, parent_all