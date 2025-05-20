###############################################
# src/DVC/vine_tree.py
###############################################
import math
import numpy as np
import random
from scipy.stats import kendalltau


def parent_var(k, ind_vine, edge):
    """
    For a given edge index at level k in the vine, find the 'parent' variable
    from the previous level (k-1)'s edges.

    Each edge "edge=[e1, e2]" indexes edges in the (k-1)-th level:
      - e1 is an edge index in ind_vine[k-1]
      - e2 is an edge index in ind_vine[k-1]
    We gather the sets of variables from those two edges, uprev1 & uprev2,
    find intersection => the 'parent' variable.

    Args:
      k: int, vine level
      ind_vine: a list of lists => ind_vine[level], each containing edges (like [varA,varB])
      edge: [e1, e2], referencing edges in level k-1
    Returns:
      (parent, uprev1, uprev2)
        parent: The variable in intersection
        uprev1, uprev2: sets from ind_vine[k-1][e1], ind_vine[k-1][e2]
                       used for debugging or flipping logic
    """
    if k == 0:
        return None, None, None
    e1, e2 = edge[0], edge[1]
    # If out-of-range, empty sets
    uprev1 = set(ind_vine[k-1][e1]) if (k-1 < len(ind_vine) and e1 < len(ind_vine[k-1])) else set()
    uprev2 = set(ind_vine[k-1][e2]) if (k-1 < len(ind_vine) and e2 < len(ind_vine[k-1])) else set()
    inter = uprev1.intersection(uprev2)
    parent = None
    if len(inter) > 0:
        # pick any variable from the intersection
        parent = next(iter(inter))
    return parent, uprev1, uprev2


def optimal_tree(data, data_flip, ind_vine, tr, rand_flag=False):
    """
    Build a maximum spanning tree on 'dimension' variables using Kendall's tau (absolute value)
    as the "weight," or random if rand_flag=True.

    data, data_flip: shape [N, dimension], for flipping logic if needed
    ind_vine: not heavily used here but can check flipping in a bigger context
    tr: current vine level
    """
    dimension = data.shape[1]
    V = set(range(dimension))
    Q = set()
    edges = []
    weights = []

    if dimension < 1:
        return edges, weights

    # Instead of random start, pick the variable with highest average correlation
    if not rand_flag and dimension > 1:
        avg_corrs = []
        for i in range(dimension):
            corr_sum = 0.0
            count = 0
            for j in range(dimension):
                if i != j:
                    tau, _ = kendalltau(data[:, i], data[:, j])
                    if math.isfinite(tau):
                        corr_sum += abs(tau)
                        count += 1
            avg_corrs.append(corr_sum / max(1, count))
        start_ = np.argmax(avg_corrs)
    else:
        start_ = random.randint(0, dimension-1)

    Q.add(start_)
    V.remove(start_)

    while V:
        best_abs_tau = -999.0
        best_u, best_v = None, None
        for i in Q:
            for j in V:
                if rand_flag:
                    tau_val = random.uniform(-1., 1.)
                else:
                    tau_val,_ = kendalltau(data[:, i], data[:, j])
                    # Ensure valid correlation - in rare cases kendalltau can return nan
                    if not math.isfinite(tau_val):
                        # Fallback to Pearson correlation
                        tau_val = np.corrcoef(data[:, i], data[:, j])[0, 1]
                        if not math.isfinite(tau_val):
                            tau_val = 0.0  # Last resort
                if abs(tau_val) > abs(best_abs_tau):
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
    tr = current vine level
    """
    dimension = vine_depth - tr
    if dimension < 1:
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
                if abs(w) > abs(best_):
                    best_ = w
                    best_u = i
                    best_v = j
        Q.add(best_v)
        V.remove(best_v)
        edges.append([best_u, best_v])
        weights.append(best_)

    return edges, weights


def random_r_matrix_gen(dim):
    """
    Creates a random R-matrix for an R-vine by building random edges
    and calling 'prepare_optimal'.

    Steps:
      1) We'll store random edges in ind_vine (just at level 0 for demonstration).
      2) Then call prepare_optimal to produce final R-matrix and E,nodes
      3) Return (r_matrix, ind_vine, nodes, E)
    """
    ind_vine = []

    # single level of random edges
    edges, weights = random_tree(dim, ind_vine, 0)

    # place them in ind_vine[0]
    ind_vine.append([])
    for e in edges:
        ind_vine[0].append(e)

    r_matrix, E, nodes = prepare_optimal(dim, ind_vine)
    return r_matrix, ind_vine, nodes, E


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
    r_matrix = np.zeros((d, d), dtype=int)
    for i in range(d):
        r_matrix[i,i] = d - i

    E = []
    for tr in range(d-1):
        if tr < len(ind_vine):
            E.append(ind_vine[tr])
        else:
            E.append([])

    nodes = r_matrix.diagonal()[::-1]
    return r_matrix, E, nodes


def prepare_regular(r_matrix):
    """
    From a user-supplied r_matrix, build (E, ind_vine, nodes, matrix_edges) for an R-vine.
    Typically used when method=='matrix'.

    Steps:
      - E, ind_vine => placeholders
      - nodes => diag reversed
      - matrix_edges => string representation
    """
    d = r_matrix.shape[0]
    n = d - 1

    E = []
    ind_vine = []
    for tr in range(n):
        E.append([])
        ind_vine.append([])

    nodes = r_matrix.diagonal()[::-1]

    matrix_edges = []
    for i in range(n,0,-1):
        edges_level = []
        for j in range(i-1, -1, -1):
            e_str = '(' + str(r_matrix[i,j]) + ',' + str(r_matrix[j,j])
            c = 0
            for ii in range(i+1, n+1):
                if c == 0:
                    e_str += '|' + str(r_matrix[ii,j])
                else:
                    e_str += ',' + str(r_matrix[ii,j])
                c += 1
            e_str += ')'
            edges_level.append(e_str)
        matrix_edges.append(edges_level)

    return E, ind_vine, nodes, matrix_edges


def prepare_vine(vine_family, dim):
    """
    Build a c-vine or d-vine r_matrix. 
    c-vine => a lower-tri matrix with diag => [dim..1]
    d-vine => diag => [dim..1] plus a pattern in the below diag
    Then we pass to prepare_regular(...) to get E, ind_vine, nodes, matrix_edges.
    """
    if vine_family == 'c-vine':
        # Diagonal (dim .. 1)
        arr_ = np.arange(dim, 0, -1)
        mat_ = np.tile(arr_, (dim, 1))
        r_matrix = np.tril(mat_.T)

        # Build explicit edge list: root variable k connected to k+1 .. d-1
        ind_vine = []
        for k in range(dim - 1):
            lvl_edges = [[k, j] for j in range(k + 1, dim)]
            ind_vine.append(lvl_edges)

    elif vine_family == 'd-vine':
        # Construct canonical d-vine R-matrix (lower-triangular numbering)
        r_matrix = np.zeros((dim, dim), dtype=int)
        for i in range(dim):
            r_matrix[i, i] = dim - i
        for j in range(dim - 1):
            c = 1
            for i in range(j + 1, dim):
                r_matrix[i, j] = c
                c += 1

        # Edge list – consecutive chain on level-0, shorter ranges after
        ind_vine = []
        # level-0
        ind_vine.append([[j, j + 1] for j in range(dim - 1)])
        # higher levels
        for k in range(1, dim - 1):
            lvl_edges = [[j, j + k + 1] for j in range(dim - k - 1)]
            ind_vine.append(lvl_edges)

    else:
        # Fallback: identity matrix, no edges.
        r_matrix = np.eye(dim, dtype=int)
        ind_vine = []

    # Nodes and matrix_edges reused from prepare_regular for consistency.
    _, _, nodes, matrix_edges = prepare_regular(r_matrix)
    return r_matrix, ind_vine, nodes, matrix_edges


def flip_check_all(ind_vine, tr, binning, n_bin):
    """
    The full flipping logic, as used in the original code:

    We want to check the edges in 'ind_vine[tr]' and see how they are used in 'ind_vine[tr+1]'.
    If the parent variable at next level doesn't match the "left" variable of the current edge,
    we set flip_flag=True. This means we interpret that next edge needs to "flip" the order.

    If binning is True, we might do extra bin-based logic, but let's replicate a typical approach:
      1) gather next-level edges => for each next edge, find parent => see if parent's left side
      2) if parent is different => flip => True
      3) store (flip_flag1, ind_edge_rel1, parent_all)
    """
    flip_flag1 = []
    ind_edge_rel1 = []
    parent_all = []

    edges_now = []
    if tr < len(ind_vine):
        edges_now = ind_vine[tr]

    # If there's no next level, no flipping
    if tr >= len(ind_vine)-1:
        for j, e in enumerate(edges_now):
            flip_flag1.append(False)
            ind_edge_rel1.append(j)
            parent_all.append([])
        return flip_flag1, ind_edge_rel1, parent_all

    # There is a next level => check how each edge is used by next-level edges
    next_edges = ind_vine[tr+1]

    # We'll do a small helper: we create an "edge usage" map => next_edge -> parent variable
    # then see if the parent's in edges_now[e][0] or not
    # but each next_edge is [u,v], referencing edges in level 'tr', so we can see if 'j' in [u,v]
    # then we do parent_var(...) to see what the parent's actual variable is
    # if that parent's not edges_now[j][0], we flip => True

    # We'll store for each e in edges_now => whether we flip or not
    for j, e in enumerate(edges_now):
        # e is a 2-variable set, e[0] is "left", e[1] is "right"
        # we check if some next_edge references j => means next_edge = [j, X] or [X, j]
        # we find the next_edge that references j => call parent_var => see if parent's in e
        flip_me = False
        par_list = []

        # gather how many next_edges reference 'j'
        for ne_idx, ne in enumerate(next_edges):
            if j in ne:
                # find the parent variable of next_edge
                par, up1, up2 = parent_var(tr+1, ind_vine, ne)
                # store for debugging
                if par is not None:
                    par_list.append(par)
                # see if par matches e[0], if not => flip
                if par is not None and (e[0] != par):
                    flip_me = True

        flip_flag1.append(flip_me)
        ind_edge_rel1.append(j)
        parent_all.append(par_list)

    # If binning is True, we might do finer logic, e.g. flipping each bin separately,
    # but let's assume the logic is the same except repeated. 
    # We'll keep it this way for demonstration.
    return flip_flag1, ind_edge_rel1, parent_all