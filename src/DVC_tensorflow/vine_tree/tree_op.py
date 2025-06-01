import math as m
import numpy as np
import random
from scipy.stats import kendalltau, gaussian_kde

###################### ENTROPY ESTIMATION FUNCTIONS ##########################

def empirical_cdf(data):
    """Transform data to empirical CDF (copula scale)"""
    ranks = np.argsort(np.argsort(data))
    return (ranks + 0.5) / len(data)

def estimate_copula_entropy_kde(u, v):
    """Estimate copula entropy using KDE method"""
    try:
        # Transform to copula scale [0,1]
        u_copula = empirical_cdf(u)
        v_copula = empirical_cdf(v)
        
        # Stack for bivariate KDE
        data = np.vstack([u_copula, v_copula])
        
        # Estimate copula density using KDE
        kde = gaussian_kde(data)
        
        # Estimate entropy using sample-based approximation
        # H ≈ -1/n Σ log c(ui, vi)
        n_entropy_samples = min(500, len(u_copula))
        indices = np.random.choice(len(u_copula), n_entropy_samples, replace=False)
        
        sample_points = data[:, indices]
        log_densities = np.log(kde(sample_points) + 1e-10)  # Add small epsilon for stability
        
        entropy = -np.mean(log_densities)
        return entropy
        
    except Exception as e:
        # Fallback to tau-based proxy
        return estimate_copula_entropy_fallback(u, v)

def estimate_copula_entropy_histogram(u, v):
    """Estimate copula entropy using histogram method"""
    try:
        # Transform to copula scale
        u_copula = empirical_cdf(u)
        v_copula = empirical_cdf(v)
        
        # Create 2D histogram
        bins = min(20, int(np.sqrt(len(u_copula))))
        hist, u_edges, v_edges = np.histogram2d(u_copula, v_copula, bins=bins)
        
        # Normalize to get density
        bin_area = (u_edges[1] - u_edges[0]) * (v_edges[1] - v_edges[0])
        density = hist / (np.sum(hist) * bin_area + 1e-10)
        
        # Calculate entropy
        density_pos = density[density > 0]
        entropy = -np.sum(density_pos * np.log(density_pos)) * bin_area
        
        return entropy
        
    except Exception as e:
        # Fallback to tau-based proxy
        return estimate_copula_entropy_fallback(u, v)

def estimate_copula_entropy_fallback(u, v):
    """Fallback entropy estimate using mutual information proxy"""
    try:
        # Use correlation as proxy for information content
        tau, _ = kendalltau(u, v)
        
        # Convert to entropy-like measure
        # Higher |tau| -> higher information -> higher entropy
        entropy_proxy = -0.5 * np.log(1 - tau**2 + 1e-10)
        
        return entropy_proxy
        
    except Exception as e:
        # Last resort fallback
        return 0.0

def estimate_copula_entropy(u, v, method='kde'):
    """
    Main entropy estimation function
    
    Parameters:
    -----------
    u, v : array-like
        Data for entropy estimation
    method : str
        'kde', 'histogram', or 'fallback'
        
    Returns:
    --------
    entropy : float
        Estimated entropy value
    """
    if method == 'kde':
        return estimate_copula_entropy_kde(u, v)
    elif method == 'histogram':
        return estimate_copula_entropy_histogram(u, v)
    elif method == 'fallback':
        return estimate_copula_entropy_fallback(u, v)
    else:
        # Default to KDE
        return estimate_copula_entropy_kde(u, v)

###################### CHECK OPTIMAL TREE ##########################

### Prim algorithm - Section 23 Minium Spanning Three - Algorithm book

def optimal_tree(data, data_flip, ind_vine, tr, rand=False, optimization_method='tau'):
    """
    Build optimal tree using different optimization criteria
    
    Parameters:
    -----------
    data : array
        Data matrix for current tree level
    data_flip : array
        Flipped data for conditional dependencies
    ind_vine : list
        Current vine structure
    tr : int
        Tree level (0 = first tree)
    rand : bool
        Legacy parameter for random optimization (deprecated, use optimization_method='random')
    optimization_method : str
        Optimization criterion: 'tau', 'entropy', 'random'
        
    Returns:
    --------
    edges : list
        Selected edges for this tree level
    weights : list
        Optimization criterion values for selected edges
    """
    
    # Handle legacy rand parameter
    if rand:
        optimization_method = 'random'
    
    random.seed(9001)
    V = set(range(0, data.shape[1] - tr))  # Available variables
    Q = set()  # Selected variables in spanning tree
    edges = []  # Selected edges
    weights = []  # Criterion values
    
    # Start with random variable
    u = random.randint(0, data.shape[1] - 1 - tr)
    Q.add(u)
    V.remove(u)
    
    # Prim's algorithm with different optimization criteria
    while V:
        max_v = -m.inf
        best_u = None
        best_v = None
        
        for i in Q:
            for j in V:
                if tr == 0:
                    # First tree: direct variable relationships
                    if optimization_method == 'tau':
                        tau, p_value = kendalltau(data[:, i], data[:, j])
                        criterion_value = abs(tau)
                        
                    elif optimization_method == 'entropy':
                        entropy = estimate_copula_entropy(data[:, i], data[:, j], method='kde')
                        criterion_value = entropy
                        
                    elif optimization_method == 'random':
                        criterion_value = abs(np.random.uniform(-1., 1., 1)[0])
                        
                    else:
                        raise ValueError(f"Unknown optimization method: {optimization_method}")
                    
                else:
                    # Higher trees: conditional relationships
                    par, inx1, inx2 = parent_var(tr, ind_vine, [i, j])
                    
                    if par is not None:
                        if par != ind_vine[tr-1][i][0]:
                            # Use flipped data
                            data_u = data_flip[:, i] if data_flip is not None else data[:, i]
                            data_v = data[:, j]
                        else:
                            data_u = data[:, i]
                            data_v = data[:, j]
                        
                        if optimization_method == 'tau':
                            tau, p_value = kendalltau(data_u, data_v)
                            criterion_value = abs(tau)
                            
                        elif optimization_method == 'entropy':
                            entropy = estimate_copula_entropy(data_u, data_v, method='kde')
                            criterion_value = entropy
                            
                        elif optimization_method == 'random':
                            criterion_value = abs(np.random.uniform(-1., 1., 1)[0])
                            
                        else:
                            raise ValueError(f"Unknown optimization method: {optimization_method}")
                    else:
                        continue  # Skip invalid edges
                
                # Select edge with maximum criterion value
                if criterion_value > max_v:
                    max_v = criterion_value
                    best_u = i
                    best_v = j
        
        if best_v is not None:
            Q.add(best_v)
            V.remove(best_v)
            edges.append([best_u, best_v])
            weights.append(max_v)
        else:
            break  # No valid edges found
    
    return edges, weights

###################### BUILD LIST OF EDGES #########################

def build_edges(tree_index):
    n_ind = tree_index.shape[0]
    
    e_0 = {tree_index[n_ind-2,n_ind-2],tree_index[n_ind-1,n_ind-2]}
    E_1 = [e_0]
    E = [E_1]
    
    u_union = [[e_0]] 
    
    for i in range(1,n_ind-1,1):
        E.append([])
        u_union.append([])  

    n = n_ind-1
    ind_e = 0
    for i in range(n-2,-1,-1): 
        e_1 = {tree_index[i,i],tree_index[n,i]}
        
        E[0].append(e_1)
        u_union[0].append(e_1) 
        
        e_new = e_1
        u_new = e_1

        for k in range(1,n_ind-i-1,1):
            u = set()
            for uu in range(n-k,n,1):
                u.add(tree_index[uu,i])
            u.add(tree_index[n,i])
            flag = False
            for hh in range(0,len(E[k-1]),1):
                if u_union[k-1][hh].issubset(u):
                    
                    flag = True
                    e_new1 = [e_new,E[k-1][hh]]
                    E[k].append(e_new1)
                    
                    u_union1 = u_new.union(u_union[k-1][hh])
                    
                    u_union[k].append(u_union1)
                    
            if not flag:
                raise Exception('The matrix is not a regular vine')
            e_new = e_new1
            u_new = u_union1
    return E


#################### RETURN LIST ON INDEX #######################

def edges_index(E,r_matrix,tr):
    edges_ind = []
    n = r_matrix.shape[0]-1
    if tr == 0:
        for i in range(n-1,-1,-1):
            ind = [r_matrix[n,i]-1,r_matrix[i,i]-1]
            edges_ind.append(ind)
    else:
        for ii in range(0,len(E[tr]),1):
            edge = E[tr][ii]
            for yy in range(0,len(E[tr-1]),1):
                if edge[0] == E[tr-1][yy]:
                    ind0 = yy
                if edge[1] == E[tr-1][yy]:
                    ind1 = yy
            ind = [ind1,ind0]
            edges_ind.append(ind)
    return edges_ind

################### CHECK IF IS AN EDGE #######################

def isedge(edge,u):
    if type(edge) is list:
        return (isedge(edge[0],u)) & (isedge(edge[1],u))
    else:
        return edge.issubset(u)


############################################ PREPARE C-VINE AND D-VINE MATRIX ################################

def prepare_vine(vine_type, dim):
    if vine_type == 'c-vine':
        r_matrix = np.tril(np.tile(np.array(range(dim,0,-1)),(dim,1)).T)
        
        ### EDGES FOR THE CODES
        ind_vine = []
        for i in range(0,len(r_matrix)-1,1):
            ind_vine1 = []
            for j in range(1,len(r_matrix)-i,1):
                ind_vine1.append([0,j])
            ind_vine.append(ind_vine1)
            
    if vine_type == 'd-vine':
        r_matrix = np.zeros((dim,dim),np.int32)
        for i in range(0,dim,1):
            r_matrix[i,i] = dim-i #-1
        for j in range(0,dim-1,1):
            c = 1 #0
            for i in range(j+1,dim,1):
                r_matrix[i,j] = c
                c += 1
                
        ### EDGES FOR THE CODES        
        ind_vine = []
        for i in range(0,len(r_matrix)-1,1):
            ind_vine1 = []
            for j in range(0,len(r_matrix)-i-1,1):
                ind_vine1.append([j,j+1])
            ind_vine.append(ind_vine1)
        
    ### NODES

    d = len(r_matrix)
    n = d-1
    nodes = np.zeros(d,np.int32)
    for i in range(0,d,1):
        nodes[i]=r_matrix[i,i]

    nodes = np.flip(nodes)
    print('nodes:')
    print(nodes)
    
    ### EDGES of the vine
    matrix_edges = []
    c = 0
    for i in range(n,0,-1):
    #     print('level',c)
        edge1 = []
        for j in range(i-1,-1,-1):
            str1 = '(' + str(r_matrix[i,j]) + ',' + str(r_matrix[j,j])
            c = 0
            for ii in range(i+1,n+1,1):
                if c == 0:
                    str1 = str1 + '|' + str (r_matrix[ii,j])
                else:
                    str1 = str1 + ','  + str (r_matrix[ii,j])
                c += 1
            str1 = str1 + ')'
            edge1.append(str1)
        c += 1
        matrix_edges.append(edge1)
    
    print('edges:')
    for i in range(0,len(matrix_edges),1):
        print(matrix_edges[i])
    return r_matrix, ind_vine, nodes, matrix_edges
    
################  PREPARE REGULAR MATRIX ######################

def prepare_regular(r_matrix):
    E = build_edges(r_matrix)
    
    ### EDGES FOR THE CODES
    
    ind_vine = []
    for i in range(0,len(E),1):
        ind_ee = edges_index(E,r_matrix,i)
        ind_vine.append(ind_ee)
        
    ### NODES

    d = len(r_matrix)
    n = d-1
    nodes = np.zeros(d,np.int32)
    for i in range(0,d,1):
        nodes[i]=r_matrix[i,i]

    nodes = np.flip(nodes)
    print('nodes:')
    print(nodes)
    
    ### EDGES of the vine
    matrix_edges = []
    c = 0
    for i in range(n,0,-1):
    #     print('level',c)
        edge1 = []
        for j in range(i-1,-1,-1):
            str1 = '(' + str(r_matrix[i,j]) + ',' + str(r_matrix[j,j])
            c = 0
            for ii in range(i+1,n+1,1):
                if c == 0:
                    str1 = str1 + '|' + str (r_matrix[ii,j])
                else:
                    str1 = str1 + ','  + str (r_matrix[ii,j])
                c += 1
            str1 = str1 + ')'
            edge1.append(str1)
        c += 1
        matrix_edges.append(edge1)
    
    print('edges:')
    for i in range(0,len(matrix_edges),1):
        print(matrix_edges[i])
    return E, ind_vine, nodes, matrix_edges


###################### GET PARENT VARIABLE ##############################

def parent_var(k,ind_vine,edge):
    u = set()
    u.add(ind_vine[k-1][edge[0]][0])
    u.add(ind_vine[k-1][edge[0]][1])

    u1 = set()
    u1.add(ind_vine[k-1][edge[1]][0])
    u1.add(ind_vine[k-1][edge[1]][1])
    
    parent = None
    inter = u.intersection(u1)
    for elem in inter:
        parent = elem
    return parent, u, u1


######################## CHECK WHEN TO FLIP - R-MATRIX ORDER ###################################

def flip_check_all(ind_vine,tr, binning, n_bin):
    if tr < len(ind_vine)-1:
        ind_ee1 = ind_vine[tr+1]

        u_set = []
        parent = []
        for edges in ind_ee1:
            parent1, inx1, inx2 = parent_var(tr+1,ind_vine,edges)
            u_union = inx1.union(inx2)

            u_set.append(u_union)
            parent.append(parent1)
    else:
        ind_ee1 = [0,1]
        u_set = [{0,1}]
        parent = [0]

    parent_all = []
    edges_now = ind_vine[tr]
    ind_edge_rel1 = []
    flip_flag1 = []

    for j in range(0,len(edges_now),1):
        edge = edges_now[j]
        uu_now = {edge[0],edge[1]}
        parent_now = []
        parent_now_set = set()
        for jj in range(0,len(u_set),1):
            uu = u_set[jj]
            if uu_now.issubset(uu):
                if not {parent[jj]}.issubset(parent_now_set):
                    parent_now.append(parent[jj])
                    parent_now_set.add(parent[jj])
                
        # Check if they are all equal
        if len(set(parent_now)) <= 1:
            parent_now = [parent_now[0]]

        parent_all.append(parent_now)
        for par in parent_now:

            if edge[0] != par:

                flip_flag1.append(True)
            else:
                flip_flag1.append(False)
            ind_edge_rel1.append(j)
    return flip_flag1, ind_edge_rel1, parent_all


################################## PREPARE R-MATRIX OPTIMAL AND RANDOM ###################

def prepare_optimal(d, ind_vine):
    E = []
    uu_uni = []
    par = []
    diff = []
    for tr in range(0,d-1,1):
        E.append([])
        uu_uni.append([])
        par.append([])
        diff.append([])
        
    for ii in range(0,len(ind_vine[0]),1):
        E[0].append({ind_vine[0][ii][0]+1,ind_vine[0][ii][1]+1})  
    
    u_union = set()
    for tr in range(1,d-1,1):
        for ii in range(0,len(ind_vine[tr]),1):
            ind1 = ind_vine[tr][ii]
            E[tr].append([E[tr-1][ind1[0]],E[tr-1][ind1[1]]])
            if tr ==1:
                u_union = E[tr-1][ind1[0]].union(E[tr-1][ind1[1]])
                parent = E[tr-1][ind1[0]].intersection(E[tr-1][ind1[1]])
                diff1 = u_union - parent
            else:
                u_union = uu_uni[tr-1][ind1[0]].union(uu_uni[tr-1][ind1[1]])
                parent = uu_uni[tr-1][ind1[0]].intersection(uu_uni[tr-1][ind1[1]])
                diff1 = u_union - parent
            uu_uni[tr].append(u_union)
            par[tr].append(parent)
            diff[tr].append(diff1)
    
    rr = np.zeros((d,d),np.int32)
    n = len(rr)-1
    
    for tr in range(d-2,-1,-1): #0
#         print('tr',tr)
        ind_list = set()
        for j in range(0,n-tr,1):
#             print('j',j)
            edge = []
            if tr > 0:
                for ii in range(0,len(diff[tr]),1):
                    edge1 = []
                    for elem in diff[tr][ii]:
                        edge1.append(elem)
                    edge.append(edge1)
            else:
                for ii in range(0,len(E[tr]),1):
                    edge1 = []
#                     print('aa',E[tr][ii])
                    for elem in E[tr][ii]:
                        edge1.append(elem)
                    edge.append(edge1)
#             print(edge)
#             print('diff',diff[tr])
#             print('ind_list',ind_list)

            if tr == d-2:
                rr[j,j] = edge[ii][0]
                rr[n-tr,j] = edge[ii][1]

            if (tr > 0) & (tr < d-2):            
                for ii in range(0,len(diff[tr]),1):
#                     print('ii',ii)
                    if {ii}.issubset(ind_list) == False:
                        a1 = edge[ii][0]
                        a2 = edge[ii][1]
#                         print('a1',a1)
                        if j == d-2-tr:
                            rr[j,j] = a1
                        if (rr[j,j] == a1):
                            ind1 = ii
                            ind2 = 1
                            ind_list.add(ind1)
                        elif (rr[j,j] == a2):
                            ind1 = ii
                            ind2 = 0
                            ind_list.add(ind1)

                rr[n-tr,j] = edge[ind1][ind2]
            else:
                for ii in range(0,len(E[tr]),1):
                    if {ii}.issubset(ind_list) == False:
                        a1 = edge[ii][0]
                        a2 = edge[ii][1]

                        if j == d-2-tr:
                            rr[j,j] = a1

                        if (rr[j,j] == a1):
                            ind1 = ii
                            ind2 = 1
                            ind_list.add(ind1)
                        elif (rr[j,j] == a2):
                            ind1 = ii
                            ind2 = 0
                            ind_list.add(ind1)
#                 print('ind1',ind1)
#                 print('ind2',ind2)
                rr[n-tr,j] = edge[ind1][ind2]

#             print(rr)
#             print('----------')

    nodes = np.zeros(d,np.int32)
    V = set(range(1,d+1))
    for i in range(0,d,1):
        nodes[i]=rr[i,i]
        u_nod = {nodes[i]}
        if u_nod.issubset(V):
            V.remove(nodes[i])
    nodes = np.flip(nodes)

    for elem in V:
        ind = np.where(nodes == 0)
        nodes[nodes == 0] = elem
        rr[n-ind[0],n-ind[0]] = elem
    
    
    return rr, E, nodes 

####################################### RANDOM R-MATRIX  #########################################################

def random_tree(vine_depth, ind_vine, tr):
    random.seed(9001)
    V = set(range(0,vine_depth-tr)) #{0,1,2,3,4,5}
    Q = set()
    edges = []
    weights = []
    u = random.randint(0,vine_depth-1-tr)
    Q.add(u)
    V.remove(u)
#     print('Q',Q)
#     print('V',V)
#     c = 0
    while V:
        max_v = -m.inf
        for i in Q:
#             print('u',i)
            for j in V:
#                 print('v',j)
                if tr == 0:
                    tau = np.random.uniform(-1.,1.,1)
                    if abs(tau) > max_v:
                        max_v = abs(tau)
                        u = i
                        v = j
                else: 
                    par, inx1, inx2 = parent_var(tr,ind_vine,[i,j])
#                     print('par',par)
#                     print('ind_vine prev 1',ind_vine[tr-1][i])
                    if par != None:
                        if par != ind_vine[tr-1][i][0]:
                            tau = np.random.uniform(-1.,1.,1)
                            if abs(tau) > max_v:
                                max_v = abs(tau)
                                u = i
                                v = j
                        else:
                            tau = np.random.uniform(-1.,1.,1)
                            if abs(tau) > max_v:
                                max_v = abs(tau)
                                u = i
                                v = j
        Q.add(v)
        V.remove(v)
#         print('---------')
#         if v>u:
#             edges.append([v,u])
#         else:

#         if c == 0:
#             edges.append([v,u])
#         else:
        edges.append([u,v])
        weights.append(max_v)
#         c += 1
    return edges,weights


def random_r_matrix_gen(dim):
    ind_vine = []
    for i in range(0,dim-1,1):
        ind_vine.append([])

    for tr in range(0,dim-1,1):
        ind_ee, weights = random_tree(dim,ind_vine,tr)
        ind_vine[tr] = ind_ee

    r_matrix, nodes, E = prepare_optimal(dim,ind_vine)
    return r_matrix, ind_vine, nodes, E