import matplotlib.pyplot as plt
import numpy as np
from tree.tree_op import edges_index

def plot_vine(typ, vine):
    d = vine.n_cop
    if typ == 'cdf':
        fig, ax = plt.subplots(d, d, sharex='col', sharey='row')
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.suptitle('VINE', weight='bold')
    elif typ == 'pdf':
        fig, ax = plt.subplots(d, d ) #, sharex='col', sharey='row')
        fig.subplots_adjust(hspace=0.7, wspace=0.7)
        fig.suptitle('VINE - COPULA PDF')
        
    n = len(vine.r_matrix) - 1
    
    ### PLOT CDF, PDF, ecc
    if typ == 'cdf':
        tr = 0
        for i in range(n,0,-1):
            ind_ee = edges_index(vine.E,vine.r_matrix,tr)
            c = 0
            for j in range(i-1,-1,-1):
                edg = ind_ee[c]
                ax[i,j].scatter(vine.theta[:,tr,edg[0]],vine.theta[:,tr,edg[1]], s=0.1 ,marker = '.')
                c += 1
            tr += 1
    elif typ == 'pdf':
        tr = 0
        for i in range(n,0,-1):
            c = 0
            for j in range(i-1,-1,-1):
                ax[i,j].imshow(vine.copulas[tr].pd_grid_uv[:,:,c], cmap="jet")
                c += 1
            tr += 1
    
    ## PLOT SUBTITLE
    for i in range(n,-1,-1):
        for j in range(i-1,-1,-1):
            str1 = '(' + str(vine.r_matrix[i,j]) + ',' + str(vine.r_matrix[j,j])
            c = 0
            for ii in range(i+1,n+1,1):
                if c == 0:
                    str1 = str1 + '|' + str (vine.r_matrix[ii,j])
                else:
                    str1 = str1 + ','  + str (vine.r_matrix[ii,j])
                c += 1
            str1 = str1 + ')'
            ax[i, j].title.set_text(str1)
    
    ## PLOT INSIDE SQUARES
    r1 = np.flip(vine.r_matrix)
    for i in range(0,n+1,1):
        for j in range(i,n+1,1):
            str1 = str(r1[i,j])
            ax[i, j].text(0.5, 0.5, str1,
                      fontsize=12, ha='center', weight = 'bold')
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    fig
    return