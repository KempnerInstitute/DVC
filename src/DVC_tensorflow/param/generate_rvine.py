import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from scipy import stats

from classes.objects import margin_obj
from param.margin_pdf import *
from param.margin_op import *
from param.cond_copula import *
from utils.dataset_op import create_bins
from utils.prob_op import kernel_cdf
from vine_tree.tree_op import parent_var
from grid.grid_op import mk_grid


###################### GENERATE SAMPLES ##############################

def generate_r_samples(cases, r_matrix, ind_vine, nodes, margin_vine, cop_vine, n_bin, binning):
    d = len(r_matrix)
    n = len(r_matrix) -1

    w = np.random.uniform(0,1,(cases,d))
    w = w.astype(np.float32)
    
    tau_bins = []
    tau_corr = []
    for tr in range(0,d-1,1):
        tau_bins1 = []
        tau_corr1 = []
        for col in range(0,d-1-tr,1):
            tau_bins11 = []
            for bb in range(0,n_bin,1):
                tau_bins11.append([])
            tau_bins1.append(tau_bins11)
            tau_corr1.append([])
        tau_bins.append(tau_bins1)
        tau_corr.append(tau_corr1)

    

    v = np.zeros([cases,d,d],w.dtype)
    v_flip = np.zeros([cases,d,d],w.dtype)
    v[:,0,0] = w[:,0]
    
    for i in range(1,d,1):
        v[:,i,i] = w[:,i]

        c = 0
        for k in range(i-1,-1,-1):
            tr = k
            col = i-k-1
            ind_now = ind_vine[k][c]

            ### To fix D-VINE
            # ind_array = np.array(vine.ind_edge_rel[tr])
            # ind_col = np.where(ind_array == col)
            # col = ind_col[0][0]

            if k == 0:
                tr1 = n-k
                col1 = n-i
                ind1 = r_matrix[tr1,col1] #- 1
                ind1 = np.where(nodes == ind1)
                ind1 = ind1[0][0]

                v2 = v[:,k,ind1][...,np.newaxis]
                v1 = v[:,k+1,i][...,np.newaxis]
                vv = np.concatenate((v1,v2),1)
                v[:,k,i] = copulainvccdf(cop_vine[tr][col],vv)
                
                tau, p_value = stats.kendalltau(vv[:,1],v[:,k,i])
                tau_corr[tr][col] = tau
                tau_bins[tr][col] = tau
            else:

                parent, inx1, inx2 = parent_var(k,ind_vine,ind_now)

                if ind_vine[k-1][ind_now[0]][0] != parent: 
                    v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
                else:
                    v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

                v1 = v[:,k+1,i][...,np.newaxis]
                vv = np.concatenate((v1,v2),1)

                ### try to fix d-vine sampling
                col = i-k-1

                if binning == False:
                    v[:,k,i] = copulainvccdf(cop_vine[tr][col],vv)
                    
                    tau, p_value = stats.kendalltau(vv[:,1],v[:,k,i])
                    tau_corr[tr][col] = tau
                    
                else:
                    
                    ind1 = parent
                    
                    if k == 1:
                        ind1 = np.where(nodes == ind1 +1)
                        ind1 = ind1[0][0]
                        bins = create_bins(v[:,k-1,ind1],n_bin)
                        val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                    else:
                        ind_par_now = ind_vine[k-1][ind_now[1]]
                        parent22, inx1, inx2 = parent_var(k-1,ind_vine,ind_par_now)  

                        ind1 = ind1 + k - 1
                        if (ind_vine[k-2][ind_par_now[0]][0] == parent22):
                            bins = create_bins(v[:,k-1,ind1],n_bin)
                            val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                        else:
                            bins = create_bins(v_flip[:,k-1,ind1],n_bin)
                            val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1

                    vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                    for bb in range(0,n_bin,1):
                        mask = np.where(val_to_bin == bb)
                        vv_bin = vv[mask[0],:]
                        
                        ### CDF FORCE UNIFORM
                        vv_bin_new = vv_bin
                        u_1, ex_u = mk_grid(tf.convert_to_tensor(50),vv_bin.dtype)
                        for zz in range(0,2,1):
                            vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],ex_u)
                        vv_bin = vv_bin_new
                        ###
                        
                        v[mask[0],k,i] = copulainvccdf(cop_vine[tr][col][bb],vv_bin)
                        
                        tau, p_value = stats.kendalltau(vv_bin[:,1],v[mask[0],k,i])
                        print('Tau value bin -',bb, '- is: ', tau)
                        tau_bins[tr][col][bb] = tau
#                         tau_binned.append(tau)
                        
                        corr = stats.pearsonr(vv_bin[:,1],v[mask[0],k,i])
                        print('Corr value  UV space, bin(',bb,')',corr[0])
            c += 1
        
        print('-----------')
        
        if i < d -1:
            for ii in range(1,i+1,1):
                for j in range(0,ii,1):
                    tr = j
                    col = ii-j-1

                    ind_now = ind_vine[j][ii-1-j]

                    if j == n-2:
                        ind_sup = ind_vine[j+1][0]
                    else:
                        ind_sup = ind_vine[j+1][i-1-j]

                    if j == 0:
                        tr1 = n-j
                        col1 = n-ii
                        ind1 = r_matrix[tr1,col1] #- 1
                        ind1 = np.where(nodes == ind1)
                        ind1 = ind1[0][0]

                        v2 = v[:,j,ind1][...,np.newaxis]
                    else:
                        parent1, inx1, inx2 = parent_var(j,ind_vine,ind_now)

                        if ind_vine[j-1][ind_now[0]][0] != parent1:
                            v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
                        else:
                            v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

                    v1 = v[:,j,ii][...,np.newaxis]

                    vv = np.concatenate((v1,v2),1)

                    parent, inx1, inx2 = parent_var(j+1,ind_vine,ind_sup)                
                    u_edge = {ind_now[0], ind_now[1]}

                    if (j == 0) | (binning == False):
                        if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                vv = np.concatenate((v2,v1),1)
                                v_flip[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
                            else:
                                v[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
                        else:
                            v[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
                    else:
                        if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                vv = np.concatenate((v2,v1),1)
                                flip_1 = True
                            else:
                                flip_1 = False
                        else:
                            flip_1 = False

                        ind1 = parent1
                        
                        if j == 1:
                            ind1 = np.where(nodes == ind1 +1)
                            ind1 = ind1[0][0]
                            bins = create_bins(v[:,j-1,ind1],n_bin)
                            val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                        else:
                            ind_par_now = ind_vine[j-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(j-1,ind_vine,ind_par_now)  

                            ind1 = ind1 + j - 1
                            if (ind_vine[j-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(v[:,j-1,ind1],n_bin)
                                val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                            else:
                                bins = create_bins(v_flip[:,j-1,ind1],n_bin)
                                val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1
                            
                        vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                        for bb in range(0,n_bin,1):
                            mask = np.where(val_to_bin == bb)
                            vv_bin = vv[mask[0],:]
                            
                            ### CDF FORCE UNIFORM
                            vv_bin_new = vv_bin
                            u_1, ex_u = mk_grid(tf.convert_to_tensor(50),vv_bin.dtype)
                            for zz in range(0,2,1):
                                vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],ex_u)
                            vv_bin = vv_bin_new
                            ###
                        
                            tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])

                            if flip_1 == True:
                                v_flip[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)
                            else:
                                v[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)
    
    u = np.reshape(v[:,0,:],np.shape(w))
    u1 = np.zeros(np.shape(u),u.dtype)
    c= 0
    for i in range(d-1,-1,-1):
        ind = r_matrix[i,i]-1
        u1[:,ind] = u[:,c]
        c += 1
    u = u1

    sample = np.zeros((cases,np.shape(u)[1]),w.dtype)
    for i in range(0,np.shape(u)[1],1):
        sample[:,i] = margininv(margin_vine[i],u[:,i])
    return sample, v, v_flip, tau_corr, tau_bins