import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np

from utils.prob_op import kernel_cdf
from evalu.vine_eval import evaluate_points
from utils.interpolation import interp1d_np
from utils.dataset_op import create_bins, check_bins
from vine_tree.tree_op import parent_var
from pre_proc.preparation import prep_copula
from pre_proc.transformation import Transform
from param.cond_copula import *

#################################### SAMPLING FROM NON-PARAMETRIC COPULA #######################################

############################ INVERSE NON-PARAMETRIC COPULA CDF ####################################

@tf.function(experimental_relax_shapes=True)
def kerncopccdfinv(w, cdf_grid, u1, u2):  
    # Function to sample from the copula
    #u1 = grid_u2.ax1
    #u2 = grid_u2.ax2
    len_w = tf.shape(w)[0]
    len_ax = tf.shape(u1)[0]
    u1_tile = tf.tile(u1,[len_w])
    u1_tile = tf.transpose(tf.reshape(u1_tile,[len_w,len_ax])) #[100,20000]

    w0_tile = tf.tile(w[:,0],[len_ax])
    w0_tile = tf.reshape(w0_tile,[len_ax,len_w]) #[100,20000]
    
    m1 = tf.math.argmin(tf.abs(u1_tile-w0_tile),0)
    
    m1 = m1[...,tf.newaxis]
    g =  tf.gather_nd(cdf_grid, m1)
    g = tf.transpose(g)
    
    len_g = tf.shape(g)[0]

    w1_tile = tf.tile(w[:,1],[len_ax])
    w1_tile = tf.reshape(w1_tile,[len_ax,len_w]) #[100,20000]

    propro = g-w1_tile #tf.abs  tf.math.negative

    mask1 = tf.greater(propro,0)
    mask1 = tf.cast(mask1,dtype=tf.int32)
    ind = tf.math.argmax(mask1)
    U2 = tf.gather(u2,ind)
    return U2

################# TRY CON FLAG

def vine_copula_sample(vine,cases):
    d = len(vine.r_matrix)
    #print(d)
    n = len(vine.r_matrix) -1
    depth = vine.vine_depth   ### Should I use this and how??

    # w = np.random.uniform(1e-3,0.999,(cases,d))   #0,1,(cases,d))

    w = np.random.uniform(0,1,(cases,d))
    mag = np.max(vine.grid_u.ex)#-1e-7
    mig = np.min(vine.grid_u.ex)#+1e-7

    w = (mag-mig)*(w-np.min(w))/(np.max(w)-np.min(w))+mig
    w = w.astype(vine.data_u.dtype)

    v = np.zeros([cases,d,d],w.dtype)
    v_flip = np.zeros([cases,d,d],w.dtype)
    v[:,0,0] = w[:,0]
    u1 = vine.grid_u.ax1
    u2 = vine.grid_u.ax2

    ## Initialize the flag for flipping
    flip_flag1 = []
    for ii in range(1,d-1,1):
        flip_flag2 = []

        for j in range(0,ii,1):
            flip_flag2.append([])
        flip_flag1.append(flip_flag2)

    ## Start from first column [1,1]
    for i in range(1,d,1):  #d
        v[:,i,i] = w[:,i]
        
        c = 0
        for k in range(i-1,-1,-1):
            tr = k
            col = i-k-1
            ind_now = vine.ind_vine[k][c]

            ind_array = np.array(vine.ind_edge_rel[tr])
            ind_col = np.where(ind_array == col)
            col = ind_col[0][0]

            if k == 0:
                tr1 = n-k
                col1 = n-i
                ind1 = vine.r_matrix[tr1,col1] #- 1
                ind1 = np.where(vine.nodes == ind1)
                ind1 = ind1[0][0]

                v2 = v[:,k,ind1][...,np.newaxis]
                v1 = v[:,k+1,i][...,np.newaxis]
                vv = tf.convert_to_tensor(np.concatenate((v2,v1),1))

                cdf_grid = vine.copulas[tr].cdf[:,:,col]
                v[:,k,i] = kerncopccdfinv(vv, cdf_grid, u1,u2)

            else:

                parent, inx1, inx2 = parent_var(k,vine.ind_vine,ind_now)

                if vine.ind_vine[k-1][ind_now[0]][0] != parent: 
                    v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
                else:
                    v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

                v1 = v[:,k+1,i][...,np.newaxis]

                ## CHANGED
                vv = np.concatenate((v2,v1),1)

            if tr > depth:
                ### Independent
                col = i-k-1

                vv = np.flip(vv,axis=1)

                if vine.binning == False:
                    v[:,k,i] = copulainvccdf(vine.copulas[tr][col],vv)

                else:
                    
                    ind1 = parent
                    
                    if k == 1:
                        ind1 = np.where(vine.nodes == ind1 +1)
                        ind1 = ind1[0][0]
                        bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                        val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                    else:
                        ind_par_now = vine.ind_vine[k-1][ind_now[1]]
                        parent22, inx1, inx2 = parent_var(k-1,vine.ind_vine,ind_par_now)  

                        ind1 = ind1 + k - 1
                        if (vine.ind_vine[k-2][ind_par_now[0]][0] == parent22):
                            bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                        else:
                            bins = create_bins(v_flip[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1
                        
                    vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                    for bb in range(0,vine.n_bin,1):
                        mask = np.where(val_to_bin == bb)
                        vv_bin = vv[mask[0],:]
                        
                        ### CDF FORCE UNIFORM
                        vv_bin_new = vv_bin
                        for zz in range(0,2,1):
                            vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],vine.grid_u.ex)
                        vv_bin = vv_bin_new
                        ###
                        
                        v[mask[0],k,i] = copulainvccdf(vine.copulas[tr][col][bb],vv_bin)
            else:

                if vine.binning == False:
                    cdf_grid = vine.copulas[tr].cdf[:,:,col]
                    vv = tf.convert_to_tensor(vv)
                    v[:,k,i] = kerncopccdfinv(vv, cdf_grid, u1,u2)
                    #if k > 9:
                        #print(v[:20,k,i])
                else:
                    
                    ind1 = parent
                    
                    if k == 1:
                        ind1 = np.where(vine.nodes == ind1 +1)
                        ind1 = ind1[0][0]
                        bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                        val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                    else:
                        ind_par_now = vine.ind_vine[k-1][ind_now[1]]
                        parent22, inx1, inx2 = parent_var(k-1,vine.ind_vine,ind_par_now)  

                        ind1 = ind1 + k - 1
                        if (vine.ind_vine[k-2][ind_par_now[0]][0] == parent22):
                            bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                            val_to_bin = check_bins(v[:,k-1,ind1],bins)
                        else:
                            bins = create_bins(v_flip[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1
                            val_to_bin = check_bins(v_flip[:,k-1,ind1],bins)

                    vv_all = np.zeros(np.shape(vv)[0],w.dtype)
                    for bb in range(0,vine.n_bin,1):
                        cdf_grid = vine.copulas[tr].cdf[:,:,col,bb]
                        mask = np.where(val_to_bin == bb)
                        vv_bin = vv[mask[0],:]
                        
                        ### CDF FORCE UNIFORM
                        vv_bin_new = vv_bin
                        for zz in range(0,2,1):
                            vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],vine.grid_u.ex)
                        vv_bin = vv_bin_new
                        ###
                        vv_bin = tf.convert_to_tensor(vv[mask[0],:])
                        
                        v[mask[0],k,i] = kerncopccdfinv(vv_bin, cdf_grid, u1,u2)
#                         corr = stats.pearsonr(vv_bin[:,1],v[mask[0],k,i])
#                         print('Corr value  UV space: ',corr[0])
            c += 1

        if i < d-1: 
            cc1 = 0
            for ii in range(1,i+1,1):

                cc2 = 0
                for j in range(0,ii,1):
                    tr = j
                    col = ii-j-1

                    ind_now = vine.ind_vine[j][ii-1-j]

                    if j == n-2:
                        ind_sup = vine.ind_vine[j+1][0]
                    else:
                        ind_sup = vine.ind_vine[j+1][i-1-j]

                    flip_flag = False

                    parent, inx1, inx2 = parent_var(j+1,vine.ind_vine,ind_sup)                
                    u_edge = {ind_now[0], ind_now[1]}
                    if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                flip_flag = True

                    flag = False
                    if i >1:
                        for el in flip_flag1[cc1][cc2]:
                            if el == flip_flag:
                                flag = True
                        if flag == False:
                            flip_flag1[cc1][cc2].append(flip_flag)
                    else:
                        flip_flag1[cc1][cc2].append(flip_flag)

                    if flag == False:

                        if j == 0:
                            tr1 = n-j
                            col1 = n-ii
                            ind1 = vine.r_matrix[tr1,col1] #- 1
                            ind1 = np.where(vine.nodes == ind1)
                            ind1 = ind1[0][0]

                            v2 = v[:,j,ind1][...,np.newaxis]
                        else:
                            parent1, inx1, inx2 = parent_var(j,vine.ind_vine,ind_now)

#                             if ind_now[0] != parent1:
                            if vine.ind_vine[j-1][ind_now[0]][0] != parent1:
                                v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
                            else:
                                v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

                        v1 = v[:,j,ii][...,np.newaxis]
                        
                        if flip_flag == False:
                            data_u = np.concatenate((v2,v1),1)
                        else:
                            data_u = np.concatenate((v1,v2),1)

                        if j > depth:
                            ## Independent
                            col = ii-j-1
                            # vv = np.concatenate((v1,v2),1)
                            vv = np.flip(data_u,axis=1)

                            parent, inx1, inx2 = parent_var(j+1,vine.ind_vine,ind_sup)                
                            u_edge = {ind_now[0], ind_now[1]}
                            

                            if (j == 0) | (vine.binning == False):
                                
                                if flip_flag == False:
                                    v[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
                                else:
                                    v_flip[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)

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
                                    ind1 = np.where(vine.nodes == ind1 +1)
                                    ind1 = ind1[0][0]
                                    bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                                    val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                                else:
                                    ind_par_now = vine.ind_vine[j-1][ind_now[1]]
                                    parent22, inx1, inx2 = parent_var(j-1,vine.ind_vine,ind_par_now)  

                                    ind1 = ind1 + j - 1
                                    if (vine.ind_vine[j-2][ind_par_now[0]][0] == parent22):
                                        bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                                        val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                                    else:
                                        bins = create_bins(v_flip[:,j-1,ind1],vine.n_bin)
                                        val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1

                                vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                                for bb in range(0,vine.n_bin,1):
                                    mask = np.where(val_to_bin == bb)
                                    vv_bin = vv[mask[0],:]

                                    ### CDF FORCE UNIFORM
                                    vv_bin_new = vv_bin
                                    for zz in range(0,2,1):
                                        vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],vine.grid_u.ex)
                                    vv_bin = vv_bin_new
                                    ###

                                    tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])

                                    if flip_1 == True:
                                        v_flip[mask[0],j+1,ii] = copulaccdf(vine.copulas[tr][col][bb],vv_bin)
                                    else:
                                        v[mask[0],j+1,ii] = copulaccdf(vine.copulas[tr][col][bb],vv_bin)

                        else:
                            
                            ind_array = np.array(vine.ind_edge_rel[tr])
                            ind_col = np.where(ind_array == col)
                            ind_fin = ind_col[0][0]
                #             print('ind_fin',col)
                            if (vine.ind_edge_rel[tr][ind_fin+1] == col) & (vine.flip_flag[tr][ind_fin+1] == flip_flag):
                                col = ind_fin + 1
                            else:
                                col = ind_fin

                            if (j==0) | (vine.binning == False):
                                
                                data_u = data_u[...,np.newaxis]
                                trans = Transform(1)

                                ## Transform data
                                data_s = trans.forward_u(data_u)
                                data_x = trans.forward_s(data_s)

                                batch_size = tf.constant(2,tf.int32)
                                pd_grid_uv = vine.copulas[tr].pd_grid_uv[:,:,col]
                                cdf1 = vine.copulas[tr].cdf[:,:,col]

                                ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s[:,:,0],vine.grid_s.min,vine.grid_s.max,cdf1,axis=-2)

                                pd_points, ccdf_points = evaluate_points(data_s[:,:,0], batch_size, vine.grid_s, cdf1, pd_grid_uv)    

                                # Update Fp
                                interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_points, vine.grid_u.ex)
                                if flip_flag == False:
                                    v[:,j+1,ii] = interp_cdf_poi
                                else:
                                    v_flip[:,j+1,ii] = interp_cdf_poi

                            else: #binning
                                
                                ind1 = parent1
                                
                                if j == 1:
                                    ind1 = np.where(vine.nodes == ind1 +1)
                                    ind1 = ind1[0][0]
                                    bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                                    val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                                else:
                                    ind_par_now = vine.ind_vine[j-1][ind_now[1]]
                                    parent22, inx1, inx2 = parent_var(j-1,vine.ind_vine,ind_par_now)  

                                    ind1 = ind1 + j - 1
                                    if (vine.ind_vine[j-2][ind_par_now[0]][0] == parent22):
                                        bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                                        val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                                        val_to_bin = check_bins(v[:,j-1,ind1],bins)
                                    else:
                                        bins = create_bins(v_flip[:,j-1,ind1],vine.n_bin)
                                        val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1
                                        val_to_bin = check_bins(v_flip[:,j-1,ind1],bins)
                                
                                vv_all = np.zeros(np.shape(vv)[0],w.dtype)
                                for bb in range(0,vine.n_bin,1):
                                    mask = np.where(val_to_bin == bb)

                                    batch_size = tf.constant(1,tf.int32)
                                    pd_grid_uv = vine.copulas[tr].pd_grid_uv[:,:,col,bb]
                                    cdf1 = vine.copulas[tr].cdf[:,:,col,bb]
                                    
                                    data_u_bin = data_u[mask[0],:]
                                    
                                    ### CDF FORCE UNIFORM
                                    vv_bin_new = data_u_bin
                                    for zz in range(0,2,1):
                                        vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],data_u_bin[:,zz],vine.grid_u.ex)
                                    data_u_bin = vv_bin_new[...,np.newaxis]
                                    ###
                                    
                                    trans = Transform(1)

                                    ## Transform data
                                    data_s_bin = trans.forward_u(data_u_bin)
                                    data_x_bin = trans.forward_s(data_s_bin)
                                    

                                    ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin[:,:,0],vine.grid_s.min,vine.grid_s.max,cdf1,axis=-2)

                                    pd_points, ccdf_points = evaluate_points(data_s_bin[:,:,0], batch_size, vine.grid_s, cdf1, pd_grid_uv)    

                                    # Update Fp
                                    interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_points, vine.grid_u.ex)
                                    if flip_flag == False:
                                        v[mask[0],j+1,ii] = interp_cdf_poi
                                    else:
                                        v_flip[mask[0],j+1,ii] = interp_cdf_poi
                    cc2 += 1
                cc1 += 1

    u = np.reshape(v[:,0,:],np.shape(w))
    u1 = np.zeros(np.shape(u),u.dtype)

    u_ax = vine.grid_u.ax1
    u_ax1 = np.tile(u_ax[...,np.newaxis],[1,np.shape(u)[0]]).T
    gr_diff = vine.grid_u.diff1

    c= 0
    for i in range(d-1,-1,-1):
        ind = vine.r_matrix[i,i]-1
        u1[:,ind] = u[:,c]

        ###### COMMENT WHEN COMPUTING INFORMATION OTHERWISE ALL WRONG
        u_p1 = u[:,c][...,np.newaxis]
        u_upd = np.tile(u_p1,[1,np.shape(u_ax)[0]])
        u_diff = np.abs(u_ax1-u_upd)
        ind1 = np.argmin(u_diff,1)
        diff_val = np.take(gr_diff,ind1)
        u1[:,ind] = u[:,c] + diff_val*np.random.uniform(0.,1.,np.shape(diff_val)[0])
        
        c += 1

    u = u1

    #print(u)
    sample1 = np.zeros([cases,d],w.dtype)
    #print(sample1.shape)
    #print(d)
    sample_pdf = np.zeros([len(vine.Mar_G[i][0]),d],w.dtype)
    sample_pds = np.zeros([len(vine.Mar_G[i][0]),d],w.dtype)
    u = tf.cast(u,w.dtype)
    for i in tf.range(0,d,1,tf.int32):
        #print(sample_pdf.shape)
        #print(vine.Mar_G[i][1].shape)
        mar_s1 = tf.cast(vine.Mar_G[i][0],w.dtype)
        mar_p1 = tf.cast(vine.Mar_G[i][1],w.dtype)
        sample1_pro = tfp.math.interp_regular_1d_grid(u[:,i],x_ref_min=tf.math.reduce_min(mar_p1),x_ref_max=tf.math.reduce_max(mar_p1),y_ref=mar_s1)
        #print(tf.math.reduce_min(mar_s1),tf.math.reduce_max(mar_s1))     
        sample1[:,i] = prep_copula(sample1_pro,0).numpy()
        sample_pdf[:,i]=mar_p1.numpy() 
        sample_pds[:,i]=mar_s1.numpy() 
        
    
    #print(type(sample1))
    
    return sample1, u, sample_pdf, sample_pds

###################### SAMPLING FROM PARAMETRIC VINE COPULAS ################################

def vine_cop_par_sample(vine, cases):
    d = len(vine.r_matrix)
    n = len(vine.r_matrix) -1

    w = np.random.uniform(0,1,(cases,d))

    mag = np.max(vine.grid_u.ex)-1e-5  ### 1e-5 to constraints the boundary of the cdf
    mig = np.min(vine.grid_u.ex)+1e-5

    w = (mag-mig)*(w-np.min(w))/(np.max(w)-np.min(w))+mig

    w = w.astype(np.float32)

    v = np.zeros([cases,d,d],w.dtype)
    v_flip = np.zeros([cases,d,d],w.dtype)
    v[:,0,0] = w[:,0]
    
    for i in range(1,d,1):
        v[:,i,i] = w[:,i]

        c = 0
        for k in range(i-1,-1,-1):
            tr = k
            col = i-k-1
            ind_now = vine.ind_vine[k][c]

            if k == 0:
                tr1 = n-k
                col1 = n-i
                ind1 = vine.r_matrix[tr1,col1] #- 1
                ind1 = np.where(vine.nodes == ind1)
                ind1 = ind1[0][0]

                v2 = v[:,k,ind1][...,np.newaxis]
                v1 = v[:,k+1,i][...,np.newaxis]
                vv = np.concatenate((v1,v2),1)
                v[:,k,i] = copulainvccdf(vine.copulas[tr][col],vv)
            else:

                parent, inx1, inx2 = parent_var(k,vine.ind_vine,ind_now)

                if vine.ind_vine[k-1][ind_now[0]][0] != parent: 
                    v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
                else:
                    v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

                v1 = v[:,k+1,i][...,np.newaxis]
                vv = np.concatenate((v1,v2),1)
                if vine.binning == False:
                    v[:,k,i] = copulainvccdf(vine.copulas[tr][col],vv)
                else:
                    
                    ind1 = parent
                    
                    if k == 1:
                        ind1 = np.where(vine.nodes == ind1 +1)
                        ind1 = ind1[0][0]
                        bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                        val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                    else:
                        ind_par_now = vine.ind_vine[k-1][ind_now[1]]
                        parent22, inx1, inx2 = parent_var(k-1,vine.ind_vine,ind_par_now)  

                        ind1 = ind1 + k - 1
                        if (vine.ind_vine[k-2][ind_par_now[0]][0] == parent22):
                            bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                        else:
                            bins = create_bins(v_flip[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1
                        
                    vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                    for bb in range(0,vine.n_bin,1):
                        mask = np.where(val_to_bin == bb)
                        vv_bin = vv[mask[0],:]
                        
                        ### CDF FORCE UNIFORM
                        vv_bin_new = vv_bin
                        for zz in range(0,2,1):
                            vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],vine.grid_u.ex)
                        vv_bin = vv_bin_new
                        ###
                        
                        v[mask[0],k,i] = copulainvccdf(vine.copulas[tr][col][bb],vv_bin)
                        
            c += 1
            
        if i < d -1:
            for ii in range(1,i+1,1):
                for j in range(0,ii,1):
                    tr = j
                    col = ii-j-1

                    ind_now = vine.ind_vine[j][ii-1-j]
                    
                    if j == n-2:
                        ind_sup = vine.ind_vine[j+1][0]
                    else:
                        ind_sup = vine.ind_vine[j+1][i-1-j]
                    
                    if j == 0:
                        tr1 = n-j
                        col1 = n-ii
                        ind1 = vine.r_matrix[tr1,col1] #- 1
                        ind1 = np.where(vine.nodes == ind1)
                        ind1 = ind1[0][0]

                        v2 = v[:,j,ind1][...,np.newaxis]
                    else:
                        parent1, inx1, inx2 = parent_var(j,vine.ind_vine,ind_now)
                        
                        if vine.ind_vine[j-1][ind_now[0]][0] != parent1:
                            v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
                        else:
                            v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

                    v1 = v[:,j,ii][...,np.newaxis]

                    vv = np.concatenate((v1,v2),1)

                    parent, inx1, inx2 = parent_var(j+1,vine.ind_vine,ind_sup)                
                    u_edge = {ind_now[0], ind_now[1]}

                    if (j == 0) | (vine.binning == False):
                        if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                vv = np.concatenate((v2,v1),1)
                                v_flip[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
                            else:
                                v[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
                        else:
                            v[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
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
                            ind1 = np.where(vine.nodes == ind1 +1)
                            ind1 = ind1[0][0]
                            bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                        else:
                            ind_par_now = vine.ind_vine[j-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(j-1,vine.ind_vine,ind_par_now)  

                            ind1 = ind1 + j - 1
                            if (vine.ind_vine[j-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                                val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                            else:
                                bins = create_bins(v_flip[:,j-1,ind1],vine.n_bin)
                                val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1

                        vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                        for bb in range(0,vine.n_bin,1):
                            mask = np.where(val_to_bin == bb)
                            vv_bin = vv[mask[0],:]

                            ### CDF FORCE UNIFORM
                            vv_bin_new = vv_bin
                            for zz in range(0,2,1):
                                vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],vine.grid_u.ex)
                            vv_bin = vv_bin_new
                            ###

                            tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])

                            if flip_1 == True:
                                v_flip[mask[0],j+1,ii] = copulaccdf(vine.copulas[tr][col][bb],vv_bin)
                            else:
                                v[mask[0],j+1,ii] = copulaccdf(vine.copulas[tr][col][bb],vv_bin)
                    
    u = np.reshape(v[:,0,:],np.shape(w))
    u1 = np.zeros(np.shape(u),u.dtype)
    
    u_ax = vine.grid_u.ax1
    u_ax1 = np.tile(u_ax[...,np.newaxis],[1,np.shape(u)[0]]).T
    gr_diff = vine.grid_u.diff1
    
    c= 0
    for i in range(d-1,-1,-1):
        ind = vine.r_matrix[i,i]-1
        u1[:,ind] = u[:,c]
        
        c += 1
    u = u1

    sample1 = np.zeros([cases,d],w.dtype)
    for i in tf.range(0,d,1,tf.int32):
        mar_s1 = vine.Mar_G[i][0]
        mar_p1 = vine.Mar_G[i][1]
        sample1_pro = tfp.math.interp_regular_1d_grid(u[:,i],tf.math.reduce_min(mar_p1),tf.math.reduce_max(mar_p1),mar_s1)
        sample1[:,i] = prep_copula(sample1_pro,0).numpy()
    
    return sample1