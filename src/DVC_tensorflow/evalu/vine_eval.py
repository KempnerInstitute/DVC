import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from utils.prob_op import biv_norm
from optim.local_lik import loclik_batch,loclik_batch_eval
from optim.bandwidth import bandwidth_mul
from evalu.cop_eval import *
from utils.interpolation import nearestInterp2d, interp1d_np
from utils.prob_op import kernel_cdf,kernel_cdf_batch


################# EVALUATE PDF (UV-SPACE), CDF AND THETA ######################

def evaluate_fit(data_dict, grid_dict, par_dict):
    grid_u = grid_dict['grid_u']
    grid_s = grid_dict['grid_s']
    grid_x = grid_dict['grid_x']
    adu11,adu22 = grid_u.diff()
    
    data_s = data_dict['data_s']
    data_x = data_dict['data_x']
    theta = data_dict['theta']
    theta_flip = data_dict['theta_flip']
    
    copulas = par_dict['copulas']
    n_eval = par_dict['n_eval']
    batch_size = par_dict['batch']
    batch_size_cdf = par_dict['batch_cdf']
    tr = par_dict['tr']
    ind_edge_rel = par_dict['ind_edge_rel']
    flip_flag = par_dict['flip_flag']

    bw1 = np.zeros([2,n_eval],data_s.dtype)
    for i in range(0,n_eval,1):
        ii = ind_edge_rel[i]
        bw1[:,i] = tf.convert_to_tensor(copulas.opt_bw[:,ii])
    B = tf.reshape(bw1,[2,n_eval])
    
    ## Bivariate normal
    x1_s, x2_s = grid_s.ax1, grid_s.ax2
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM[...,tf.newaxis]
    NORM = tf.tile(NORM,[1, 1, n_eval])
    
    ker_grid_fin = loclik_batch_eval(B, data_x, grid_x, n_eval, batch_size)
    
    ker_grid_all = tf.transpose(tf.reshape(ker_grid_fin,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_eval]),perm=[1, 0, 2])
    
    ### The following was added to avoid points with 0 probability but to have it to be very low. Otherwise the log goes to inf
    ker_grid_all = ker_grid_all + 1e-15*NORM # Before it was 1e-30, do not know which one is better 
    
    pdf1 = eval_rs_cop(adu11, adu22, ker_grid_all, NORM, n_eval)  #eval_rs_p

    pd_grid_uv = pdf1/NORM
    
    cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_eval)

    for i in range(0,n_eval,1):
        ## Update theta
        ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s[:,:,i],grid_s.min,grid_s.max,cdf1[:,:,i],axis=-2)
        interp_cdf, mar_s1, mar_p1 = kernel_cdf(ccdf_data,ccdf_data,grid_u.ex)

        if flip_flag[i] == False:
            theta[:,tr+1,ind_edge_rel[i]] = interp_cdf
        else:
            theta_flip[:,tr+1,ind_edge_rel[i]] = interp_cdf
    return pd_grid_uv, cdf1, theta, theta_flip

################# EVALUATE PDF AND CCDF ON THE POINTS

def evaluate_points(points_s, batch_size, grid_s, cdf1, pd_grid_uv):
    pd_points = tf.TensorArray(points_s.dtype, size = batch_size)
    ccdf_points = tf.TensorArray(points_s.dtype, size = batch_size)
    batch_len = tf.shape(points_s)[0]/batch_size
    batch_len = tf.cast(batch_len,tf.int32)
    
    s_ax1 = grid_s.ax1
    s_ax2 = grid_s.ax2  

    for j in tf.range(0,batch_size,1):
        points_batch = points_s[batch_len*j:batch_len*(j+1),:]
        if tf.math.equal(j,batch_size-1):
            points_batch = points_s[batch_len*j:,:]
        pd_points1 = nearestInterp2d(points_batch, s_ax1, s_ax2, pd_grid_uv) 
        
        ccdf_points1 = tfp.math.batch_interp_regular_nd_grid(points_batch,grid_s.min,grid_s.max,cdf1,axis=-2)
        pd_points = pd_points.write(j, pd_points1)
        ccdf_points = ccdf_points.write(j, ccdf_points1)
    
    pd_points = pd_points.stack()
    ccdf_points = ccdf_points.stack()
    pd_points = tf.reshape(pd_points,[-1])
    ccdf_points = tf.reshape(ccdf_points,[-1])
    return pd_points, ccdf_points


#################### EVALUATE BINNING ###########################

def evaluate_fit_bin(data_dict, grid_dict, par_dict):
    grid_u = grid_dict['grid_u']
    grid_s = grid_dict['grid_s']
    grid_x = grid_dict['grid_x']
    adu11 = grid_u.diff1
    adu22 = grid_u.diff2
    
    data_s = data_dict['data_s']
    data_x = data_dict['data_x']
#     bb = data_dict['bin']
    
    bw = par_dict['bw']
    n_cop1 = tf.convert_to_tensor(par_dict['n_cop'])
    batch_size = tf.convert_to_tensor(par_dict['batch'])
    tr = par_dict['tr']
    ind_edge_rel = par_dict['ind_edge_rel']
    
    ## Bandwidth

    bw1 = np.empty([2,n_cop1],data_s.dtype)
    for i in range(0,n_cop1,1):
        ii = ind_edge_rel[i]
        bw1[:,i] = bw[:,ii]
    
    B = tf.reshape(tf.convert_to_tensor(bw1),[2,n_cop1])

    ## Bivariate normal
    x1_s, x2_s = grid_s.ax1, grid_s.ax2
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM[...,tf.newaxis]
    NORM = tf.tile(NORM,[1, 1, n_cop1])
    
    data_s = tf.convert_to_tensor(data_s)
    data_x = tf.convert_to_tensor(data_x)
    
    ker_grid_fin = loclik_batch_eval(B, data_x, grid_x, n_cop1, batch_size)
    
#     ker_grid_fin = loclik_batch(B, data_x, grid_x, n_cop1, batch_size)  #vine.data_x[:,:,0][...,tf.newaxis]
    ker_grid_all = tf.transpose(tf.reshape(ker_grid_fin,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop1]),perm=[1, 0, 2])
    
    ker_grid_all = ker_grid_all + 1e-10*NORM
    
    pdf1 = eval_rs_cop(adu11, adu22, ker_grid_all, NORM, n_cop1)  #eval_rs_p

    pd_grid_uv = pdf1/NORM

    cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_cop1)

    return pd_grid_uv, cdf1