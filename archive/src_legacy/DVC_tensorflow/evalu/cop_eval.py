import tensorflow as tf
from utils.tensor_op import *

############################## COPULA PDF #####################################

@tf.function(experimental_relax_shapes=True)
def eval1(adu11_col1, adu22_1, t2, n_cop):
    # Compute normalization

    I1 = tf.math.reduce_sum(adu22_1*t2,1)
    I2 = tf.math.reduce_sum(adu11_col1*t2,0)
    
    K5 = tf.TensorArray(t2.dtype,size=n_cop) #,element_shape=[tf.shape(t2)[0].eval(),tf.shape(t2)[0].eval()])
    for i in tf.range(0,n_cop,1):
        K1 = tf.tensordot(I1[:,i],I2[:,i],0)
        #print(K1)
        K5 = K5.write(i,K1)
    K5 = K5.stack()
    K5 = tf.transpose(K5, perm=[1,2,0])

    #t2 = t2/K5
    t2 = tf.math.multiply(t2,tf.math.reciprocal(K5))
    
    if tf.reduce_any(tf.math.logical_or(tf.math.is_nan(t2),tf.math.is_inf(t2))) == True:
        t2 = replace_nan_inf(tf.reshape(t2,[-1]))    
        t2 = tf.reshape(t2,[tf.shape(K5)[0],tf.shape(K5)[1],n_cop])             
    return t2

@tf.function(experimental_relax_shapes=True)
def eval_rs_p(adu11, adu22, ker_fit, NORM1, n_cop):
    # Copula normalization for MISE cost function with 100 cycle
    #adu11 = grid_u2.diff1
    #adu22 = grid_u2.diff2
    adu11_col = adu11[...,tf.newaxis]  #Make it a columns vector
    
    #t1 = ker_fit/NORM1     # Projecct on the u-v space
    t1 = tf.math.multiply(ker_fit,tf.math.reciprocal(NORM1))
    
    if tf.math.reduce_any(tf.math.reduce_max(t1) < 1e-6):  ######## THRESHOLD TOP UT ALL COPULA VALUES TO ONE
        t2 = tf.TensorArray(t1.dtype,size=tf.shape(t1)[2])
        for i in range(0,tf.shape(t1)[2],1):
            if tf.math.reduce_max(t1[:,:,i]) < 1e-6:   ######## THRESHOLD TOP UT ALL COPULA VALUES TO ONE
                upd = tf.ones(tf.shape(t1[:,:,i]),t1.dtype)
                t2 = t2.write(i,upd)
            else:
                t2 = t2.write(i,t1[:,:,i])
        t2 = t2.stack()
        t2 = tf.transpose(t2,perm=[1, 2, 0])
        t1 = t2
    
    adu22_1 = adu22[...,tf.newaxis]
    adu22_1 = tf.tile(adu22_1,[1, n_cop])

    adu11_col1 = tf.tile(adu11_col,[1, n_cop])
    adu11_col1 = tf.reshape(adu11_col1,[tf.shape(adu11)[0],1,n_cop])
    
#     t1 = t1*tf.constant(1e-5,t1.dtype)   #### HOUMAN
    
    for i in tf.range(0,50,1,dtype=tf.int32):   #50
        t1 = tf.reshape(eval1(adu11_col1, adu22_1, t1, n_cop),tf.shape(t1))
    
    adu11_col1 = tf.transpose(adu11_col1,perm=[1,0,2])
    II = tf.math.reduce_sum(adu11_col1*tf.math.reduce_sum(adu22_1*t1,1),1)
    t1 = t1/II
    t1 = t1 * NORM1     # Projecct back on the r-s space
    return t1

@tf.function(experimental_relax_shapes=True)
def eval_rs_cop(adu11, adu22, ker_fit, NORM1, n_cop):
    # Copula normalization for MISE cost function with 100 cycle
    #adu11 = grid_u2.diff1
    #adu22 = grid_u2.diff2
    adu11_col = adu11[...,tf.newaxis]  #Make it a columns vector
    
    #t1 = ker_fit/NORM1     # Projecct on the u-v space
    
    t1 = tf.math.multiply(ker_fit,tf.math.reciprocal(NORM1))
    
    if tf.math.reduce_any(tf.math.reduce_max(t1) < 1e-6):  ######## THRESHOLD TOP UT ALL COPULA VALUES TO ONE
        t2 = tf.TensorArray(t1.dtype,size=tf.shape(t1)[2])
        for i in range(0,tf.shape(t1)[2],1):
            if tf.math.reduce_max(t1[:,:,i]) < 1e-6:   ######## THRESHOLD TOP UT ALL COPULA VALUES TO ONE
                upd = tf.ones(tf.shape(t1[:,:,i]),t1.dtype)
                t2 = t2.write(i,upd)
            else:
                t2 = t2.write(i,t1[:,:,i])
        t2 = t2.stack()
        t2 = tf.transpose(t2,perm=[1, 2, 0])
        t1 = t2
    
    adu22_1 = adu22[...,tf.newaxis]
    adu22_1 = tf.tile(adu22_1,[1, n_cop])

    adu11_col1 = tf.tile(adu11_col,[1, n_cop])
    adu11_col1 = tf.reshape(adu11_col1,[tf.shape(adu11)[0],1,n_cop])
    
#     t1 = t1*tf.constant(1e-5,t1.dtype)  ### Houman

    for i in tf.range(0,500,1,dtype=tf.int32): 
        t1 = tf.reshape(eval1(adu11_col1, adu22_1, t1, n_cop),tf.shape(t1))
    
    adu11_col1 = tf.transpose(adu11_col1,perm=[1,0,2])
    II = tf.math.reduce_sum(adu11_col1*tf.math.reduce_sum(adu22_1*t1,1),1)
    t1 = t1/II
    t1 = t1 * NORM1     # Projecct back on the r-s space
    return t1


######################### COPULA CDF #####################################

@tf.function(experimental_relax_shapes=True)
def cdf_grid_fun(pd_grid_uv, ex_u, u1d, u2d, n_cop):
    # Compute the cdf on the grid
    knots = tf.shape(pd_grid_uv)[0]
    u2d = tf.reshape(u2d, [knots,1,1])
    u2d_tile = tf.tile(u2d,[1, knots, n_cop])
    pd_grid_uv_transp = tf.transpose(tf.reshape(pd_grid_uv,[knots, knots, n_cop]),perm=[1, 0, 2])
    integ = tf.math.cumsum(pd_grid_uv_transp*u2d_tile,0)
    norm_p = tf.math.reduce_sum(pd_grid_uv*u2d_tile,0)
    
    #REPLACE ZEROS
    ind_zeros = tf.where(tf.equal(norm_p,0))
    repl_zeros = tf.ones(tf.shape(ind_zeros)[0],u1d.dtype)
    norm_p = tf.tensor_scatter_nd_update(norm_p, ind_zeros, repl_zeros)
    
    cdf1 = tf.transpose(tf.reshape(integ/norm_p,[knots, knots, n_cop]),perm=[1, 0, 2])
    cdf1 = tf.reshape(cdf1, [-1]) 
    cdf1 = check_bound(cdf1,ex_u)
    cdf1 = tf.reshape(cdf1, [tf.shape(u1d)[0],tf.shape(u2d)[0],n_cop])
    return cdf1