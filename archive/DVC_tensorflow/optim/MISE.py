import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from evalu.cop_eval import eval_rs_p
from optim.local_lik import loclik_batch

############################## MISE COST FUNCTION ################################

#@tf.function(experimental_relax_shapes=True)
def MISE_mul(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, all_x, data_x_train, data_s_test, n_cop, batch_size, NORM1, norm_flag):
    if tf.math.equal(tf.shape(tf.shape(all_x)),2):
        all_x = all_x[...,tf.newaxis]
    
    bw1 = tf.abs(a*bw)
    n_splits = tf.shape(data_x_train)[3]  

    if tf.math.equal(tf.shape(tf.shape(grid_x)),2):
        grid_x = grid_x[...,tf.newaxis]

    ker_grid_all = loclik_batch(bw1, all_x, grid_x, n_cop, batch_size) #data_x
    if norm_flag == True:
        ker_grid_all = tf.transpose(tf.reshape(ker_grid_all,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop]),perm=[1, 0, 2])
        pd_grid = eval_rs_p(adu11, adu22, ker_grid_all, NORM1, n_cop)
    else:
        pd_grid = tf.zeros([tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop],bw.dtype)
        
    kkk_fin = tf.TensorArray(grid_x.dtype,size=n_splits)
    for k in tf.range(0,n_splits,1,tf.int32):
        ker_grid_fin = loclik_batch(bw1, data_x_train[:,:,:,k], grid_x, n_cop, batch_size)
        pd_grid1 = tf.transpose(tf.reshape(ker_grid_fin,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop]),perm=[1, 0, 2])
        if norm_flag == True:
            pd_grid1 = eval_rs_p(adu11, adu22, pd_grid1, NORM1, n_cop)
            
        interp_data = tf.TensorArray(data_x_train.dtype, size=n_cop)
        for kk in tf.range(0,n_cop,1,tf.int32):
            interp_data1 = tfp.math.batch_interp_regular_nd_grid(data_s_test[:,:,kk,k], min_s, max_s, pd_grid1[:,:,kk], axis=-2)
            interp_data = interp_data.write(kk,interp_data1)
        interp_data = interp_data.stack()
        interp_data = tf.transpose(interp_data)
        if norm_flag == True:
            interp_data = interp_data / tf.math.reduce_sum(pd_grid * step_s,[0,1])
        else:
            interp_data = interp_data / tf.math.reduce_sum(ker_grid_fin * step_s,0)
        kkk_fin = kkk_fin.write(k,interp_data)
    kkk_fin = kkk_fin.stack()
    kkk_fin = tf.reshape(kkk_fin,[tf.shape(kkk_fin)[0]*tf.shape(kkk_fin)[1],n_cop]) #kkk_fin

    if norm_flag == True:
        err = tf.math.reduce_sum(pd_grid**2 * step_s,[0,1]) - 2 *tf.math.reduce_mean(kkk_fin,0)

    else:
        pd_grid = ker_grid_all / (tf.math.reduce_sum(ker_grid_all * step_s,0))
        err = tf.math.reduce_sum(pd_grid**2 * step_s,0) - 2 *tf.math.reduce_mean(kkk_fin,0)

    ### Put to err +- 0.001*err if out of bounds
    ind_err = tf.where(tf.math.logical_or(tf.math.less_equal(a,1e-4),tf.math.greater_equal(a,2)))
    ind_err = tf.cast(ind_err,tf.int32)
    new_err = tf.TensorArray(grid_x.dtype,size=tf.shape(err)[0])
    for ind in tf.range(0,tf.shape(err)[0],1,tf.int32):
        if tf.math.reduce_any(tf.math.equal(ind,ind_err)):
            if tf.math.sign(err[ind]) > 0:
                new_err_tmp = err[ind][...,tf.newaxis]+err[ind][...,tf.newaxis]*0.001 #tf.constant([0.1],grid_x.dtype)
            else:
                new_err_tmp = err[ind][...,tf.newaxis]-err[ind][...,tf.newaxis]*0.001
        else:
            new_err_tmp = err[ind][...,tf.newaxis]
        new_err = new_err.write(ind,new_err_tmp)
    new_err = tf.squeeze(new_err.stack())
    err = new_err
    if tf.shape(tf.shape(err)) == 0:
        err = err[...,tf.newaxis]
    return err
