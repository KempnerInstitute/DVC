import tensorflow as tf

from utils.tensor_op import check_bound3, replace_nan_inf
from optim.MISE import MISE_mul

############################# NADAM OPTIMIZATION #################################

#@tf.function(experimental_relax_shapes=True)
def fit_ban(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag, pos_trace, max_iter, convergence_tol, lr):
    eps = 1e-6
    
    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace = tf.ones([n_cop],bw.dtype)

    err = err_trace + 10*convergence_tol
    
    err = MISE_mul(pos_trace, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)

    m = tf.zeros(tf.shape(bw)[1],bw.dtype)
    v = tf.zeros(tf.shape(bw)[1],bw.dtype)
    m_hat = tf.zeros(tf.shape(bw)[1],bw.dtype)
    v_hat = tf.zeros(tf.shape(bw)[1],bw.dtype)
    beta_1 = tf.constant(0.9,bw.dtype)
    beta_2 = tf.constant(0.999,bw.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):
        err_trace = err
        err_trace = tf.reshape(err_trace, [tf.shape(bw)[1]])
        
        err = MISE_mul(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
        
        grad = (err - err_trace)/(a-pos_trace)
        grad = replace_nan_inf(grad)

        pos_trace = a
        iter1 = tf.cast(iter_err,bw.dtype)

        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, [tf.shape(bw)[1]])
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, [tf.shape(bw)[1]])
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)
        
        #### To be sure that bw does not go lower than 5e-3
        bw_n = tf.abs(a*bw)
        ind = tf.where(tf.math.less(bw_n[1,:],tf.constant(1e-2,bw.dtype)))  ##It was 5e-3 but too low
        if tf.shape(ind)[0] > 0:
            bu1 = tf.tile(tf.constant([5e-3],bw.dtype),[tf.shape(ind)[0]])
            gat = tf.gather_nd(bw[1,:],ind)
            aa1 = bu1/gat
            a = tf.tensor_scatter_nd_update(a,ind,aa1)
        
        a = a + diff
        
        a_new = check_bound3(a,tf.constant(4,bw.dtype),tf.constant(1e-2,bw.dtype))  ##It was 5e-3 but too low
        a = a_new
        
        a = tf.reshape(a, [tf.shape(bw)[1]])

        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)

@tf.function(experimental_relax_shapes=True)
def fit_banLL2(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag, pos_trace, max_iter, convergence_tol, lr):
    eps = 1e-6

    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace_x1y = tf.ones([n_cop],bw.dtype)
    err_trace_xy1 = tf.ones([n_cop],bw.dtype)
    
    err = err_trace_x1y + 10*convergence_tol
    
    m = tf.zeros(tf.shape(bw),bw.dtype)
    v = tf.zeros(tf.shape(bw),bw.dtype)
    m_hat = tf.zeros(tf.shape(bw),bw.dtype)
    v_hat = tf.zeros(tf.shape(bw),bw.dtype)
    beta_1 = tf.constant(0.9,bw.dtype)
    beta_2 = tf.constant(0.999,bw.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),
                              tf.math.logical_or(
                                  tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_x1y),convergence_tol)),
                                  tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_xy1),convergence_tol))
                              )
                             ):
        
        x1y = tf.concat([tf.gather(pos_trace,[0]),tf.gather(a,[1])],0)
        xy1 = tf.concat([tf.gather(a,[0]),tf.gather(pos_trace,[1])],0) 
        err_x1y = MISE_mul(x1y, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
        err_xy1 = MISE_mul(xy1, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)

        err_trace_x1y = err_x1y
        err_trace_xy1 = err_xy1
        err_trace_x1y = tf.reshape(err_trace_x1y, [tf.shape(bw)[1]])
        err_trace_xy1 = tf.reshape(err_trace_xy1, [tf.shape(bw)[1]])
        
        err = MISE_mul(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
        
        err = tf.reshape(err, [tf.shape(bw)[1]])
        
        grad_x1y = (err - err_trace_x1y)/(tf.gather(a,[0])-tf.gather(x1y,[0]))
        grad_xy1 = (err - err_trace_xy1)/(tf.gather(a,[1])-tf.gather(xy1,[1]))

        if tf.shape(tf.shape(grad_x1y)) == 1:
            grad_x1y[...,tf.newaxis]
            grad_xy1[...,tf.newaxis]
            
        grad = tf.concat([grad_x1y,grad_xy1],0)
        grad = replace_nan_inf(grad)

        pos_trace = a
        
        iter1 = tf.cast(iter_err,bw.dtype)
        
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(bw))
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(bw))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        
        a_new = check_bound3(a,tf.constant(2,bw.dtype),tf.constant(1e-2,bw.dtype))  ##It was 5e-3 but too low

        a = a_new
        a = tf.reshape(a, tf.shape(bw))

        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.logical_or(
                                  tf.math.reduce_any(tf.math.less(tf.abs(err-err_trace_x1y),convergence_tol)),
                                  tf.math.reduce_any(tf.math.less(tf.abs(err-err_trace_xy1),convergence_tol))
                              )