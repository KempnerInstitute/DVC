import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from utils.tensor_op import check_bound3, replace_nan_inf
from param.margin_cost import *

################################# GAUSSIAN FITTING ###############################################

@tf.function(experimental_relax_shapes=True)
def fit_gaussian(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    eps = tf.constant(1e-6,a.dtype)
    
    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace = tf.ones([n_cop],u.dtype)
    err = err_trace + 10*convergence_tol
    
    err = gaussian_cost(u, pos_trace)
    
    m = tf.zeros(tf.shape(a),u.dtype)
    v = tf.zeros(tf.shape(a),u.dtype)
    m_hat = tf.zeros(tf.shape(a),u.dtype)
    v_hat = tf.zeros(tf.shape(a),u.dtype)
    beta_1 = tf.constant(0.9,u.dtype)
    beta_2 = tf.constant(0.999,u.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):
        err_trace = err
        err_trace = tf.reshape(err_trace, [n_cop])
        
        err = gaussian_cost(u, a)
        
        grad = (err - err_trace)/(a-pos_trace)
        grad = replace_nan_inf(grad)

        pos_trace = a
        

        iter1 = tf.cast(iter_err,u.dtype)
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(a))
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(a))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        a_new = check_bound3(a,tf.constant(1-1e-3,u.dtype),tf.constant(0+1e-3,u.dtype))
        a = a_new
        a = tf.reshape(a, tf.shape(a))
        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)


################################# STUDENT FITTING ###############################################

# @tf.function(experimental_relax_shapes=True)
def fit_student(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    eps = 1e-6

    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace_x1y = tf.ones([n_cop],u.dtype)
    err_trace_xy1 = tf.ones([n_cop],u.dtype)
    
    err = err_trace_x1y + 10*convergence_tol
    
    m = tf.zeros(tf.shape(a),u.dtype)

    v = tf.zeros(tf.shape(a),u.dtype)
    m_hat = tf.zeros(tf.shape(a),u.dtype)
    v_hat = tf.zeros(tf.shape(a),u.dtype)
    beta_1 = tf.constant(0.9,u.dtype)
    beta_2 = tf.constant(0.999,u.dtype)
    

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),
                              tf.math.logical_or(
                                  tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_x1y),convergence_tol)),
                                  tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_xy1),convergence_tol))
                              )
                             ):
    
        x1y = tf.concat([pos_trace[:,0][...,tf.newaxis],a[:,1][...,tf.newaxis]],1)
        xy1 = tf.concat([a[:,0][...,tf.newaxis],pos_trace[:,1][...,tf.newaxis]],1)
        
        err_x1y = student_cost(u, x1y)
        err_xy1 = student_cost(u, xy1)

        err_trace_x1y = err_x1y
        err_trace_xy1 = err_xy1
        err_trace_x1y = tf.reshape(err_trace_x1y, [n_cop])
        err_trace_xy1 = tf.reshape(err_trace_xy1, [n_cop])
        
        err = student_cost(u, a)
        
        err = tf.reshape(err, [n_cop])
        
        grad_x1y = (err - err_trace_x1y)/(a[:,0]-x1y[:,0])
        grad_xy1 = (err - err_trace_xy1)/(a[:,1]-xy1[:,1])

        if tf.shape(tf.shape(grad_x1y)) == 1:
            grad_x1y[...,tf.newaxis]
            grad_xy1[...,tf.newaxis]
            
        grad = tf.concat([grad_x1y[...,tf.newaxis],grad_xy1[...,tf.newaxis]],1)
        grad = replace_nan_inf(grad)

        pos_trace = a
        
        iter1 = tf.cast(iter_err,u.dtype)
        
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(a))

        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(a))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        
        a_new1 = check_bound3(a[:,0][...,tf.newaxis],tf.constant(1,u.dtype),tf.constant(-1,u.dtype))
        a_new2 = check_bound3(a[:,1][...,tf.newaxis],tf.constant(1000,u.dtype),tf.constant(1e-3,u.dtype))
        a_new1 = tf.reshape(a_new1,[n_cop,1])
        a_new2 = tf.reshape(a_new2,[n_cop,1])
        a_new = tf.concat([a_new1,a_new2],1)
        a = a_new
        a = tf.reshape(a, [n_cop,2]) 
        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
        
    return a, err, iter_err, tf.math.logical_or(
                                tf.equal( tf.shape(tf.where(tf.equal(tf.math.less(tf.abs(err-err_trace_x1y),convergence_tol),True)))[0] , n_cop),
                                tf.equal( tf.shape(tf.where(tf.equal(tf.math.less(tf.abs(err-err_trace_xy1),convergence_tol),True)))[0] , n_cop)
                                                )

################################# CLAYTON FITTING ###############################################

@tf.function(experimental_relax_shapes=True)
def fit_clayton(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    eps = 1e-6
    
    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace = tf.ones([n_cop],u.dtype)
    
    err = err_trace + 10*convergence_tol
    
    err = clayton_cost(u, pos_trace)
    err = tf.reshape(err, [n_cop])
    
    m = tf.zeros(tf.shape(a),u.dtype)
    v = tf.zeros(tf.shape(a),u.dtype)
    m_hat = tf.zeros(tf.shape(a),u.dtype)
    v_hat = tf.zeros(tf.shape(a),u.dtype)
    beta_1 = tf.constant(0.9,u.dtype)
    beta_2 = tf.constant(0.999,u.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):

        err_trace = err
        err_trace = tf.reshape(err_trace, [n_cop])
        
        err = clayton_cost(u, a)
        
        err = tf.reshape(err, [n_cop])
        
        grad = (err - err_trace)/(a-pos_trace)
        grad = replace_nan_inf(grad)
        
        pos_trace = a
        
        iter1 = tf.cast(iter_err,u.dtype)
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(a))
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(a))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        a_new = check_bound3(a,tf.constant(20,u.dtype),tf.constant(1e-1,u.dtype))
        a = a_new
        a = tf.reshape(a, tf.shape(a))

        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)

################################# CLAYTON ROT 90 FITTING ###############################################

@tf.function(experimental_relax_shapes=True)
def fit_claytonrot90(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    eps = 1e-6
    
    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace = tf.ones([n_cop],u.dtype)
    
    err = err_trace + 10*convergence_tol
    
    err = claytonrot90_cost(u, pos_trace)
    err = tf.reshape(err, [n_cop])
    
    m = tf.zeros(tf.shape(a),u.dtype)
    v = tf.zeros(tf.shape(a),u.dtype)
    m_hat = tf.zeros(tf.shape(a),u.dtype)
    v_hat = tf.zeros(tf.shape(a),u.dtype)
    beta_1 = tf.constant(0.9,u.dtype)
    beta_2 = tf.constant(0.999,u.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):

        err_trace = err
        err_trace = tf.reshape(err_trace, [n_cop])
        
        err = claytonrot90_cost(u, a)
        
        err = tf.reshape(err, [n_cop])
        
        grad = (err - err_trace)/(a-pos_trace)
        grad = replace_nan_inf(grad)
        
        pos_trace = a
        
        iter1 = tf.cast(iter_err,u.dtype)
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(a))
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(a))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        a_new = check_bound3(a,tf.constant(20,u.dtype),tf.constant(1e-1,u.dtype))
        a = a_new
        a = tf.reshape(a, tf.shape(a))
        
        
        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)
