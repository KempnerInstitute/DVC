import tensorflow as tf
from utils.tensor_op import replace_nan_inf,replace_nan_with, replace_inf_with
from param.margin_pdf import *

############################################ GAUSSIAN COST FUNCTION  ######################################

@tf.function(experimental_relax_shapes=True)
def gaussian_cost(u,theta_par):
    p = gaussian_pdf(u,theta_par)
    p = replace_nan_with(p,tf.constant(1,u.dtype))
    eps = tf.constant(2.220446049250313e-16,u.dtype)
    err = -tf.math.reduce_sum(tf.math.log(p+eps),[0])
    return err

############################################ STUDENT COST FUNCTION  ######################################

# @tf.function(experimental_relax_shapes=True)
def student_cost(u,theta):
    p = student_pdf(u,theta)
    eps = tf.constant(2.220446049250313e-16,u.dtype)
    err = -tf.math.reduce_sum(tf.math.log(p+eps),[0])
    return err

############################################ CLAYTON COST FUNCTION  ######################################

@tf.function(experimental_relax_shapes=True)
def clayton_cost(u,theta_cla1):
    p = clayton_pdf(u,theta_cla1)
    p = replace_nan_with(p,tf.constant(1,u.dtype))
    p = replace_inf_with(p, tf.constant(p.dtype.max,u.dtype))
    eps = tf.constant(2.220446049250313e-16,u.dtype)
    err = -tf.math.reduce_sum(tf.math.log(p+eps),[0])
    return err

############################################ CLAYTON ROT 90 COST FUNCTION  ######################################

@tf.function(experimental_relax_shapes=True)
def claytonrot90_cost(u,theta_cla1):
    p = claytonrot90_pdf(u,theta_cla1)
    p = replace_nan_with(p,tf.constant(1,u.dtype))
    p = replace_inf_with(p, tf.constant(p.dtype.max,u.dtype))
    eps = tf.constant(2.220446049250313e-16,u.dtype)
    err = -tf.math.reduce_sum(tf.math.log(p+eps),[0])
    return err

