import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from scipy.special import gammaln
from scipy.stats import t
from utils.tensor_op import replace_nan_inf,update_tensor2D
import math as m

############################################ GAUSSIAN MARGIN PDF  ######################################

@tf.function(experimental_relax_shapes=True)
def gaussian_pdf(u,theta_par):
    norm_dis = tfd.Normal(loc=tf.constant(0.,u.dtype), scale=tf.constant(1.,u.dtype))
    x = norm_dis.quantile(u)
    p = tf.exp((2*theta_par*x[:,0,:]*x[:,1,:] - (theta_par**2) * (x[:,0,:]**2 + x[:,1,:]**2))/(2*(1-theta_par**2))) / tf.math.sqrt(1-theta_par**2)
    return p

############################################ CLAYTON MARGIN PDF  ######################################

@tf.function(experimental_relax_shapes=True)
def clayton_pdf(u,theta):
    p = (1 + theta) * (u[:,0,:] * u[:,1,:])**(-1-theta) * (u[:,0,:]**(-theta) + u[:,1,:]**(-theta) - 1)**(-1/theta-2)
    if tf.shape(tf.shape(theta))[0] == 0:
        theta = theta[...,tf.newaxis]
    cond = tf.math.equal(theta,0)
    ind = tf.where(tf.equal(cond,True))
    ind = tf.cast(ind,tf.int32)

    for i in ind:
        newval = tf.ones(tf.shape(u)[0],u.dtype)
        p = update_tensor2D(p, i[0] , newval)
    return p

############################################ CLAYTON ROT 90 MARGIN PDF  ######################################

@tf.function(experimental_relax_shapes=True)
def claytonrot90_pdf(u,theta):
    p = (1 + theta) * (u[:,0,:] * (1-u[:,1,:]))**(-1-theta) * (u[:,0,:]**(-theta) + (1-u[:,1,:])**(-theta) - 1)**(-1/theta-2)
    if tf.shape(tf.shape(theta))[0] == 0:
        theta = theta[...,tf.newaxis]
    cond = tf.math.equal(theta,0)
    ind = tf.where(tf.equal(cond,True))
    ind = tf.cast(ind,tf.int32)

    for i in ind:
        newval = tf.ones(tf.shape(u)[0],u.dtype)
        p = update_tensor2D(p, i[0] , newval)
    return p

############################################ STUDENT MARGIN PDF  ######################################

# @tf.function(experimental_relax_shapes=True)
def gammaln1(x):
    return tf.py_function(gammaln, [x], x.dtype)

# @tf.function(experimental_relax_shapes=True)
def tpdf(x,vk):
    term = tf.exp(gammaln1((vk + 1) / 2) - gammaln1(vk/2))
    pi = tf.constant(m.pi,x.dtype)
    y = term / (tf.math.sqrt(vk*pi) * (1 + (x**2) / vk) ** ((vk + 1)/2))
    return y

# @tf.function(experimental_relax_shapes=True)
def student_pdf(u,theta):
    if tf.shape(tf.shape(theta)) == 1:
        theta = theta[tf.newaxis,...]
    
    df = theta[:,1]
    loc = tf.constant(0,u.dtype)
    scale = tf.constant(1,u.dtype)
    pi = tf.constant(m.pi,u.dtype)

    x = tf.py_function(t.ppf, [u, theta[:,1], loc, scale], u.dtype)
    factor1 = gammaln1(theta[:,1]/2+1)
    factor2 = -gammaln1(theta[:,1]/2) - tf.math.log(pi) - tf.math.log(theta[:,1]) - tf.math.log(1-theta[:,0]**2)/2 - tf.math.log(tpdf(x[:,0,:],theta[:,1])) - tf.math.log(tpdf(x[:,1,:],theta[:,1]))

    factor3 = (-(theta[:,1]+2)/2) * tf.math.log(1 + (x[:,0,:]**2 + x[:,1,:]**2 - theta[:,0] * x[:,0,:] * x[:,1,:]) / (theta[:,1]*(1-theta[:,0]**2)))
    p = tf.exp(factor1 + factor2 + factor3)
    p = replace_nan_inf(p)
    return p
