import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from scipy import stats

from utils.bijector import *
from param.margin_pdf import *

############################################ COPULA PDF      ############################################

def copulapdf(vine_par,u):
    c = np.zeros(np.shape(u)[0],u.dtype)
    u = tf.convert_to_tensor(u)
    
    if vine_par.family == 'ind':
        c = tf.ones(tf.shape(u)[0],u.dtype)
    elif vine_par.family == 'gaussian':
        theta = tf.convert_to_tensor(vine_par.theta,u.dtype)
        c = gaussian_pdf(u,theta)
    if vine_par.family == 'student':
        theta = tf.convert_to_tensor(vine_par.theta,u.dtype)
        c = student_pdf(u,theta)
    if vine_par.family == 'clayton':
        theta = tf.convert_to_tensor(vine_par.theta,u.dtype)
        c = clayton_pdf(u,theta)
    if vine_par.family == 'claytonrot90':
        theta = tf.convert_to_tensor(vine_par.theta,u.dtype)
        c = claytonrot90_pdf(u,theta)
    return c

############################################ COPULA CONDITIONED CDF      ############################################

def copulaccdf(vine_par,u):
    loc = 0
    scale = 1
    
    u[u>=1-1e-7] = 1-1e-7
    u[u<=1e-7] = +1e-7
    
    c = np.zeros(np.shape(u)[0],u.dtype)
    if vine_par.family == 'ind':
        c = u[:,0]
    elif vine_par.family == 'gaussian':
        x = NormalCDF.forward(NormalCDF(loc,scale), u)
        theta = vine_par.theta
        tmp = (x[:,0] - theta * x[:,1]) / np.sqrt(1-theta**2)
        c = NormalCDF.inverse(NormalCDF(loc,scale), tmp)
    elif vine_par.family == 'student':
        theta1 = vine_par.theta[0]
        theta2 = vine_par.theta[1]
        x = stats.t.ppf(u, theta2, loc, scale)
        tmp = np.sqrt((theta2+1) / (theta2+x[:,1]**2)) * (x[:,0] - theta1 * x[:,1]) / (np.sqrt(1-theta1**2)) #, theta[1]+1)
        c = stats.t.cdf(tmp, theta2+1, loc, scale)
    elif vine_par.family == 'clayton':   
        theta = vine_par.theta
        if theta == 0:
            c = u[:,0]
        else:
            c = np.maximum(u[:,1]**(-1-theta) * (u[:,0]**(-theta) + u[:,1]**(-theta) - 1) ** (-1-1/theta),0)
    elif vine_par.family == 'claytonrot90':   
        theta = vine_par.theta
        if theta == 0:
            c = u[:,0]
        else:
            c = np.maximum((1-u[:,1])**(-1-theta) * (u[:,0]**(-theta) + (1-u[:,1])**(-theta) - 1) ** (-1-1/theta),0)
    
    return c

############################################ COPULA INVERSE CONDITIONED CDF      ############################################

def copulainvccdf(vine_par,u):
    loc = 0
    scale = 1
    
    u[u>=1-1e-7] = 1-1e-7
    u[u<=1e-7] = +1e-7
    c = np.zeros(np.shape(u)[0],u.dtype)
    
    if vine_par.family == 'ind':
        c = u[:,0]
    elif vine_par.family == 'gaussian':
        x = NormalCDF.forward(NormalCDF(loc,scale), u)
        theta = vine_par.theta
        tmp = x[:,0] * np.math.sqrt(1-theta**2) + theta * x[:,1]        
        c = NormalCDF.inverse(NormalCDF(loc,scale), tmp)
    if vine_par.family == 'student':
        theta1 = vine_par.theta[0]
        theta2 = vine_par.theta[1]
        x = stats.t.ppf(u, theta2, loc, scale)
        param = theta2 + 1 
        tmp_inv = stats.t.ppf(u[:,0], param, loc, scale)
        tmp = np.sqrt( ((1-theta1**2) * (theta2 + x[:,1]**2)) / (theta2+1) ) * tmp_inv + theta1 * x[:,1]
        c = stats.t.cdf(tmp, theta2, loc, scale)
    if vine_par.family == 'clayton':
        theta = vine_par.theta
        if theta == 0:
            c = u[:,0]
        else:           
            c = (1 - u[:,1]**(-theta) + (u[:,0] * (u[:,1]**(1+theta)))**(-theta/(1+theta)))**(-1/theta)
    if vine_par.family == 'claytonrot90':
        theta = vine_par.theta
        if theta == 0:
            c = u[:,0]
        else:           
            c = (1 - (1 - u[:,1]) **(-theta) + (u[:,0] * ((1 - u[:,1])**(1+theta)))**(-theta/(1+theta)))**(-1/theta)
    return c