import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from scipy import stats
from utils.bijector import *


############################################ MARGIN PDF      ############################################

def marginpdf(marg,x): 
    logf_tmp = np.zeros(np.shape(x)[0],x.dtype)
    if marg.dist == 'norm': 
        loc = marg.theta[0]
        scale = marg.theta[1]
        norm_dis = tfd.Normal(loc,scale)
        logf_tmp = norm_dis.prob(x)      
    elif marg.dist == 'gamma':
        concentration = marg.theta[0]
        rate = 1/marg.theta[1]
        gamma_dis = tfd.Gamma(concentration,rate)
        logf_tmp = gamma_dis.prob(x)
    return logf_tmp


############################################ MARGIN CDF      ############################################

def margincdf(marg,x):
    Fp_tmp = np.zeros(np.shape(x)[0],x.dtype)

    if marg.dist == 'norm':
        loc = marg.theta[0]
        scale = marg.theta[1]
        Fp_tmp = stats.norm.cdf(x,loc,scale)
    elif marg.dist == 'gamma':
        concentration = marg.theta[0]
        rate = marg.theta[1]
        Fp_tmp = stats.gamma.cdf(x, concentration, 0, rate)
    return Fp_tmp


############################################ MARGIN INV      ############################################

def margininv(marg,x): 
    Fp_tmp = np.zeros(np.shape(x)[0],x.dtype)
    if marg.dist == 'norm':   #"gaussian"
        loc = marg.theta[0]
        scale = marg.theta[1]
        Fp_tmp = NormalCDF.forward(NormalCDF(loc,scale), x)    
    elif marg.dist == 'gamma':  #"gamma"
        concentration = marg.theta[0]
        rate = marg.theta[1]
        loc = 0
        Fp_tmp = stats.gamma.ppf(x, concentration, loc, rate)
    return Fp_tmp