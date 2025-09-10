import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from scipy import stats

from classes.objects import margin_obj
from param.margin_pdf import *
from param.margin_op import marginpdf


##################### PARAMETRIC MARGIN FITTING   ###############################

# @tf.function
def marginfit(fam,x):   
    theta = []
    if fam == 'norm':
        loc, scale = stats.norm.fit(x)
        theta = [loc,scale]
    elif fam == 'gamma':
        concentration, loc, rate = stats.gamma.fit(x)
        theta = [concentration, rate]
    return theta

def marginfit_all(x):
    if np.any(x < 0):
        families = ["norm"]
    else:
        families = ["gamma"]
    aic = []
    theta = []
    k = 0
    
    for i in families:
        theta1 = marginfit(i,x)
        iscont = True
        
        mar_pp  = margin_obj(i, theta1, iscont)
        
        logp = np.sum(np.log(marginpdf(mar_pp,x) + 1e-30))

        aic1 = 2*tf.cast(tf.size(theta1),x.dtype) - 2*logp
        aic = aic.append(aic1)
        theta.append(theta1)
        k = k+1
    
    ind_min = np.argmin(aic)
    return families[ind_min], theta[ind_min]