import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from utils.prob_op import kernel_pdf2
from utils.tensor_op import update_tensor, update_tensor2D, replace_nan_inf
from pre_proc.preparation import prep_copula
from sampling.vine_sample import vine_copula_sample, vine_cop_par_sample

########################## INFO ESTIMATION ##################################

def vine_entropy(vine,info_dict):
    alpha = info_dict['alpha']
    cases = info_dict['cases'] #number of samples in each iteration 
    max_iter = info_dict['iterations']
    d = vine.n_cop

    norm_dis = tfd.Normal(loc=0., scale=1.) 
    conf = norm_dis.quantile(1-alpha)
    tim = 0  #Add as parameter if you want to change it

    mo = 0 
    varsum1 = 0 
    infoc1 = 0
    stderr1 = 1e+6
    stderr2 = 1e+6 
    stderr_tot = 1e+6
    erreps = 1e-3

    mag = tf.math.reduce_max(vine.grid_u.ex)
    mig = tf.math.reduce_min(vine.grid_u.ex)

    while ((stderr1 >= erreps) | (stderr2 >= erreps) | (stderr_tot >= erreps) ) & (mo < max_iter):
        mo = mo+1
        if vine.param == False:
            w = tf.random.uniform([cases,d], minval=0, maxval=1, dtype=vine.data_x.dtype)
            w = (mag-mig)*(w-tf.math.reduce_min(w))/(tf.math.reduce_max(w)-tf.math.reduce_min(w))+mig
            
            sample = vine_copula_sample(vine,cases)

            #sample = tf.convert_to_tensor(sample)
            #print(type(sample))      

            p, p_copula, plog = vine.evaluation(sample)
            
            log2pp = np.log2(p_copula.numpy())
            log2pp[p_copula == 0] = 0 

            infoc1 = infoc1 + ( np.mean(log2pp) - infoc1) / mo  #tf.math.reduce_mean

            varsum1 = varsum1 + np.sum((log2pp - infoc1)**2)
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
        else:
            sample, _, _, _ = vine_cop_par_sample(vine,cases)

            # Compute pdf of samples
            p, pcop, _ = vine.evaluation(sample)
            
            log2pp = np.log2(pcop.numpy())
            log2pp[pcop == 0] = 0 

            infoc1 = infoc1 + ( np.mean(log2pp) - infoc1) / mo
            varsum1 = varsum1 + np.sum((log2pp - infoc1)**2)
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
            
    return infoc1