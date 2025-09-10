#!/usr/bin/env python
# coding: utf-8

# ### Generate data AWGN channel

# In[1]:

import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import pickle
np.random.seed(42)
tf.random.set_seed(42)
plt.style.use('ggplot')
import sys
import numpy

### SETUP COPULA
sys.path.append("D:/NPC")
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
device = gpu_devices[0]
# tf.config.experimental.set_memory_growth(device, True)
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from classes.objects import *
from vine_tree.tree_op import *

from scipy import stats
import pickle
import copy

###########

from param.generate_rvine import *
from param.margin_fit import *
from param.margin_op import *
from param.copula_fit import *
from param.cond_copula import *
from pre_proc.preparation import prep_cop
from pred.prediction import*
from sampling.vine_sample import *
from info.info_estimation import vine_entropy

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

# In[2]:


def sample_AWGN_channel(batch_size, dim, SIGNAL_NOISE = 0.5, SIGNAL_POWER = 2):
    """Simple additive white Gaussian noise channel"""
    x_sample = tf.random.normal((batch_size, dim), stddev = np.sqrt(SIGNAL_POWER))
    y_sample = x_sample + tf.random.normal((batch_size, dim), stddev = np.sqrt(SIGNAL_NOISE))   
    
    return tf.cast(x_sample, tf.float32), tf.cast(y_sample, tf.float32)


# In[3]:


### THIS IS TO GENERATE THE DATA BUT IF YOU ALREADY LOADED IS NOT NECESSARY

batch_size = 2000
var1, var2 = sample_AWGN_channel(batch_size, 8)
# print(y.shape)

var1 = np.array(var1)
var2 = np.array(var2)

data = np.array([var1,var2]).transpose(1,0,2)
data = data.reshape(data.shape[0],-1)
print(data.shape)

# data = np.array([var1.flatten(),var2.flatten()]).T
# print(data.shape)


# ### Fit copula var1-var2

# 

# In[4]:


################################ -  DEFINE THE VINE FOR FITTING - ####################################

### When defining the vine object
### If you use "r-vine", you have to add 'method' and 'r_matrix'
### E.g. using:
### r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, dim)
### and:
### vine = vine_obj_bin(vine_type, families, vine_total_dim, margin_vine, knots, method, r_matrix)

vine_type = "d-vine"
method = 'optimal' #'matrix'#'matrix' 
families = "kercop"
knots = 100

vine_total_dim = data.shape[1]

## Define the margins
margin_vine = []
for i in range(0,vine_total_dim,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine.append(mar_p)

#r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, vine_total_dim)

vine = vine_obj_bin(vine_type, families, vine_total_dim, margin_vine, knots)

######### IF YOU WANT TO LOAD THE SAVED VINE --> Put load_pickle = True
load_pickle = False

if load_pickle:
    pickle_in = open("awgn_vine_16_2000","rb")
    dict_save = pickle.load(pickle_in)
    # print(dict_save.keys())
    vine_copulas = dict_save["vine_copulas"]
    data = dict_save["data"]
    r_matrix = dict_save["r_matrix"]
    vine_depth = 16

    ## Here you load the copulas in the vine
    vine.copulas = vine_copulas
    vine.r_matrix = r_matrix
    vine.vine_depth = vine_depth


# In[5]:


################################ -  VINE FITTING - ####################################
### Vine fitting instructions

# General parameters:
# - parallel: True or False           (Fit in parallel each level)
# - binning: True or False            (It can be True only if parallel is False)
# - param: True or False              (Parametric or Non-parametric)
# - vine_depth: any                   (Max level of vine to fit)
# - Fitted: True or False             (If the vine was already fitted, recompute some needed variables)

# Parametric parameters:
# - param_families: ["ind","gaussian","student","clayton","claytonrot90"]   (Decide which parametric families to fit)

# Non-parametric parameters:
# - opt_method: 'LL1' or 'LL2'

# Binning parameters:
# - n_bin: any                        (Number of bins)

### Parameters

param = False
binning = False
n_bins = 3

### Make data divisible for bins and k-fold

x = np.array(data,np.float32)

if binning == True:
    if param == False:
        exc = x.shape[0] % n_bins*5 #tf.math.floormod(tf.shape(x)[0],n_bin*5)
    else:
        exc = x.shape[0] % n_bins #tf.math.floormod(tf.shape(x)[0],n_bin)
    x = x[:x.shape[0]-exc,:]
else:
    if param == False:
        exc = x.shape[0] % 5 #tf.math.floormod(tf.shape(x)[0],5)
        x = x[:x.shape[0]-exc,:]

### Prepare copula

sort_n = 'rand'
e = prep_cop(x, vine, sort_n)
#print(e)

### FITTING
# Add parameters in a dictionary
vine_depth_fit = 4

gen_dict = {'parallel':True, 'binning':binning, 'param':param, 'vine_depth':vine_depth_fit, 'fitted':False}  #vine_depth
par_dict = {'param_families':["ind","gaussian"]}  #["ind","gaussian","student","clayton","claytonrot90"]
npc_dict = {'opt_method':'LL1','batch_paral':3}
bin_dict = {'n_bin':n_bins}

save_vine = False

vine.fit(x,gen_dict,npc_dict,par_dict,bin_dict)

if save_vine:
    dict_save = {'vine_copulas': vine.copulas, 'r_matrix': vine.r_matrix, 'data': x}
    pickle_out = open("awgn_vine_16_2000","wb")
    pickle.dump(dict_save,pickle_out)
    pickle_out.close()


# ### Fit copula v2

# In[6]:


def copy_copulas(vine):
    new_copula = []
    for tr in range(len(vine.copulas)):
        if tr <= vine.vine_depth:
            new_copula.append(copy.copy(vine.copulas[tr]))
        else:
            list_copula_tmp = []
            for jj in range(len(vine.copulas)-tr):
                list_copula_tmp.append(copy.copy(vine.copulas[tr][jj]))
            new_copula.append(list_copula_tmp)
    return new_copula


# In[7]:


################################ -  DEFINE THE VINE FOR FITTING - ####################################

### When defining the vine object
### If you use "r-vine", you have to add 'method' and 'r_matrix'
### E.g. using:
### r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, dim)
### and:
### vine = vine_obj_bin(vine_type, families, vine_total_dim, margin_vine, knots, method, r_matrix)

#vine_type = "c-vine"
#method = 'matrix' #'matrix' 'optimal'
#families = "kercop"
#knots = 50

var_x2 = data[:,8:]

vine_total_dim_x2 = var_x2.shape[1]

## Define the margins
margin_vine_x2 = []
for i in range(0,vine_total_dim_x2,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine_x2.append(mar_p)
    
vine_x2 = vine_obj_bin(vine_type, families, vine_total_dim_x2, margin_vine_x2, knots)

######### IF YOU WANT TO LOAD THE SAVED VINE --> Put load_pickle = True
# load_pickle = False

# if load_pickle:
#     pickle_in = open("awgn_vine_x2_8","rb")
#     dict_save = pickle.load(pickle_in)
#     # print(dict_save.keys())
#     vine_copulas_x2 = dict_save["vine_copulas"]
#     # sample = dict_save["data"]
#     r_matrix_x2 = dict_save["r_matrix"]
#     vine_depth_x2 = 8

#     ## Here you load the copulas in the vine
#     vine_x2.copulas = vine_copulas_x2
#     vine_x2.r_matrix = r_matrix_x2
#     vine_x2.vine_depth = vine_depth_x2

######################## Nest the vine
## Here you load the copulas in the vine
vine_x2.copulas = copy_copulas(vine)
# vine_x2.copulas = vine.copulas.copy()
# vine_x2.r_matrix = vine.r_matrix.copy()
# vine_x2.vine_depth = 8


# In[8]:


################################ -  VINE FITTING - ####################################
### Vine fitting instructions

# General parameters:
# - parallel: True or False           (Fit in parallel each level)
# - binning: True or False            (It can be True only if parallel is False)
# - param: True or False              (Parametric or Non-parametric)
# - vine_depth: any                   (Max level of vine to fit)
# - Fitted: True or False             (If the vine was already fitted, recompute some needed variables)

# Parametric parameters:
# - param_families: ["ind","gaussian","student","clayton","claytonrot90"]   (Decide which parametric families to fit)

# Non-parametric parameters:
# - opt_method: 'LL1' or 'LL2'

# Binning parameters:
# - n_bin: any                        (Number of bins)

### Parameters

param = False
binning = False
n_bins = 3

### Make data divisible for bins and k-fold

x = np.array(var_x2,np.float32)

if binning == True:
    if param == False:
        exc = x.shape[0] % n_bins*5 #tf.math.floormod(tf.shape(x)[0],n_bin*5)
    else:
        exc = x.shape[0] % n_bins #tf.math.floormod(tf.shape(x)[0],n_bin)
    x = x[:x.shape[0]-exc,:]
else:
    if param == False:
        exc = x.shape[0] % 5 #tf.math.floormod(tf.shape(x)[0],5)
        x = x[:x.shape[0]-exc,:]

### Prepare copula

sort_n = 'rand'
e = prep_cop(x, vine_x2, sort_n)
print(e)

### FITTING
# Add parameters in a dictionary
#vine_depth_fit = 8

gen_dict = {'parallel':True, 'binning':binning, 'param':param, 'vine_depth':vine_depth_fit, 'fitted':False}  #vine_depth
par_dict = {'param_families':["ind","gaussian"]}  #["ind","gaussian","student","clayton","claytonrot90"]
npc_dict = {'opt_method':'LL1','batch_paral':3}
bin_dict = {'n_bin':n_bins}

save_vine = False

vine_x2.fit(x,gen_dict,npc_dict,par_dict,bin_dict)

if save_vine:
    dict_save = {'vine_copulas': vine.copulas, 'r_matrix': vine.r_matrix, 'data': x}
    pickle_out = open("awgn_8_2000","wb")
    pickle.dump(dict_save,pickle_out)
    pickle_out.close()


# In[9]:


def cond_vine_entropy(vine,vine_f2,info_dict):
    alpha = info_dict['alpha']
    cases = info_dict['cases'] #number of samples in each iteration 
    max_iter = info_dict['iterations']
    d = vine.n_cop
    d_f2 = vine_f2.n_cop

    norm_dis = tfd.Normal(loc=0., scale=1.) 
    conf = norm_dis.quantile(1-alpha)
    tim = 0  #Add as parameter if you want to change it

    mo = 0 
    varsum1 = 0 
    infoc1 = 0
    cond_entr = 0
    entr_f2 = 0
    stderr1 = 1e+6
    stderr2 = 1e+6 
    stderr_tot = 1e+6
    erreps = 1e-3

    MI_XY=numpy.zeros(max_iter+1,numpy.float)
    
    mag = tf.math.reduce_max(vine.grid_u.ex)
    mig = tf.math.reduce_min(vine.grid_u.ex)

    mag_f2 = tf.math.reduce_max(vine_f2.grid_u.ex)
    mig_f2 = tf.math.reduce_min(vine_f2.grid_u.ex)

    while ((stderr1 >= erreps) | (stderr2 >= erreps) | (stderr_tot >= erreps) ) & (mo < max_iter):
        mo = mo+1
        if vine.param == False:

            ## Sample from joint copula and compute prob.
            w = tf.random.uniform([cases,d], minval=0, maxval=1, dtype=vine.data_x.dtype)
            w = (mag-mig)*(w-tf.math.reduce_min(w))/(tf.math.reduce_max(w)-tf.math.reduce_min(w))+mig
            
            sample = vine_copula_sample(vine,cases)
            
            p, p_copula = vine.evaluation(sample)

            ## Sample from var2 copula and compute prob.
    
            sample_f2 = sample[:,d_f2:]
            # sample_f2 = vine_copula_sample(vine_f2,cases)
            
            p_f2, p_copula_f2 = vine_f2.evaluation(sample_f2)

            ## Compute cond entr.

            p_cond = np.exp(np.log(p.numpy()) - np.log(p_f2.numpy()))
            
            log2_cond = np.log2(p_cond)
            log2_cond[p_cond == 0] = 0 

            cond_entr = cond_entr + ( np.mean(log2_cond) - cond_entr) / mo  #tf.math.reduce_mean

            varsum1 = varsum1 + np.sum((log2_cond - cond_entr)**2)
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
            
            ## Compute entr.
            log2_f2 = np.log2(p_f2.numpy())
            log2_f2[p_f2 == 0] = 0 

            entr_f2 = entr_f2 + ( np.mean(log2_f2) - entr_f2) / mo
            print('MI: ', -(entr_f2 - cond_entr))

            MI_XY[mo-1] = -(entr_f2 - cond_entr)
        else:
            sample = vine_cop_par_sample(vine,cases)

            # Compute pdf of samples
            p, pcop = vine.evaluation(sample)
            
            log2pp = np.log2(pcop.numpy())
            log2pp[pcop == 0] = 0 

            infoc1 = infoc1 + ( np.mean(log2pp) - infoc1) / mo
            varsum1 = varsum1 + np.sum((log2pp - infoc1)**2)
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
    
    
    return MI_XY, entr_f2, cond_entr


# In[ ]:


info_dict = {'cases':10000, 'iterations':20, 'alpha': 0.05}
MI_XY, H2,H1_2 = cond_vine_entropy(vine,vine_x2,info_dict)
print('MI: ', MI_XY)
plt.plot(MI_XY[:20])


# In[12]:


plt.plot(MI_XY[:20])


# In[28]:


def theoretic_mutual_information_AWGN(power, noise, dim):
    return dim * 0.5 * np.log2(1 + power/noise)

th_mi = theoretic_mutual_information_AWGN(2, 0.5,8)
print(th_mi)


# ### Example sample vine

# In[13]:


cases = 2000
sample = vine_copula_sample(vine,cases)

fig, axs = plt.subplots(vine.n_cop, vine.n_cop,figsize=(30,30))
for i in range(0,vine.n_cop,1):
    for j in range(i+1,vine.n_cop,1):
        axs[i,j].plot(data[:,i],data[:,j],'b.')    
        axs[i,j].plot(sample[:,i],sample[:,j],'r.')    
        axs[i,j].set_title(str(i+1)+","+str(j+1))


# In[37]:


cases = 2000
sample_x2 = vine_copula_sample(vine_x2,cases)

fig, axs = plt.subplots(vine_x2.n_cop, vine_x2.n_cop,figsize=(30,30))
for i in range(0,vine_x2.n_cop,1):
    for j in range(i+1,vine_x2.n_cop,1):
        axs[i,j].plot(data[:,vine_x2.n_cop+i],data[:,vine_x2.n_cop+j],'b.')    
        axs[i,j].plot(sample_x2[:,i],sample_x2[:,j],'r.')    
        axs[i,j].set_title(str(i+1)+","+str(j+1))

