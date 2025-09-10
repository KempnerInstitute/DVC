import os
import sys

# Suppress TensorFlow informational messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warnings, 3=errors only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations message

# ### Generate data AWGN channel

import numpy as np
# matplotlib inline equivalent for regular Python scripts
import matplotlib as mpl
mpl.use('Agg')  # Use non-interactive backend for scripts
import matplotlib.pyplot as plt
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import tensorflow as tf

# Additional TensorFlow logging suppression
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import pickle
np.random.seed(42)
#tf.set_random_seed(42)
tf.random.set_seed(42)
plt.style.use('ggplot')

#tf.compat.v1.disable_eager_execution()

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
### SETUP COPULA

# Add the DVC_tensorflow directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
dvc_tensorflow_dir = os.path.join(current_dir, '..', '..')
sys.path.append(dvc_tensorflow_dir)

print(sys.path)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#device = gpu_devices[0]
# tf.config.experimental.set_memory_growth(device, True)
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import scipy.io as sio

from classes.objects import *
from vine_tree.tree_op import *

from scipy import stats

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




def sample_AWGN_channel(batch_size, dim, SIGNAL_NOISE = 0.5, SIGNAL_POWER = 2):
    """Simple additive white Gaussian noise channel"""
    x_sample = tf.random.normal((batch_size, dim), stddev = np.sqrt(SIGNAL_POWER))
    y_sample = x_sample + tf.random.normal((batch_size, dim), stddev = np.sqrt(SIGNAL_NOISE))   
    
    return tf.cast(x_sample, tf.float32), tf.cast(y_sample, tf.float32)


### THIS IS TO GENERATE THE DATA BUT IF YOU ALREADY LOADED IS NOT NECESSARY

batch_size = 100
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


#### Generate random matrix

#cases = 2000        ### Number of samples
vine_type = 'c-vine' # or 'd-vine' or 'c-vine'
method = 'matrix'  # or 'r_matrix'  only with r-vine
binning = False
n_bin = 3
dim = 16               # Dimension of the vine for random r-vine or c-vine or d-vine
#dim  =8

r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, dim)

print(r_matrix)

binning = False
n_bin = 3

########## DEFINE MARGINS

margin_vine = []
for i in range(0,dim,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine.append(mar_p)

#for i in range(0,dim,1):
#    print(margin_vine[i].dist, end =' ')
#    print(margin_vine[i].theta, end =' ')

############## DEFINE COPULAS

tr = 0
cop_vine = []
for i in range(dim,1,-1):
    cop_vine1 = []
    for j in range(0,i-1,1):
        if (tr == 0) | (binning == False):
            cop_p = cop_par_obj('clayton',4.5)  #
            cop_vine1.append(cop_p)
        else:
            cop_vine11 = []
            for bb in range(0,n_bin,1):
                cop_p = cop_par_obj('gaussian',0.9)
                cop_vine11.append(cop_p)
            cop_vine1.append(cop_vine11)
    cop_vine.append(cop_vine1)
    tr += 1

d = len(r_matrix)
#for tr in range(0,d-1,1):
    #for col in range(0,d-1-tr,1):
        #if (tr == 0) | (binning == False):
            #print('edge: {} '.format(matrix_edges[tr][col]), 'cop family: {}'.format(cop_vine[tr][col].family), 'theta: {}'.format(cop_vine[tr][col].theta))
        #else:
            #for bb in range(0,n_bin,1):
               # print('edge: {} '.format(matrix_edges[tr][col]),'bin: {} '.format(bb), 'cop family: {}'.format(cop_vine[tr][col][bb].family), 'theta: {}'.format(cop_vine[tr][col][bb].theta))



################### DEFINE VINE #################

#vine_type = "c-vine"
method = 'matrix' #'matrix' 'optimal'
families = "kercop"
knots = 50

vine_depth = len(r_matrix)

#vine_depth = 8

margin_vine = []
for i in range(0,vine_depth,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine.append(mar_p)
    
vine = vine_obj_bin(vine_type, families, vine_depth, margin_vine, knots, method, r_matrix)


######### IF YOU WANT TO LOAD THE SAVED VINE --> Put load_pickle = True
load_pickle = False

if load_pickle:
    pickle_in = open("awgn_vine_16","rb")
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


param = False
binning = False
n_bin = 3

##### Make data divisible for bins and k-fold  #####

# x = dat # sample  #dat
x = data ## CHANGE THIS IF YOU WANT TO USE DATA THAT ARE LOADED
# x = np.array(x,np.float32)


if binning == True:
    if param == False:
        exc = tf.math.floormod(tf.shape(x)[0],n_bin*5)
    else:
        exc = tf.math.floormod(tf.shape(x)[0],n_bin)
    x = x[:tf.shape(x)[0]-exc,:]
else:
    if param == False:
        exc = tf.math.floormod(tf.shape(x)[0],5)
        x = x[:tf.shape(x)[0]-exc,:]

## Prepare copula
sort_n = 'rand'
e = prep_cop(x, vine, sort_n)
print(e)


### FITTING
# Parameters:
# - Data: x
# - Parallel: True or False
# - Bandwidth optimization: LL1 or LL2
# - Binning: True or false.    It can be True only if parallel is false
# - n_bin: Select the number of bins
# parallel = False

#vine_depth=

gen_dict = {'parallel':True, 'binning':binning, 'param':param, 'vine_depth':vine_depth, 'fitted':False}  #vine_depth
par_dict = {'param_families':["ind","gaussian"]}  #["ind","gaussian","student","clayton","claytonrot90"]
npc_dict = {'opt_method':'LL1','batch_paral':4}
bin_dict = {'n_bin':n_bin}

save_vine = False
print('x')
print(x.shape)

vine.fit(x,gen_dict,npc_dict,par_dict,bin_dict)

if save_vine:
    dict_save = {'vine_copulas': vine.copulas, 'r_matrix': vine.r_matrix, 'data': data}
    pickle_out = open("awgn_vine_16","wb")
    pickle.dump(dict_save,pickle_out)
    pickle_out.close()


# ### Fit copula - var2


#### Generate random matrix

#cases = 2000        ### Number of samples
#vine_type = 'c-vine' # or 'd-vine' or 'c-vine'
#method = 'matrix'  # or 'r_matrix'  only with r-vine
#binning = False
#n_bin = 3
dim = 8                # Dimension of the vine for random r-vine or c-vine or d-vine

r_matrix_x2, ind_vine_x2, nodes_x2, matrix_edges_x2 = prepare_vine(vine_type, dim)

print(r_matrix_x2)


########## DEFINE MARGINS

margin_vine_x2 = []
for i in range(0,dim,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine_x2.append(mar_p)

#for i in range(0,dim,1):
#    print(margin_vine_x2[i].dist, end =' ')
#    print(margin_vine_x2[i].theta, end =' ')

############## DEFINE COPULAS

tr = 0
cop_vine_x2 = []
for i in range(dim,1,-1):
    cop_vine1 = []
    for j in range(0,i-1,1):
        if (tr == 0) | (binning == False):
            cop_p = cop_par_obj('clayton',4.5)  #
            cop_vine1.append(cop_p)
        else:
            cop_vine11 = []
            for bb in range(0,n_bin,1):
                cop_p = cop_par_obj('gaussian',0.9)
                cop_vine11.append(cop_p)
            cop_vine1.append(cop_vine11)
    cop_vine_x2.append(cop_vine1)
    tr += 1

d = len(r_matrix_x2)
#for tr in range(0,d-1,1):
#    for col in range(0,d-1-tr,1):
#        if (tr == 0) | (binning == False):
#            print('edge: {} '.format(matrix_edges_x2[tr][col]), 'cop family: {}'.format(cop_vine_x2[tr][col].family), 'theta: {}'.format(cop_vine_x2[tr][col].theta))
#        else:
#            for bb in range(0,n_bin,1):
#                print('edge: {} '.format(matrix_edges_x2[tr][col]),'bin: {} '.format(bb), 'cop family: {}'.format(cop_vine_x2[tr][col][bb].family), 'theta: {}'.format(cop_vine_x2[tr][col][bb].theta))



################### DEFINE VINE #################

#vine_type = "c-vine"
#method = 'matrix' #'matrix' 'optimal'
#families = "kercop"
#knots = 100

vine_depth_x2 = len(r_matrix_x2)

# vine_depth = 20 #len(r_matrix)

margin_vine_x2 = []
for i in range(0,vine_depth_x2,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine_x2.append(mar_p)
    
vine_x2 = vine_obj_bin(vine_type, families, vine_depth_x2, margin_vine_x2, knots, method, r_matrix_x2)

## Here you load the copulas in the vine
# vine_x2.copulas = vine_copulas #_x2

load_pickle = False

if load_pickle:
    pickle_in = open("awgn_vine_x2_8","rb")
    dict_save = pickle.load(pickle_in)
    # print(dict_save.keys())
    vine_copulas_x2 = dict_save["vine_copulas"]
    # sample = dict_save["data"]
    r_matrix_x2 = dict_save["r_matrix"]
    vine_depth_x2 = 8

    ## Here you load the copulas in the vine
    vine_x2.copulas = vine_copulas_x2
    vine_x2.r_matrix = r_matrix_x2
    vine_x2.vine_depth = vine_depth_x2


vine_x2.copulas=vine.copulas.copy()


#param = False
#binning = False
#n_bin = 3

## Make data divisible for bins and k-fold
# x = dat # sample  #dat
# x2 = var2 ## CHANGE THIS IF YOU WANT TO USE DATA THAT ARE LOADED
#x2 = data[:,8:]
x2 = data[:,8:]
# x = np.array(x,np.float32)
print(x2.shape)

if binning == True:
    if param == False:
        exc = tf.math.floormod(tf.shape(x2)[0],n_bin*5)
    else:
        exc = tf.math.floormod(tf.shape(x2)[0],n_bin)
    x2 = x2[:tf.shape(x)[0]-exc,:]
else:
    if param == False:
        exc = tf.math.floormod(tf.shape(x2)[0],5)
        x2 = x2[:tf.shape(x2)[0]-exc,:]

## Prepare copula

sort_n = 'rand'
e_x2 = prep_cop(x2, vine_x2, sort_n)
print(e_x2)


### FITTING
# Parameters:
# - Data: x
# - Parallel: True or False
# - Bandwidth optimization: LL1 or LL2
# - Binning: True or false.    It can be True only if parallel is false
# - n_bin: Select the number of bins
# parallel = False

gen_dict = {'parallel':True, 'binning':binning, 'param':param, 'vine_depth':8, 'fitted':False}  #vine_depth
par_dict = {'param_families':["ind","gaussian"]}  #["ind","gaussian","student","clayton","claytonrot90"]
npc_dict = {'opt_method':'LL1','batch_paral':4}
bin_dict = {'n_bin':n_bin}

save_vine = False

vine_x2.fit(x2,gen_dict,npc_dict,par_dict,bin_dict)


if save_vine:
    dict_save = {'vine_copulas': vine_x2.copulas, 'r_matrix': vine_x2.r_matrix, 'data': x2}
    pickle_out = open("awgn_vine_x2_8","wb")
    pickle.dump(dict_save,pickle_out)
    pickle_out.close()
    
    print('done')



@tf.function
def compute_max(tensor):
    return tf.math.reduce_max(tensor)


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
    info=[]
    
 
    mag = tf.math.reduce_max(vine.grid_u.ex)
    mig = tf.math.reduce_min(vine.grid_u.ex)

    mag_f2 = tf.math.reduce_max(vine_f2.grid_u.ex)
    mig_f2 = tf.math.reduce_min(vine_f2.grid_u.ex)

    while ((stderr1 >= erreps) | (stderr2 >= erreps) | (stderr_tot >= erreps) ) & (mo < max_iter):
        mo = mo+1
        info.append(cond_entr-entr_f2)
        print(cond_entr-entr_f2)
        if vine.param == False:

            ## Sample from joint copula and compute prob.
            w = tf.random.uniform([cases,d], minval=0, maxval=1, dtype=vine.data_x.dtype)
            w = (mag-mig)*(w-tf.math.reduce_min(w))/(tf.math.reduce_max(w)-tf.math.reduce_min(w))+mig
            
            sample , u, p1, p2 = vine_copula_sample(vine,cases)
            
            p, p_copula, log_marg_f = vine.evaluation(sample)

            ## Sample from var2 copula and compute prob.
            # w_f2 = tf.random.uniform([cases,d_f2], minval=0, maxval=1, dtype=vine_f2.data_x.dtype)
            # w_f2 = (mag_f2-mig_f2)*(w_f2-tf.math.reduce_min(w_f2))/(tf.math.reduce_max(w_f2)-tf.math.reduce_min(w_f2))+mig_f2
                       
            sample_f2 , u, p1, p2 = vine_copula_sample(vine_f2,cases)

            #sample_f2 = sample[:,0:d_f2]
            
            p_f2, p_copula_f2, log_marg_f = vine_f2.evaluation(sample_f2)

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

        else:
            sample = vine_cop_par_sample(vine,cases)

            # Compute pdf of samples
            p, pcop = vine.evaluation(sample)
            
            log2pp = np.log2(pcop.numpy())
            log2pp[pcop == 0] = 0 

            infoc1 = infoc1 + ( np.mean(log2pp) - infoc1) / mo
            varsum1 = varsum1 + np.sum((log2pp - infoc1)**2)
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
        
    return entr_f2, cond_entr, info


vine.copulas[1].pd_grid_uv.shape


info_dict = {'cases':1000, 'iterations':50, 'alpha': 0.05}
H2,H1_2,info = cond_vine_entropy(vine,vine_x2,info_dict)


print(vine.ind_edge_rel)


print('Entropy H2',H2)
print('Entropy H1|2',H1_2)


MI_XY = -H2 + H1_2
print(MI_XY)


def theoretic_mutual_information_AWGN(power, noise, dim):
    return dim * 0.5 * np.log2(1 + power/noise)

th_mi = theoretic_mutual_information_AWGN(2, 0.5,8)
print(th_mi)


# ### Example sample vine

cases = 1000
sample_vine , u, p1, p2 = vine_copula_sample(vine,cases)

fig, axs = plt.subplots(vine_depth, vine_depth,figsize=(30,30))
for i in range(0,vine_depth,1):
    for j in range(i+1,vine_depth,1):
        axs[i,j].plot(data[:,i],data[:,j],'b.')    
        axs[i,j].plot(sample[:,i],sample[:,j],'r.')    
        axs[i,j].set_title(str(i+1)+","+str(j+1))


plt.figure()
# plt.plot(var1,var2,'.')
plt.plot(var1[:,0],var2[:,0],'.')
plt.plot(sample_vine[:,0],sample_vine[:,8],'.')




