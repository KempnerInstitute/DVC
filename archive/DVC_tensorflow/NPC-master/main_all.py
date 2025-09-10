import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

########### Packages from the vine library

from .pre_proc.define_copulas import define_copulas
from param.generate_rvine import generate_r_samples
from pre_proc.preparation import prep_cop
from sampling.vine_sample import vine_copula_sample
from utils.tensor_op import create_points
from classes.objects import vine_obj_bin, margin_obj
from pred.prediction import predict_vine
from info.info_estimation import vine_entropy


################################ -  DEFINE THE VINE FOR GENERATING THE DATA - ####################################

#### Generate random matrix
cases = 1000        ### Number of samples
vine_type = 'c-vine' # or 'd-vine' or 'r-vine'
method = 'matrix'  # or 'r_matrix'  only with r-vine
binning = False
n_bin = 3
dim = 5                # Dimension of the vine for random r-vine or c-vine or d-vine

### Define copulas and vine. Please look at the function in ./pre_proc/define_copulas and change directly in there
r_matrix, cop_vine, ind_vine, nodes, matrix_edges, margin_vine = define_copulas(vine_type, method, binning, n_bin, dim)

# if binning == True:
#     exc = tf.math.floormod(cases,n_bin)
#     cases = cases - exc

sample, v, v_flip, tau_corr, tau_bins = generate_r_samples(cases, r_matrix, ind_vine, nodes, margin_vine, cop_vine, n_bin, binning)
print(sample)

## Plot one example of sample
plt.figure()
plt.plot(sample[:,0],sample[:,1],'.')
plt.show()

################################ -  LOAD MATLAB OR PICKLE FILE - ####################################

#  If you want to load a matlab file or a vine saved in a pickle file
load_mat = False
load_pickle = False #True

if load_mat:
    mat_contents = sio.loadmat('stu_ex_01.mat') #sim_vine #sim17_10000.mat')   #X_sim17.mat')
    # print(mat_contents)
    dat = mat_contents.get('x') #X_sim17
    dat = np.array(dat,np.float64)

    pdf_cop = mat_contents.get('pdf_cop') #X_sim17
    pdf_cop = np.array(pdf_cop,np.float32)
    
if load_pickle:
    pickle_in = open("clay_20_ale","rb")  #clay_20_ale
    dict_save = pickle.load(pickle_in)
    vine_copulas = dict_save["vine_copulas"]
    x = dict_save["data"] #x
    r_matrix = dict_save["r_matrix"]
    vine_depth = 20

x = sample
################################ -  DEFINE THE VINE FOR FITTING - ####################################

vine_type = "d-vine"
method = 'matrix' #'matrix' 'optimal'
families = "kercop"
knots = 50

vine_depth = len(r_matrix)

## Define the margins
margin_vine = []
for i in range(0,vine_depth,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine.append(mar_p)
    
vine = vine_obj_bin(vine_type, families, vine_depth, margin_vine, knots, method, r_matrix)

if load_pickle:
    vine.copulas = vine_copulas

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
n_bin = 3

### Make data divisible for bins and k-fold

x = np.array(x,np.float32)

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
print(e)

### FITTING
# Add parameters in a dictionary
tf.config.experimental_run_functions_eagerly(True)

gen_dict = {'parallel':True, 'binning':binning, 'param':param, 'vine_depth':vine_depth, 'fitted':False}  #vine_depth
par_dict = {'param_families':["ind","gaussian"]}  #["ind","gaussian","student","clayton","claytonrot90"]
npc_dict = {'opt_method':'LL1','batch_paral':3}
bin_dict = {'n_bin':n_bin}

save_vine = False

vine.fit(x,gen_dict,npc_dict,par_dict,bin_dict)

if save_vine:
    dict_save = {'vine_copulas': vine.copulas, 'r_matrix': vine.r_matrix, 'data': x}
    pickle_out = open("clay_20_ale","wb")
    pickle.dump(dict_save,pickle_out)
    pickle_out.close()

################################ -  VINE SAMPLING - ####################################

sample = vine_copula_sample(vine,2000)

################################ -  VINE EVALUATION - ####################################

### Create points for evaluation

exp_dim = 100
dim = 0
 
points = create_points(x,dim,exp_dim)
print(points)
print(type(points))

p, p_cop, plog = vine.evaluation(points)



################################ -  PREDICT VINE - ####################################
dim = 0
exp_dim = 100

p, y_ml, y_em = predict_vine(x,vine,dim,exp_dim)


## To remove this, check why a NaN is generated. One point only gives this problem, probably at the boundary
print(np.where(np.isnan(y_ml)))
print(np.where(np.isnan(y_em)))

#### IF NAN
from utils.tensor_op import replace_nan_inf

y_em = replace_nan_inf(y_em)

## Compute correlation and plot

from scipy import stats
corr = stats.pearsonr(x[:,dim], y_em)

print(corr[0])

plt.figure()
plt.plot(x[:,dim], y_ml, 'r.')
plt.plot(x[:,dim], y_em, 'b.')
plt.title('Correlation: ' + str(corr[0]))
plt.show()

################################ -  MI ESTIMATION - ####################################
info_dict = {'cases':1000, 'iterations':10, 'alpha': 0.05}
MI = vine_entropy(vine,info_dict)
print(MI)