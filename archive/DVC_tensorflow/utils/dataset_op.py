import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold

def kfold(data, n_splits):    
    train_ind = tf.TensorArray(tf.int32,size=n_splits)
    test_ind = tf.TensorArray(tf.int32,size=n_splits)
    k = 0
    kf = KFold(n_splits=n_splits, shuffle=True) # Instantiate the cross validator
    for train_index, test_index in kf.split(data): #[:,:,0]
        train_ind = train_ind.write(k,train_index)
        test_ind = test_ind.write(k,test_index)
        k = k+1

    train_ind = train_ind.stack()
    test_ind = test_ind.stack()
    return train_ind, test_ind

@tf.function(experimental_relax_shapes=True)
def data_split(data,ind):
    # DIVIDE DATA IN TRAINING AND TEST SET
    if tf.math.equal(tf.shape(tf.shape(data)),2):
        data = data[...,tf.newaxis]
    
    n_splits = tf.shape(ind)[0]
    data_new = tf.TensorArray(data.dtype,size=n_splits)
    for j in tf.range(0,n_splits,1,tf.int32):
        data_new1 = tf.gather_nd(data,ind[j,:][...,tf.newaxis])
        data_new = data_new.write(j,data_new1)

    data_new = data_new.stack()
    data_new = tf.transpose(data_new,perm=[1,2,3,0])
    return data_new


############### BIN FUNCTIONS

def create_bins(data, n_bin):
    len_bin = tf.shape(data)[0]/n_bin
    len_bin = tf.cast(len_bin,tf.int32)
    data = np.sort(data)
    bins = [data[0]-1e-15]
    for i in range(1,n_bin,1):
        bins.append(data[len_bin*i])
    bins.append(data[-1]+1e-15)
    return bins

def check_bins(data,bins):
    n_bin = np.shape(bins)[0] -1
    len_bin = np.shape(data)[0]/n_bin
    len_bin = tf.cast(len_bin,tf.int32)
    val_to_bin = np.digitize(data, bins) - 1
    
    ind_sort = np.argsort(data)
    
    val_to_bin2 = val_to_bin
    for bb in range(0,n_bin,1):
        val_to_bin2[ind_sort[bb*len_bin:(bb+1)*len_bin]] = bb*np.ones(len_bin,data.dtype)

    return val_to_bin2