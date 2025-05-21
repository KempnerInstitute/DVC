import tensorflow as tf
import numpy as np
import sys
import tensorflow_probability as tfp
tfd = tfp.distributions
from utils.tensor_op import *

def prep_cop(x, vine1, sort_n):
    d = x.shape[1]
    if sort_n == 'sort':
        corr = tf.TensorArray(x.dtype,size=d*d)
        for i in tf.range(0,d,1):
            for j in tf.range(0,d,1):
                corr1 = tf.py_function(stats.kendalltau, [x[:,i],x[:,j]], x.dtype)
                corr1 = tf.math.abs(corr1)
                corr.write(3*i+j, corr1)
        corr = corr.stack()
        corr = tf.reshape(corr,[3,3])
        ord1 = tf.constant([0],tf.int32)
        for i in tf.range(1,d-3,-1):
            # Difference between 0:d e 0
            #ord1 = tf.reshape(ord1,[1,tf.shape(ord1)[0]])
            ss = tf.sets.difference(tf.constant(tf.range(0,d,1),shape=[1,d]),tf.constant(ord1,shape=[1,tf.shape(ord1)[0]]))
            #ss = tf.sets.difference(tf.constant(tf.range(0,d,1),shape=[1,d]),ord1)
            ss = tf.sparse.to_dense(ss)
            ss = tf.transpose(ss)
            
            # Index of the difference to take in the correlation
            pp1 = tf.tile(ord1[i-1][...,tf.newaxis], [tf.shape(ss)[0]])
            pp1 = pp1[...,tf.newaxis]
            pp2 = tf.concat([pp1,ss],1)
        
            # Index of the difference to take in the correlation
            prova = tf.gather_nd(corr, pp2) 
            ind_max = tf.math.argmax(prova)
            ord1 = tf.concat([ord1,ss[ind_max]],0)
        # Difference between 0:d e ord1
        ss1 = tf.sets.difference(tf.constant(tf.range(0,d,1),shape=[1,d]),tf.constant(ord1,shape=[1,tf.shape(ord1)[0]]))
        ss1 = tf.sparse.to_dense(ss1)
    
        ord1 = tf.concat([ord1,ss1[0]],0) 
    else:
        ord1 = range(0,d,1)
    
    # Change columns corresponding with the order
    x = x[:,ord1]
    
    e = np.empty(x.shape,x.dtype)
    for i in range(0,d,1):   #d
        x_new1 = tf.constant(x[:,i])
        e[:,i] = prep_copula(x_new1,0).numpy()
        vine1.margin[i].ker = e[:,i]
    
    # del x_new1
    return e
    
# @tf.function
def prep_copula(X_new,tim):
    # Preallocate
    u1 = tfd.Uniform(low=tf.constant(0,dtype=X_new.dtype), high=tf.constant(1,dtype=X_new.dtype))
    
    samples = u1.sample(tf.shape(X_new)[0])

    samples = tf.expand_dims(samples, 1)
    margin1 = tf.zeros(tf.shape(X_new),X_new.dtype)
    if tf.equal(tf.size(uniquetol(X_new,1e-5)),1):
        margin1 = X_new + samples[:, 0] * 1e-10
    else:
        ad1, idx = tf.unique(X_new)
        ad1 = tf.sort(ad1, axis=0)
        ad1 = ad1[...,tf.newaxis]

        # Calculate diff vector
        ad = ad1[1:] - ad1[:-1]
        ad = tf.concat([ad, tf.expand_dims(ad[-1, :], 0)], 0)

        # Compute distances
        """
        Di2 = tf.zeros(tf.shape(X_new)[0],X_new.dtype)
        for pro in tf.range(0,tf.shape(ad_add)[0],1,tf.int32):
            ind22 = tf.where(tf.equal(X_new, ad_add[pro]))
            ind22 = tf.cast(ind22, tf.int32) #X_new.dtype
            pp1 = ad[pro]
            pp1 = tf.tile(pp1,[tf.shape(ind22)[0]])
            Di2 = tf.tensor_scatter_nd_update(Di2, ind22, pp1)
        """
        
        batch_size = tf.constant(1,tf.int32)
        if tf.math.less(tf.shape(X_new)[0],500):
            batch_size = tf.constant(1,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],1000):
            batch_size = tf.constant(2,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],4000):
            batch_size = tf.constant(10,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],10000):
            batch_size = tf.constant(20,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],20000):
            batch_size = tf.constant(50,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],100000):
            batch_size = tf.constant(100,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],200000):
            batch_size = tf.constant(200,tf.int32)
        
        #Di2 = tf.zeros(1,X_new.dtype) 
        batch_len = tf.shape(X_new)[0]/batch_size
        batch_len = tf.cast(batch_len,tf.int32)
        Di2 = tf.zeros(tf.shape(X_new)[0],X_new.dtype)
        
        for j in tf.range(0,batch_size,1):
            x_batch = X_new[batch_len*j:batch_len*(j+1)]
            if tf.math.equal(j,batch_size-1):
                x_batch = X_new[batch_len*j:]
                
#             ind22 = tf.where(tf.equal(x_batch, ad_add))
            ind22 = tf.where(tf.equal(x_batch, ad1))
            ind22_1 = ind22[:,1][...,tf.newaxis]
            
            
#             ra = tf.range(0,tf.shape(ad_add)[0],1,tf.int32)
            ra = tf.range(0,tf.shape(ad1)[0],1,tf.int32)
            pro = ind22[:,0]
            pro = pro[...,tf.newaxis]

            pp1 = tf.gather_nd(ad,pro)
            Di2 = tf.tensor_scatter_nd_update(Di2, ind22_1, tf.squeeze(pp1))
            # Add uniform noise proportional to NN distance
            #Di2 = tf.concat([Di2,Dii],0)

        #Di2 = Di2[1:]
        if tim == 0:
            margin1 = X_new + Di2 * samples[:, 0] * 1e-10  #Di1
        if tim == 1:
            margin1 = X_new + Di2 * samples[:, 0]
    return margin1
