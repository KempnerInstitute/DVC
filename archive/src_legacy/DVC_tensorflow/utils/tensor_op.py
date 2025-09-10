import tensorflow as tf

#################### CHECK BOUNDARIES OF TENSOR ########################

@tf.function(experimental_relax_shapes=True)
def check_bound(data,mesh):
    # Clips tensor value to its minimum and maximum
    #data = tf.cast(data, tf.float32)
    mesh = tf.cast(mesh, data.dtype)

    # Other operations...

    max_m = tf.math.reduce_max(mesh)
    min_m = tf.math.reduce_min(mesh)
    ind_max = tf.where(tf.math.greater(data,max_m))
    ind_min = tf.where(tf.math.less(data,min_m))
    upd_min = tf.tile([min_m],[ tf.shape(ind_min)[0]])
    upd_max = tf.tile([max_m],[ tf.shape(ind_max)[0]])
    data = tf.tensor_scatter_nd_update(data, ind_min, upd_min)
    data = tf.tensor_scatter_nd_update(data, ind_max, upd_max)
    return data

# @tf.function
def check_bound3(data,maxx,minn):
    # Replace nan and inf
    ind_max = tf.where(tf.math.greater_equal(data,maxx))
    ind_min = tf.where(tf.math.less_equal(data,minn))
    upd_max = tf.tile([maxx-tf.constant(1e-10,data.dtype)],[tf.shape(ind_max)[0]])
    upd_min = tf.tile([minn+tf.constant(1e-10,data.dtype)],[tf.shape(ind_min)[0]])
    data = tf.tensor_scatter_nd_update(data, ind_max, upd_max)
    data = tf.tensor_scatter_nd_update(data, ind_min, upd_min)
    return data

def constraints_bound(data,mesh):
    # Clips tensor value to its minimum and maximum

    
    max_m = tf.math.reduce_max(mesh)
    min_m = tf.math.reduce_min(mesh)
    max_m = tf.cast(max_m, data.dtype)
    min_m = tf.cast(min_m, data.dtype)

    ind_max = tf.where(tf.math.greater(data,max_m))
    ind_min = tf.where(tf.math.less(data,min_m))
    
    upd_max = max_m*(1-1e-10*tf.random.normal([tf.shape(ind_max)[0]], mean=0.0, stddev=1.0, dtype=data.dtype))
    upd_min = min_m*(1+1e-10*tf.random.normal([tf.shape(ind_min)[0]], mean=0.0, stddev=1.0, dtype=data.dtype))
    
    data = tf.tensor_scatter_nd_update(data, ind_min, upd_min)
    data = tf.tensor_scatter_nd_update(data, ind_max, upd_max)
    return data

def check_bound_and_nan(data,maxx,minn):
    # Replace nan and inf
    ind_max = tf.where(tf.math.greater_equal(data,maxx))
    ind_min = tf.where(tf.math.less_equal(data,minn))
    ind_nan = tf.where(tf.math.is_nan(data))
    
    upd_max = maxx*(1-1e-10*tf.random.normal([tf.shape(ind_max)[0]], mean=0.0, stddev=1.0, dtype=data.dtype))
    upd_min = minn*(1+1e-10*tf.random.normal([tf.shape(ind_min)[0]], mean=0.0, stddev=1.0, dtype=data.dtype))
    upd_nan = minn*(1+1e-10*tf.random.normal([tf.shape(ind_nan)[0]], mean=0.0, stddev=1.0, dtype=data.dtype))
    
    data = tf.tensor_scatter_nd_update(data, ind_max, upd_max)
    data = tf.tensor_scatter_nd_update(data, ind_min, upd_min)
    data = tf.tensor_scatter_nd_update(data, ind_nan, upd_nan)
    return data

##################### UNIQUE TENSORS #################################

@tf.function
def uniquetol(data,tol):
    # Return if unique values are all below a given tolerance
    y,ii = tf.unique(data)
    d = tf.abs(y[1:] - y[:-1])
    check =tf.math.greater(d,tol)
    isTol = tf.concat([tf.constant(True,dtype=tf.bool,shape=[1]),check],0)
    z = tf.boolean_mask(y,isTol)
    return z

#################### UPDATE TENSORS 2D ################################

#@tf.function
def update_tensor2D(tensor, i , newval):
    cases = tf.shape(tensor)[0]
    ind1 = tf.range(0,cases,1)[...,tf.newaxis]
    ind2 = tf.tile([i],[cases])[...,tf.newaxis]

    ind = tf.concat([ind1,ind2],1)

    tensor = tf.tensor_scatter_nd_update(tensor, ind, newval)
    return tensor


################### UPDATE TENSOR 3D ################################

# @tf.function(experimental_relax_shapes=True)
def update_tensor(tensor,newval,i,j):
    cases = tf.shape(tensor)[0]
    ind1 = tf.range(0,cases,1)[...,tf.newaxis]
    ind2 = tf.tile([i],[cases])[...,tf.newaxis]
    ind3 = tf.tile([j],[cases])[...,tf.newaxis]

    ind = tf.concat([ind1,ind2,ind3],1)

    tensor = tf.tensor_scatter_nd_update(tensor, ind, newval)
    return tensor


################ REPLACE NEGATIVE/INF OR NAN ###########################


#@tf.function
def replace_inf(data, newval):
    # Replace negative to eps and inf to maximum
    ind_inf = tf.where(tf.math.logical_and(tf.math.less(data,0),tf.math.is_inf(data)))
    upd_inf = tf.tile([newval],[tf.shape(ind_inf)[0]])
    data = tf.tensor_scatter_nd_update(data, ind_inf, upd_inf)
    return data

#@tf.function
def replace_negative(data, newval):
    # Replace negative to eps
    ind_nan = tf.where(tf.math.less(data,0))
    upd_nan = tf.tile([newval],[tf.shape(ind_nan)[0]])
    data = tf.tensor_scatter_nd_update(data, ind_nan, upd_nan)
    return data

# @tf.function
def replace_nan_inf(data):
    # Replace nan and inf
    ind_nan = tf.where(tf.math.is_nan(data))
    ind_inf = tf.where(tf.math.is_inf(data))
    upd_nan = tf.tile([tf.constant(0,data.dtype)],[tf.shape(ind_nan)[0]])
    upd_inf = tf.tile([tf.constant(data.dtype.max,data.dtype)],[tf.shape(ind_inf)[0]])
    data = tf.tensor_scatter_nd_update(data, ind_inf, upd_inf)
    data = tf.tensor_scatter_nd_update(data, ind_nan, upd_nan)
    return data

# @tf.function
def replace_nan_with(data,newval):
    # Replace nan and inf
    ind_nan = tf.where(tf.math.is_nan(data))
    upd_nan = tf.tile([newval],[tf.shape(ind_nan)[0]])
    data = tf.tensor_scatter_nd_update(data, ind_nan, upd_nan)
    return data

def replace_inf_with(data,newval):
    # Replace inf
    ind_inf = tf.where(tf.math.is_inf(data))
    upd_inf = tf.tile([newval],[tf.shape(ind_inf)[0]])
    data = tf.tensor_scatter_nd_update(data, ind_inf, upd_inf)
    return data

############################ MOVING AVERAGE ###############################

@tf.function
def moving_average(a, n):
    smoothed = a
    orig = a
    ret = tf.math.cumsum(a)
    ind1 = tf.range(0,tf.shape(a)[0],1,tf.int32)
    ind1 = ind1[...,tf.newaxis]
    ret1 = ret[:n]
    ret2 = ret[n:] - ret[:-n]
    ret_new = tf.concat([ret1,ret2],0)
    smoothed = tf.tensor_scatter_nd_update(smoothed, ind1, ret_new)
    n1 = tf.cast(n,a.dtype)
    smoothed1 = smoothed[n - 1:] / n1
    smoothed2 = tf.concat([a[:n-1],smoothed1],0)
    return smoothed2


########################### Extend tensor from one dimension #################

@tf.function
def create_points(x, dim, exp_dim):
    eps = tf.constant(1e-16,x.dtype)
    d = tf.shape(x)[1]
    min_dim = tf.math.reduce_min(x[:,dim])
    max_dim = tf.math.reduce_max(x[:,dim])
    y_vec = tf.linspace(min_dim-eps+1e-5,max_dim+eps,exp_dim)

    yy = tf.tile(y_vec,[tf.shape(x)[0]])

    vv = tf.TensorArray(x.dtype,size=d)
    for i in tf.range(0,d,1,tf.int32):
        vv1 = tf.tile(x[:,i][...,tf.newaxis],[1,tf.size(y_vec)])
        vv1 = tf.reshape(vv1,[-1])
        vv = vv.write(i,vv1)

    vv = vv.stack()
    #e = tf.reshape(e,[d,tf.shape(x)[0]])
    vv = tf.transpose(vv)
    points = update_tensor2D(vv, dim, yy)
    return points