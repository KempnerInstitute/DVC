import tensorflow as tf
import math as m
from utils.tensor_op import replace_nan_inf

########################## COMPUTE LOCAL LIKELIHOOD ################################

@tf.function(experimental_relax_shapes=True)
def loclik_batch(B, data, grid_points, n_cop, batch_size):
    
    ker_grid1 = tf.TensorArray(data.dtype,size=batch_size)
    ker_grid2 = tf.TensorArray(data.dtype,size=batch_size)
    ker_grid3 = tf.TensorArray(data.dtype,size=batch_size)
    ker_grid4 = tf.TensorArray(data.dtype,size=batch_size)
    ker_grid5 = tf.TensorArray(data.dtype,size=batch_size)

    batch_len = tf.shape(grid_points)[0]/batch_size
    batch_len = tf.cast(batch_len,tf.int32)

    for i in tf.range(0,batch_size,1):
        pp1,pp2,pp3,pp4,pp5 = dense_naive_batch(B,data,grid_points[batch_len*i:batch_len*(i+1),:,:])
        ker_grid1 = ker_grid1.write(i, pp1)
        ker_grid2 = ker_grid2.write(i, pp2)
        ker_grid3 = ker_grid3.write(i, pp3)
        ker_grid4 = ker_grid4.write(i, pp4)
        ker_grid5 = ker_grid5.write(i, pp5)

    ker_grid1 = ker_grid1.stack()
    ker_grid2 = ker_grid2.stack()
    ker_grid3 = ker_grid3.stack()
    ker_grid4 = ker_grid4.stack()
    ker_grid5 = ker_grid5.stack()
    ker_grid1 = tf.reshape(ker_grid1,[tf.shape(grid_points)[0],n_cop])
    ker_grid2 = tf.reshape(ker_grid2,[tf.shape(grid_points)[0],n_cop])
    ker_grid3 = tf.reshape(ker_grid3,[tf.shape(grid_points)[0],n_cop])
    ker_grid4 = tf.reshape(ker_grid4,[tf.shape(grid_points)[0],n_cop])
    ker_grid5 = tf.reshape(ker_grid5,[tf.shape(grid_points)[0],n_cop])
    ker_grid_fin = kern_LL(B, ker_grid1,ker_grid2,ker_grid3,ker_grid4,ker_grid5)
#     ker_grid_fin = tf.cast(ker_grid_fin,tf_dtype)
    return ker_grid_fin

@tf.function(experimental_relax_shapes=True)
def loclik_batch_eval(B, data, grid_points, n_cop, batch_size):
    tf_dtype = B.dtype
    B = tf.cast(B,tf.float64)
    data = tf.cast(data,tf.float64)
    grid_points = tf.cast(grid_points,tf.float64)
    
    ker_grid1 = tf.TensorArray(data.dtype,size=batch_size)
    ker_grid2 = tf.TensorArray(data.dtype,size=batch_size)
    ker_grid3 = tf.TensorArray(data.dtype,size=batch_size)
    ker_grid4 = tf.TensorArray(data.dtype,size=batch_size)
    ker_grid5 = tf.TensorArray(data.dtype,size=batch_size)

    batch_len = tf.shape(grid_points)[0]/batch_size
    batch_len = tf.cast(batch_len,tf.int32)

    for i in tf.range(0,batch_size,1):
        pp1,pp2,pp3,pp4,pp5 = dense_naive_batch(B,data,grid_points[batch_len*i:batch_len*(i+1),:,:])
        ker_grid1 = ker_grid1.write(i, pp1)
        ker_grid2 = ker_grid2.write(i, pp2)
        ker_grid3 = ker_grid3.write(i, pp3)
        ker_grid4 = ker_grid4.write(i, pp4)
        ker_grid5 = ker_grid5.write(i, pp5)

    ker_grid1 = ker_grid1.stack()
    ker_grid2 = ker_grid2.stack()
    ker_grid3 = ker_grid3.stack()
    ker_grid4 = ker_grid4.stack()
    ker_grid5 = ker_grid5.stack()
    ker_grid1 = tf.reshape(ker_grid1,[tf.shape(grid_points)[0],n_cop])
    ker_grid2 = tf.reshape(ker_grid2,[tf.shape(grid_points)[0],n_cop])
    ker_grid3 = tf.reshape(ker_grid3,[tf.shape(grid_points)[0],n_cop])
    ker_grid4 = tf.reshape(ker_grid4,[tf.shape(grid_points)[0],n_cop])
    ker_grid5 = tf.reshape(ker_grid5,[tf.shape(grid_points)[0],n_cop])
    ker_grid_fin = kern_LL(B, ker_grid1,ker_grid2,ker_grid3,ker_grid4,ker_grid5)
    ker_grid_fin = tf.cast(ker_grid_fin,tf_dtype)
    return ker_grid_fin

@tf.function(experimental_relax_shapes=True)
def kern_LL(B, ker_grid1,ker_grid2,ker_grid3,ker_grid4,ker_grid5):
    e1 = B[0,:] * tf.math.sqrt(tf.abs((ker_grid4/ker_grid1) - (ker_grid2/ker_grid1)**2))
    e2 = B[1,:] * tf.math.sqrt(tf.abs((ker_grid5/ker_grid1) - (ker_grid3/ker_grid1)**2))
    
    e1 = replace_nan_inf(e1)
    e2 = replace_nan_inf(e2)
    
    C = -e1**2 * ((ker_grid2/ker_grid1)**2 / (2*B[0,:]**2)) -e2**2 * ((ker_grid3/ker_grid1)**2 / (2*B[1,:]**2))
    
    C = replace_nan_inf(C)
    
    ker_grid_fin = ker_grid1 *e1 * e2 * tf.exp(C)
    return ker_grid_fin

@tf.function(experimental_relax_shapes=True)
def dense_naive_batch(B, data_p, grid_point):
    
    d = tf.shape(data_p)[2]
    d_n = tf.shape(data_p)[0]
    
    gr1_tile = tf.reshape(grid_point,[1, tf.shape(grid_point)[0],  tf.shape(grid_point)[1], d])
    gr1_tile = tf.tile(gr1_tile,[tf.shape(data_p)[0], 1, 1, 1])
    
    d1_tile = tf.reshape(data_p,[tf.shape(data_p)[0], 1,  tf.shape(grid_point)[1], d])
    d1_tile = tf.tile(d1_tile,[1, tf.shape(grid_point)[0], 1, 1])
    
    c = gr1_tile - d1_tile
    
    d_n = tf.cast(d_n,data_p.dtype)  #64
    pi = tf.cast(m.pi,data_p.dtype)  #64

    a = tf.exp(-(c[:,:,0,:]**2) / (2*B[0,:]**2)) * tf.exp(-(c[:,:,1,:]**2) / (2*B[1,:]**2)) / (2*pi*B[0,:]*B[1,:]*d_n) 
    
    ker_grid1 = tf.math.reduce_sum(a, 0)
    ker_grid2 = tf.math.reduce_sum(a*c[:,:,0,:], 0)
    ker_grid3 = tf.math.reduce_sum(a*c[:,:,1,:], 0)
    ker_grid4 = tf.math.reduce_sum(a*c[:,:,0,:]**2, 0)
    ker_grid5 = tf.math.reduce_sum(a*c[:,:,1,:]**2, 0)
    return ker_grid1,ker_grid2,ker_grid3,ker_grid4,ker_grid5 #ker_grid_fin