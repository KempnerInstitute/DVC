import tensorflow as tf
import numpy as np

###################### NEAREST INTERPOLATION #####################

@tf.function(experimental_relax_shapes=True)
def nearestInterp2d(sample_s, pro_s1, pro_s2, pd_grid_uv):
    # Nearest neighbor interpolation on the grid
    len_sample = tf.shape(sample_s[:,0])[0]
    len_grid = tf.shape(pro_s1)[0]
    pro_s1_tile = tf.tile(pro_s1,[len_sample])
    pro_s1_tile = tf.transpose(tf.reshape(pro_s1_tile,[len_sample,len_grid])) 
    pro_s2_tile = tf.tile(pro_s2,[len_sample])
    pro_s2_tile = tf.transpose(tf.reshape(pro_s2_tile,[len_sample,len_grid])) 
    
    sample_s1_tile = tf.tile(sample_s[:,0],[len_grid])
    sample_s1_tile = tf.reshape(sample_s1_tile,[len_grid,len_sample]) 
    sample_s2_tile = tf.tile(sample_s[:,1],[len_grid])
    sample_s2_tile = tf.reshape(sample_s2_tile,[len_grid,len_sample])
    
    xi = tf.math.argmin(tf.abs(pro_s1_tile-sample_s1_tile),0)
    yi = tf.math.argmin(tf.abs(pro_s2_tile-sample_s2_tile),0)

    xi = xi[...,tf.newaxis]
    yi = yi[...,tf.newaxis]

    ind_int = tf.concat([xi,yi],1)
    inter =  tf.gather_nd(pd_grid_uv,ind_int)
    return inter

######################### LINEAR INTERPOLATION NUMPY #################

@tf.function(experimental_relax_shapes=True)
def interp1d_np(x,xref,yref):
    # 1-D linear interpolation python function
    y = tf.numpy_function(np.interp, [x,xref,yref], tf.float64) #np.interp(x,xref,yref)
    y = tf.cast(y,x.dtype)
    return y


@tf.function(experimental_relax_shapes=True)
def interp1d_np_Jan24(x, xp, fp):
    """
    Perform linear interpolation on 1D data.

    Args:
    x (Tensor): The x-coordinates at which to evaluate the interpolated values.
    xp (Tensor): The x-coordinates of the data points, must be increasing.
    fp (Tensor): The y-coordinates of the data points, same length as xp.

    Returns:
    Tensor: Interpolated values corresponding to x.
    """
    n = tf.shape(fp)[0]
    i = tf.clip_by_value(tf.searchsorted(xp, x) - 1, 0, n - 2)
    dx = tf.gather(xp, i + 1) - tf.gather(xp, i)
    dy = tf.gather(fp, i + 1) - tf.gather(fp, i)

    # Calculate the slope
    slope = dy / dx

    # Perform linear interpolation
    output = tf.gather(fp, i) + slope * (x - tf.gather(xp, i))

    return output
