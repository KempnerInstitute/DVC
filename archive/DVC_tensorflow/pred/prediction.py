import tensorflow as tf
import numpy as np
from utils.tensor_op import moving_average
from utils.tensor_op import create_points
from utils.tensor_op import replace_nan_inf

################## PREDICT VINE ########################

def predict_vine(x, vine, dim, exp_dim):

    points = create_points(x,dim,exp_dim)
    
    p, p_cop, logp = vine.evaluation(points)
    
    p1 = tf.reshape(p,[x.shape[0],exp_dim])
    p1 = replace_nan_inf(p1)
    
    min_dim = tf.math.reduce_min(x[:,dim])
    max_dim = tf.math.reduce_max(x[:,dim])
    y_vec = tf.linspace(min_dim-2e-16+1e-5,max_dim+2e-16,exp_dim)
    
    y_vec = tf.cast(y_vec,tf.float64)
    mov_p = tf.TensorArray(p1.dtype, size=tf.shape(p1)[0])
    for i in tf.range(0,tf.shape(p1)[0],1,tf.int32): 
        movag = smooth(p1[i,:].numpy(),4,'flat')
        mov_p = mov_p.write(i,movag)

    mov_p = mov_p.stack()
    mov_p = mov_p[:,3:]
    
    ############### Y MAXIMUM LIKELIHOOD  ##################

    y_diff = y_vec[1:] - y_vec[:-1]
    y_diff = tf.concat([y_diff, tf.expand_dims(y_diff[-1], 0)], 0)

    ind_max1 = tf.math.argmax(mov_p,1) 
    ind_max1 = ind_max1[...,tf.newaxis]
    y_ml = tf.gather_nd(y_vec,ind_max1)

    ############### Y EXPECTATION MAXIMIZATION  ##################

    y_diff1 = y_diff[...,tf.newaxis]
    y_diff_tile = tf.tile(y_diff1, [1, tf.shape(p1)[0]])

    q1 = tf.math.reduce_sum(mov_p*tf.transpose(y_diff_tile),1)
    q2 = q1[...,tf.newaxis]
    q1 = tf.tile(q2,[1,tf.shape(p1)[1]])
    q = mov_p/q1

    y_tmp = y_vec*y_diff
    y_tmp1 = y_tmp[...,tf.newaxis]
    y_tmp1 = tf.tile(y_tmp1,[1,tf.shape(p1)[0]])

    y_em = tf.math.reduce_sum(q*tf.transpose(y_tmp1),1)
    
    return p, y_ml, y_em

###################  PREDICT RESPONSE   ######################

# @tf.function
def predict_response(p1, y_vec):
    y_vec = tf.cast(y_vec,y_vec.dtype)
    mov_p = tf.TensorArray(p1.dtype, size=tf.shape(p1)[0])
    for i in tf.range(0,tf.shape(p1)[0],1,tf.int32):
        movag = smooth(p1[i,:].numpy(),4,'flat')
        mov_p = mov_p.write(i,movag)

    mov_p = mov_p.stack()
    mov_p = mov_p[:,3:]
    
    ############### Y MAXIMUM LIKELIHOOD  ##################

    y_diff = y_vec[1:] - y_vec[:-1]
    y_diff = tf.concat([y_diff, tf.expand_dims(y_diff[-1], 0)], 0)

    ind_max1 = tf.math.argmax(mov_p,1) 
    ind_max1 = ind_max1[...,tf.newaxis]
    y_ml = tf.gather_nd(y_vec,ind_max1)

    ############### Y EXPECTATION MAXIMIZATION  ##################

    y_diff1 = y_diff[...,tf.newaxis]
    y_diff_tile = tf.tile(y_diff1, [1, tf.shape(p1)[0]])

    q1 = tf.math.reduce_sum(mov_p*tf.transpose(y_diff_tile),1)
    q2 = q1[...,tf.newaxis]
    q1 = tf.tile(q2,[1,tf.shape(p1)[1]])
    q = mov_p/q1

    y_tmp = y_vec*y_diff
    y_tmp1 = y_tmp[...,tf.newaxis]
    y_tmp1 = tf.tile(y_tmp1,[1,tf.shape(p1)[0]])

    y_em = tf.math.reduce_sum(q*tf.transpose(y_tmp1),1)
    
    return y_ml, y_em


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y