import tensorflow as tf
from utils.tensor_op import *

class grid_obj(object):
    """Grid object.
    """
    def __init__(self, ex):
        """Create a grid object.
        Args:
            ex: Expanded grid.
        """
        self.ex = ex
        self.ax1 = None
        self.ax2 = None
        self.step = None
        self.min = None
        self.max = None
        self.diff1 = None
        self.diff2 = None
        
    def axis(self):
        # Compute axis of the grid
        self.ax1,idx1 = tf.unique(self.ex[:,0])
        self.ax2,idx2 = tf.unique(self.ex[:,1])
        del idx1,idx2
        return self.ax1,self.ax2

    def diff(self):
        # Compute diff vector of axis
        if not tf.is_tensor(self.ax1):
            #tf.math.reduce_any(tf.math.logical_not(self.ax1)):
            self.ax1, self.ax2 = self.axis()
        ad1 = self.ax1[1:] - self.ax1[:-1]
        self.diff1 = tf.concat([ad1, tf.expand_dims(ad1[-1], 0)], 0)
        ad2 = self.ax2[1:] - self.ax2[:-1]
        self.diff2 = tf.concat([ad2, tf.expand_dims(ad2[-1], 0)], 0)
        del ad1,ad2
        return self.diff1,self.diff2
    
    def step_grid(self):
        # Compute the step of the grid
        if not tf.is_tensor(self.diff1):
            #tf.math.reduce_any(tf.math.logical_not(self.diff1)):
            self.diff1, self.diff2 = self.diff()
        dx = (self.diff1, self.diff2)
        dx = tf.map_fn(lambda x: x[0]*x[1], dx, dtype=self.ex.dtype)
        dx1 = uniquetol(dx,1e-5)
        len_dx = tf.shape(dx1)[0]
        if tf.equal(tf.size(uniquetol(dx,1e-5)),1):
            #dx = uniquetol(dx,1e-5)
            self.step = dx1
        else:
            raise Exception('The grid is not uniform. There are different steps that exceeds tol: {}'.format(dx1))
        del dx, dx1, len_dx
        return self.step
    
    def min_grid(self):
        min1 = tf.math.reduce_min(self.ex[:,0],axis=0)
        min2 = tf.math.reduce_min(self.ex[:,1],axis=0)
        min1 = min1[...,tf.newaxis]
        min2 = min2[...,tf.newaxis]
        self.min = tf.concat([min1,min2],0)
        return self.min
    
    def max_grid(self):
        max1 = tf.math.reduce_max(self.ex[:,0],axis=0)
        max2 = tf.math.reduce_max(self.ex[:,1],axis=0)
        max1 = max1[...,tf.newaxis]
        max2 = max2[...,tf.newaxis]
        self.max = tf.concat([max1,max2],0)
        return self.max