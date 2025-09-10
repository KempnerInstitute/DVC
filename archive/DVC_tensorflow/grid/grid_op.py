import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from utils.bijector import *

#@tf.function
def mk_grid(knots, dtype):
    """Create matrix grid and expanded grid:
    Args:
        knots: num. of knots of the grid.
    Returns:
        coordinates: matrix grid
        expanded: expanded grid
    """

    
    loc=tf.constant(0,dtype=dtype)
    scale=tf.constant(1,dtype=dtype)
    points = NormalCDF.inverse(NormalCDF(loc,scale), tf.linspace(tf.constant(-3.2,dtype=dtype), tf.constant(3.2,dtype=dtype), num=knots))
    x_grid, y_grid = tf.meshgrid(points, points)

    coordinates = tf.concat([tf.transpose(x_grid[0,:])[...,tf.newaxis], y_grid[:,0][...,tf.newaxis]], axis=1)
    x_grid1 = tf.reshape(x_grid, [-1])
    y_grid1 = tf.reshape(y_grid, [-1])
    expanded = tf.concat([x_grid1[..., tf.newaxis], y_grid1[..., tf.newaxis]], axis=1)
    return coordinates, expanded


"""
@tf.function
def mk_grid2(knots, dtype):
    
    tf.print("Inside mk_grid. Eager execution:", tf.executing_eagerly())

    loc = tf.constant(0, dtype=dtype)
    scale = tf.constant(1, dtype=dtype)
    
    normal_dist = tfp.distributions.Normal(loc=loc, scale=scale)
    #points = normal_dist.quantile(tf.linspace(tf.constant(-3.2, dtype=dtype), tf.constant(3.2, dtype=dtype), num=knots))
    points = normal_dist.quantile(tf.linspace(tf.constant(-3.2, dtype=dtype), tf.constant(3.2, dtype=dtype), num=tf.cast(knots, dtype=tf.int32)))

    x_grid, y_grid = tf.meshgrid(points, points)

    coordinates = tf.concat([tf.transpose(x_grid[0, :])[..., tf.newaxis], y_grid[:, 0][..., tf.newaxis]], axis=1)
    x_grid1 = tf.reshape(x_grid, [-1])
    y_grid1 = tf.reshape(y_grid, [-1])
    expanded = tf.concat([x_grid1[..., tf.newaxis], y_grid1[..., tf.newaxis]], axis=1)
    
    return coordinates, expanded


"""
 