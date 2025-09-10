import tensorflow as tf

############################ BANDWIDTH RULE THUMB ######################

# Rule thumb for bandwidth
@tf.function(experimental_relax_shapes=True)
def bandwidth_mul(data, deg, n_cop):
    # Rule of thumb for computing the bandwidth
    n = tf.cast(tf.shape(data)[0],data.dtype)
    
    xc = data - tf.math.reduce_sum(data,0)/n 
    if tf.math.equal(tf.shape(tf.shape(xc)),2):
        xc = xc[...,tf.newaxis]
        
    chol = tf.TensorArray(data.dtype, size = n_cop)
    for jj in tf.range(0,n_cop,1,tf.int32):
        c1 = tf.tensordot(tf.transpose(xc[:,:,jj]),xc[:,:,jj],1) / (n-1)
        chol1 = tf.transpose(tf.linalg.cholesky(c1))
        chol = chol.write(jj,chol1)
    chol = chol.stack()
    chol = tf.transpose(chol,perm=[1,2,0])

    bw = 5 * n**(-1 / (4 * deg + 2)) * chol

    bw1 = bw[0,0,:]
    bw2 = bw[1,1,:]
    bw = tf.concat([bw1,bw2],0)
    bw = tf.reshape(bw,[2,n_cop])
    bw = bw/10
    return bw