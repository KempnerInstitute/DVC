import tensorflow as tf
from utils.bijector import *
from utils.tensor_op import check_bound3

class Transform(object):
    """Copula object.
    """
    def __init__(self,n_cop):
        """Create a Copula object.
        Args:
            data_u: Data in the (u,v) space.
            data_s: Data in the (s,r) space.
            data_x: Data rotated in the (s,r) space.
            points_u: Points in the (u,v) space.
            points_s: Points in the (s,r) space.
            points_x: Points rotated in the (s,r) space.
        """
        self.n_cop = n_cop
        self.mu = None
        self.coeff = None
        
    #@tf.function
    def forward_u(self,obj_u):
        #if tf.math.equal(obj_u, self.ex_u):
        loc = tf.constant(0,obj_u.dtype)
        scale = tf.constant(1,obj_u.dtype)
        obj_s = NormalCDF.forward(NormalCDF(loc, scale), obj_u)
        obj_s = check_bound3(obj_s,tf.constant(3.2,obj_u.dtype),tf.constant(-3.2,obj_u.dtype))
        return obj_s
    
    #@tf.function
    def forward_s(self,obj_s):
        # PCA
        if not tf.is_tensor(self.coeff):
            coeff = tf.TensorArray(obj_s.dtype, size = self.n_cop)
            for i in tf.range(0,self.n_cop,1,tf.int32):
                s,u,coeff1 = tf.linalg.svd(obj_s[:,:,i])
                coeff = coeff.write(i,coeff1)
            coeff = tf.transpose(coeff.stack(),perm=[1,2,0])   

            # Enforce to have positive maximum
            
            #coeff1 = tf.zeros([2,2,1],obj_s.dtype)
            coeff1 = tf.TensorArray(obj_s.dtype, size = self.n_cop)
            for i in tf.range(0,self.n_cop,1,tf.int32):                   # TAKE THE ROW OF FIST MAXIMUM AND CHANGE SIGN BASED ON THAT
                #ee, ind_p = tf.math.top_k(tf.abs(coeff[:,:,i]),1)
                ind_p = tf.math.argmax(coeff[:,:,i])
                max_val = tf.gather_nd(coeff[:,:,i],[ind_p[0]])
                sign_val = tf.math.sign(max_val)
                sign_val = sign_val[...,tf.newaxis]
                sign_val = tf.tile(sign_val,[tf.shape(sign_val)[0],1])
                sign_val = tf.reshape(sign_val,tf.shape(coeff[:,:,i]))
                coeff2 = sign_val*coeff[:,:,i]
                #coeff2 = coeff2[...,tf.newaxis]
                
                coeff1 = coeff1.write(i,coeff2)
                #coeff1 = tf.concat([coeff1,coeff2],2)
            #coeff = coeff1[:,:,1:]
            coeff1 = tf.transpose(coeff1.stack(),perm=[1,2,0])
            self.coeff = coeff1
            del coeff,coeff1,coeff2,ind_p,max_val,sign_val,s,u
        
        if not tf.is_tensor(self.mu):
            self.mu = tf.math.reduce_mean(obj_s,0)
            
        if tf.math.equal(tf.shape(tf.shape(obj_s)),2):
            obj_s = obj_s[...,tf.newaxis]
            obj_s = tf.tile(obj_s,[1,1,self.n_cop])
            
        mu1 = tf.tile(self.mu, [tf.shape(obj_s)[0], 1])
        mu1 = tf.reshape(mu1,tf.shape(obj_s)) #[1000,2]

        obj_x = tf.TensorArray(obj_s.dtype, size = self.n_cop)
        for i in tf.range(0,self.n_cop,1,tf.int32):
            data_x1 = tf.linalg.matmul(obj_s[:,:,i],self.coeff[:,:,i]) - tf.linalg.matmul(mu1[:,:,i],self.coeff[:,:,i])
            obj_x = obj_x.write(i,data_x1)
        obj_x = tf.transpose(obj_x.stack(),perm=[1,2,0])
        
#         elif tf.math.equal(tf.shape(tf.shape(obj_s)),2):
#             mu1 = tf.tile(self.mu[:,0],tf.constant(tf.shape(obj_s)[0],dtype=tf.int32,shape=[1]))
#             mu1 = tf.reshape(mu1,tf.shape(obj_s)) #[1000,2]
#             obj_x = tf.linalg.matmul(obj_s,self.coeff[:,:,0]) - tf.linalg.matmul(mu1,self.coeff[:,:,0])
            
        return obj_x