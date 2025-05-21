import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from utils.tensor_op import *
import math as m
import numpy as np
from scipy.interpolate import interp1d

@tf.function
def biv_norm(x1_s, x2_s):    
    norm = tfd.Normal(loc=tf.constant(0,dtype=x1_s.dtype), scale=tf.constant(1,dtype=x1_s.dtype))
    
    P1 = norm.prob(x1_s)[...,tf.newaxis]
    P2 = norm.prob(x2_s)[...,tf.newaxis]

    NORM = tf.transpose(P1*tf.transpose(P2))
    del P1,P2,x1_s,x2_s,norm
    return NORM

@tf.function(experimental_relax_shapes=True)
def op_cdf(data,margin_s_exc):
    data_ti = tf.tile(data[...,tf.newaxis],[1,tf.shape(margin_s_exc)[0]])  #[batch_len*i:batch_len*(i+1)]
    margin_s_ti = tf.transpose(tf.tile(margin_s_exc[...,tf.newaxis],[1,tf.shape(data)[0]]))
    dif1 = margin_s_ti - data_ti 
    dif1 = tf.maximum(dif1,0)
    dif1 = tf.math.sign(dif1)
    kka1 = tf.math.reduce_sum(dif1,0)
    kka1 = tf.cast(kka1,tf.int32)
    return kka1

@tf.function(experimental_relax_shapes=True)
def kernel_cdf_batch_Jan24(data, y, ex, batch_size): #changed from this to the new one -Houman
    
    margin_s = tf.sort(data)
    margin_s, idx = tf.unique(margin_s)
    
    exc = tf.shape(data)[0] - tf.shape(margin_s)[0]
    margin_s_exc = tf.concat([margin_s,tf.zeros(exc,data.dtype)],0)
    
    kka = tf.TensorArray(tf.int32,size=batch_size)

    batch_len = tf.shape(margin_s_exc)[0]/batch_size
    batch_len = tf.cast(batch_len,tf.int32)

    for i in tf.range(0,batch_size,1):
        data_ti = tf.tile(data[...,tf.newaxis],[1,tf.shape(margin_s_exc[batch_len*i:batch_len*(i+1)])[0]])  #[batch_len*i:batch_len*(i+1)]
        margin_s_ti = tf.transpose(tf.tile(margin_s_exc[batch_len*i:batch_len*(i+1)][...,tf.newaxis],[1,tf.shape(data)[0]]))
        dif1 = margin_s_ti - data_ti 
        dif1 = tf.maximum(dif1,0)
        dif1 = tf.math.sign(dif1)
        kka1 = tf.math.reduce_sum(dif1,0)
        kka1 = tf.cast(kka1,tf.int32)
        kka = kka.write(i,kka1)

    kka = kka.stack()
    kka = tf.reshape(kka,[-1])
    if tf.math.greater(exc,0):
        kka = kka[:-exc]
    
    margin_p = kka/(tf.shape(data)[0]+1)
    margin_p = tf.cast(margin_p,data.dtype)
    interp_cdf = interp1d_np(y, margin_s, margin_p)
    
    interp_cdf = constraints_bound(interp_cdf,ex)
    interp_cdf = check_bound_and_nan(interp_cdf,tf.math.reduce_max(margin_s),tf.math.reduce_min(margin_s))
    
    return interp_cdf, margin_s, margin_p

@tf.function(experimental_relax_shapes=True)
def kernel_cdf_batch(data, y, ex, batch_size):
    margin_s = tf.sort(data)
    margin_s, _ = tf.unique(margin_s)
    
    # Calculate the batch length and handle data types
    batch_len = tf.cast(tf.shape(margin_s)[0] / batch_size, tf.int32)

    kka = tf.TensorArray(dtype=margin_s.dtype, size=batch_size)

    for i in tf.range(0, batch_size, 1):
        start_idx = batch_len * i
        end_idx = batch_len * (i + 1)

        data_ti = tf.tile(data[..., tf.newaxis], [1, batch_len])
        margin_s_ti = tf.transpose(tf.tile(margin_s[start_idx:end_idx][..., tf.newaxis], [1, tf.shape(data)[0]]))

        dif1 = margin_s_ti - data_ti
        dif1 = tf.maximum(dif1, 0)
        dif1 = tf.math.sign(dif1)
        kka1 = tf.math.reduce_sum(dif1, axis=0)
        
        kka = kka.write(i, kka1)

    kka = kka.stack()
    kka = tf.reshape(kka, [-1])

    # Handle extra elements if any
    extra_elements = tf.shape(data)[0] % batch_size
    if extra_elements != 0:
        kka = kka[:-extra_elements]

    margin_p = kka / (tf.shape(data)[0] + 1)
    margin_p = tf.cast(margin_p, data.dtype)

    interp_cdf = interp1d_np(y, margin_s, margin_p)
    interp_cdf = constraints_bound(interp_cdf, ex)
    interp_cdf = check_bound_and_nan(interp_cdf, tf.math.reduce_max(margin_s), tf.math.reduce_min(margin_s))

    return interp_cdf, margin_s, margin_p


@tf.function(experimental_relax_shapes=True)
def kernel_cdf(data, y, ex):
    margin_s = tf.sort(data)
    margin_s, idx = tf.unique(margin_s)
    data_ti = tf.tile(data[...,tf.newaxis],[1,tf.shape(margin_s)[0]])
    dif1 = margin_s - data_ti 
    dif1 = tf.maximum(dif1,0)
    dif1 = tf.math.sign(dif1)
    kka = tf.math.reduce_sum(dif1,0)
    kka = tf.cast(kka,tf.int32) + 1   #Important because otherwise it does not take into account its own value and they are all shifted of 1
    margin_p = kka/(tf.shape(data)[0]+1)
    margin_p = tf.cast(margin_p,data.dtype)
    
    
    interp_cdf = interp1d_np(y, margin_s, margin_p)
    
    interp_cdf = constraints_bound(interp_cdf,ex)
    interp_cdf = check_bound_and_nan(interp_cdf,tf.math.reduce_max(margin_s),tf.math.reduce_min(margin_s))
    
    return interp_cdf, margin_s, margin_p



#@tf.function
def kernel_pdf2(x):  
    # Kernel density estimation 
    x_ker = x
    iscont = tf.constant(1,tf.int32)
    density = tf.zeros(128,x.dtype)
    mesh = tf.zeros(128,x.dtype)
    if tf.math.logical_not(tf.math.reduce_any(tf.math.less(x_ker,0))):
        ## BIMODAL DISTRIBUTION
        indpp1 = tf.where(tf.math.less(x_ker,1e-6))
        indpp2 = tf.where(tf.math.greater_equal(x_ker,1e-6))
        if tf.math.logical_and(tf.math.greater(tf.size(indpp1),1),tf.math.greater(tf.size(indpp2),1)):
            pow1 = tf.squeeze(tf.gather(x_ker,indpp1))
            pow2 = tf.squeeze(tf.gather(x_ker,indpp2))

            #den2, mden2 = kde(pow1,128,tf.math.reduce_min(pow1), tf.math.reduce_max(pow1)+2e-16)
            
            max_pow1 = tf.math.reduce_max(pow1)
            min_pow1 = tf.math.reduce_min(pow1)
            p_uni = 1/(max_pow1-min_pow1)
            den2 = tf.tile([p_uni],[128])

            R = max_pow1 + 2e-16 - min_pow1
            mden2 = tf.linspace(tf.constant(0,pow1.dtype),R,128) + min_pow1
            
            
            den3, mden3 = kde(pow2,128,tf.math.reduce_min(pow2), tf.math.reduce_max(pow2)+2e-16)

            m_diff = mden2[1:] - mden2[:-1]
            m_diff = tf.concat([m_diff, tf.expand_dims(m_diff[-1], 0)], 0)

            norm = tf.math.reduce_sum(den2*tf.transpose(m_diff),0)
            den2 = den2/norm

            m_diff = mden3[1:] - mden3[:-1]
            m_diff = tf.concat([m_diff, tf.expand_dims(m_diff[-1], 0)], 0)

            norm = tf.math.reduce_sum(den3*tf.transpose(m_diff),0)
            den3 = den3/norm

            SM = tf.linspace(tf.math.reduce_max(mden2)+1e-6, tf.math.reduce_min(mden3)-1e-6, 100)
            mesh = tf.concat([mden2, SM, mden3],0)

            part1 = tf.cast(tf.size(indpp1)/tf.shape(x_ker)[0],x_ker.dtype)
            part2 = tf.cast(tf.size(indpp2)/tf.shape(x_ker)[0],x_ker.dtype)
            density = tf.concat([den2*part1, tf.zeros(100,x_ker.dtype), den3*part2],0)
        else:
            if tf.equal(iscont,1):
                #density, mesh = kde(x_ker,128,MIN, MAX) #-1e-10,MAX+1e-10)   PROBLEMI SHAPES 128-127
                density, mesh = kde(x_ker,128,tf.math.reduce_min(x_ker), tf.math.reduce_max(x_ker))
            else:
                density, mesh = kde(x_ker,128,tf.math.reduce_min(x_ker), tf.math.reduce_max(x_ker))
            m_diff = mesh[1:] - mesh[:-1]
            m_diff = tf.concat([m_diff, tf.expand_dims(m_diff[-1], 0)], 0)
            area = tf.math.reduce_sum(density*tf.transpose(m_diff),0)
            density = density/area
    else:
        if tf.equal(iscont,1):
            density, mesh = kde(x_ker,128,tf.math.reduce_min(x_ker), tf.math.reduce_max(x_ker))
        else:
            density, mesh = kde(x_ker,128,tf.math.reduce_min(x_ker), tf.math.reduce_max(x_ker))
        m_diff = mesh[1:] - mesh[:-1]
        m_diff = tf.concat([m_diff, tf.expand_dims(m_diff[-1], 0)], 0)
        area = tf.math.reduce_sum(density*tf.transpose(m_diff),0)
        density = density/area
    return density, mesh

######## FUNCTION USED IN THE KDE:
# fixed_point: Function to evaluate best point, when it is equal zero.
# dct1d: Discrete cosine transform 1-D
# dct1d: Inverse discrete cosine transform 1-D
# histc: Python function to count how many times a value goes into a predefined intervals
# histc1: Tensorflow interface for histc function
# kde: Main kernel density estimation

@tf.function
def fixed_point_Jan24(xx,N,I,a2):
        # Ir represents function t-zeta*gamma^[l](t)

    dtype = a2.dtype  # Assuming a2's data type is the desired type for all calculations
    xx = tf.cast(xx, dtype)
    N = tf.cast(N, dtype)
    I = tf.cast(I, dtype)

    pi = tf.cast(m.pi, dtype)
    l = tf.constant(7, dtype=dtype)

    f = 2*tf.pow(pi,2*l)*tf.reduce_sum(tf.pow(I, l)*a2*tf.exp(-I*tf.square(pi)*xx)) 


    for i in tf.range(l-1, 1, -1):
        i = tf.cast(i, dtype=a2.dtype)  #added
        K0 = tf.reduce_prod(tf.range(1,2*i,2,dtype = a2.dtype))/tf.sqrt(2*pi)   #32
        const = (1+tf.pow(tf.constant(1/2, dtype=a2.dtype),i+1/2))/3
        time = tf.pow(2*const*K0/N/f,2/(3+2*i))
        f = 2*pi**(2*i)*tf.reduce_sum((I**i)*a2*tf.exp(-I*tf.square(pi)*time))   #tf.pow(pi,2*i)

    out = xx - tf.pow(2*N*tf.sqrt(pi)*f,-2/5);

    return out

@tf.function
def dct1d_Jan24(data):   # changed -Houman 
        typo = data.dtype
        pi = tf.cast(m.pi,typo)
        # Discrete cosine transform 1-D
        nrows = tf.shape(data)[0]
    
        #data = tf.cast(data,dtype =tf.float64)    
    
        nrows = tf.cast(nrows, dtype =typo)  #32
        #pp1 = 2*(tf.exp(-j*tf.range(1,nrows-1)*pi/(2*nrows)))
        pp1 = -tf.range(1,nrows)*pi/(2*nrows)
    
        imag = tf.dtypes.complex(tf.constant(0,dtype=typo),pp1)  #32
        imag = tf.cast(imag,dtype=tf.complex128)
        ww = 2 * tf.exp(imag)
        ww1 = tf.constant(1, dtype=tf.complex128, shape=[1])   #64
        weight = tf.concat([ww1,ww],0)

        ind = tf.range(0, tf.shape(data)[0],2, dtype=tf.int32)
        ind1 = tf.range(tf.shape(data)[0]-1,0,-2, dtype=tf.int32)

        a11 = tf.gather(data, ind)
        a22 = tf.gather(data, ind1)

        data = tf.concat([a11,a22],0)
        data = tf.cast(data,dtype = tf.complex128)  #64

        ff = tf.signal.fft(data)

        nnn1 = tf.math.real(weight * ff);
        nnn1 = tf.cast(nnn1,typo)
        return nnn1

@tf.function
def dct1d_Jan24(data):
    data_complex = tf.cast(data, dtype=tf.complex64)
    return tf.signal.dct(data_complex, type=2, norm='ortho')


@tf.function
def idct1d_Jan24(data):
        typo = data.dtype
        pi = tf.cast(m.pi,typo)
        # Inverse Discrete cosine transform 1-D
        nrows =  tf.shape(data)[0]
        data = tf.cast(data,dtype =tf.complex128)  
        nrows = tf.cast(nrows, dtype =typo)  #32

        pp1 = tf.range(0,nrows,dtype=typo)*pi/(2*nrows)
        imag = tf.dtypes.complex(tf.constant(0,dtype=typo),pp1)
        imag = tf.cast(imag,dtype=tf.complex128)
        nrow = tf.cast(nrows, dtype=tf.complex128)
        weight = nrow * tf.exp(imag)
    
        ddd1 = weight*data
        data = tf.math.real(tf.signal.ifft(ddd1))
        data = tf.cast(data,typo)
        
        ar1 = tf.range(0,nrows/2)
        ar2 = tf.range(nrows-1,nrows/2-1,-1)
        ar1 = tf.cast(ar1, dtype =tf.int32)
        ar2 = tf.cast(ar2, dtype =tf.int32)
        ar1 = ar1[...,tf.newaxis]
        ar2 = ar2[...,tf.newaxis]
        arr = tf.concat([ar1,ar2],1)
        arr1 = tf.reshape(arr,[-1])
    
        out = tf.gather(data, arr1)
        return out
    
def histc(X, bins):
        map_to_bins = np.digitize(X,bins)
        r = np.zeros(bins.shape)
        for i in map_to_bins:
            r[i-1] += 1
        return [r, map_to_bins]
    
@tf.function
def histc1_Jan24(X, bins): # changed -Houman
        map_to_bins1 = tf.py_function(np.digitize, [X,bins], tf.int32)
        def cond(i, u):
            return tf.math.less(i, tf.shape(bins)[0]+tf.constant(1,tf.int32))  #len(xmesh)

        def body(i, u):
            count = tf.where(tf.equal(map_to_bins1,i))
            u1 = tf.shape(count)[0]#tf.constant([len(count)],dtype=tf.int32)
            u1 = u1[...,tf.newaxis]
            u = tf.concat([u,u1],0)
            i = tf.add(i,tf.constant(1,tf.int32))
            return i,u

        u = tf.zeros([1],dtype=tf.int32)
        i = tf.constant(0,dtype=tf.int32)
        out_map = tf.while_loop(cond, body, [i, u], [i.get_shape(), tf.TensorShape([None])])
        
        return out_map[1][2:]

def histc1(X, bins):
    bin_indices = tf.histogram_fixed_width_bins(X, [tf.reduce_min(bins), tf.reduce_max(bins)], nbins=len(bins)-1)
    counts = tf.math.bincount(bin_indices, minlength=len(bins)-1, maxlength=len(bins)-1, dtype=tf.int32)
    return counts


"""
@tf.function
def kde(data,N,MIN,MAX):
    
    pi = tf.cast(m.pi,data.dtype)
    R =  MAX-MIN
    
    nbins = N
    dx = R/(nbins-1); 
    #xmesh = MIN + tf.range(0,R+dx,dx)
    xmesh = tf.linspace(tf.constant(0,data.dtype),R,128) + MIN
    
    N, idx1 = tf.unique(data)
    N = tf.cast(tf.math.ceil((tf.shape(data)[0]-1)/2)*2,dtype=tf.int32)

    #provas1,provas2 = tf.py_function(histc, [data, xmesh], tf.float64)  #tf.numpy_function
    provas1 = histc1(data, xmesh)
    provas1 = tf.cast(provas1,dtype=data.dtype)
    init_data = provas1/tf.cast(N,dtype=data.dtype)
    init_data = init_data/tf.reduce_sum(init_data)
    
    #pi = tf.cast(pi,dtype=tf.float64)
    #a = dct1d(init_data)
    a = dct1d(init_data)
    I = tf.square(tf.range(1,128,1,dtype = data.dtype))
    a2 = tf.square(a[1:]/2)
    N = tf.cast(N,dtype = data.dtype)

    tol = 1e-12 + 0.01*(N-50)/1000;
    t_star = tfp.math.secant_root(objective_fn = lambda t: fixed_point(t,N,I,a2), initial_position = tol)

    a_1 = tf.exp(- (tf.range(0,nbins, dtype=data.dtype)**2* (pi**2)*t_star.estimated_root)/2) 

    a_t = a * a_1
    density = idct1d(a_t)/R
    if tf.math.reduce_any(tf.math.less(density,0)):
        eps = tf.constant(2.220446049250313e-16,density.dtype)
        density = replace_negative(density, eps)
        
    return density, xmesh
"""

@tf.function
def fit_den(err_trace, iter_err, pos_trace, t, N,I,a2, max_iter, convergence_tol, lr):
    #err_trace.assign(MISE33(pos_trace, bw, grid_u2, grid_s2, grid_x, data_x, data_x_train, data_s_test, n_cop, batch_size, NORM1))
    #a.assign(tf.random.uniform(shape=[], minval=1e-4, maxval=2, dtype=tf.float64))
    #lr = 0.1
    eps = 1e-6
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.02)
    err = tf.math.abs(fixed_point(pos_trace,N,I,a2)) #tf.constant(1.,tf.float64)
    ###print('err:',err.numpy())
    ###print('  ')
    ###print('band:',a.numpy())
    m = tf.constant(0,t.dtype)
    v = tf.constant(0,t.dtype)
    m_hat = tf.constant(0,t.dtype)
    v_hat = tf.constant(0,t.dtype)
    beta_1 = tf.constant(0.9,t.dtype)
    beta_2 = tf.constant(0.999,t.dtype)
    #print(tf.math.less(iter_err, max_iter))
    #print(tf.math.greater(tf.abs(err-err_trace),convergence_tol))
    while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.greater(tf.abs(err-err_trace),convergence_tol)):
        err_trace = err
        err = tf.math.abs(fixed_point(t,N,I,a2))
        grad = (err - err_trace)/(t-pos_trace)
        if tf.math.logical_or(tf.math.is_nan(grad),tf.math.is_inf(grad)):
            grad = tf.constant(-0.001,t.dtype)
        pos_trace= t
        #print('err:',err.numpy())
        ###print('grad:',grad.numpy())
        iter1 = tf.cast(iter_err,t.dtype)
        m = beta_1 * m + (1 - beta_1) * grad
        ###print('m',m.numpy())
        v = beta_2 * v + (1 - beta_2) * grad**2
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)
        ###print('diff:',diff.numpy())
        t = (t + diff)
        
        #if tf.math.greater(t,.1):
        #    t.assign(.1)
        #if tf.math.less(t,0):
        #    t.assign(1e-5) 

        ###print('   ')
        ###print('iter:',iter_err.numpy())
        ###print('band:',a.numpy())
        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))

        ###print(tf.math.greater(tf.abs(err-err_trace),convergence_tol).numpy())
    if tf.math.less(err_trace,err):
        t = pos_trace
    return t, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)

@tf.function
def fit_den1(N, I, a2):
    err_trace = tf.constant(1,dtype=a2.dtype)
    pos_trace = tf.constant(0.05,dtype=a2.dtype)
    iter_err = tf.constant(1,dtype=tf.int32)

    max_iter = tf.constant(100,tf.int32)
    convergence_tol = tf.constant(0.0000001,a2.dtype)
    lr = tf.constant(0.001,a2.dtype)
    t = pos_trace + lr

    t_star, opt2, opt3, opt4 = fit_den(err_trace, iter_err, pos_trace, t, N, I, a2, max_iter, convergence_tol, lr)
    if tf.math.less(t_star,0):
        t_star = tf.constant(1e-5,a2.dtype)
    return t_star

@tf.function
def kde_Jan24(data,N,MIN,MAX): # changes -Houman
    
    pi = tf.cast(m.pi,data.dtype)
    R =  MAX-MIN
    
    nbins = N
    dx = R/(nbins-1); 
    #xmesh = MIN + tf.range(0,R+dx,dx)
    xmesh = tf.linspace(tf.constant(0,data.dtype),R,128) + MIN
    
    N, idx1 = tf.unique(data)
    N = tf.cast(tf.math.ceil((tf.shape(data)[0]-1)/2)*2,dtype=tf.int32)

    #provas1,provas2 = tf.py_function(histc, [data, xmesh], tf.float64)  #tf.numpy_function
    provas1 = histc1(data, xmesh)
    provas1 = tf.cast(provas1,dtype=data.dtype)
    init_data = provas1/tf.cast(N,dtype=data.dtype)
    init_data = init_data/tf.reduce_sum(init_data)
    
    #pi = tf.cast(pi,dtype=tf.float64)
    #a = dct1d(init_data)
    a = dct1d(init_data)
    I = tf.square(tf.range(1,128,1,dtype = data.dtype))
    a2 = tf.square(a[1:]/2)
    N = tf.cast(N,dtype = data.dtype)
    
    tol = 1e-12 + 0.01*(N-50)/1000
    
    t_star = secant_root1(objective_fn = lambda t: fixed_point(t,N,I,a2), initial_position = tol, max_iterations=50)

    #t_star = tfp.math.secant_root(objective_fn = lambda t: fixed_point(t,N,I,a2), initial_position = tol, max_iterations=5)
    if tf.math.equal(t_star.num_iterations,50):
        t_star = fit_den1(N,I,a2)
        a_1 = tf.exp(- (tf.range(0,nbins, dtype=data.dtype)**2* (pi**2)*t_star)/2)
    else:
        a_1 = tf.exp(- (tf.range(0,nbins, dtype=data.dtype)**2* (pi**2)*t_star.estimated_root )/2)

    a_t = a * a_1
    density = idct1d(a_t)/R
    if tf.math.reduce_any(tf.math.less(density,0)):
        eps = tf.constant(2.220446049250313e-16,density.dtype)
        density = replace_negative(density, eps)
    
    return density, xmesh




##### these are new ones -Houman
def dct1d(tensor):
    # Performs the discrete cosine transform of the tensor
    n = tf.shape(tensor)[0]
    if tensor.dtype == tf.float64:
        complex_dtype = tf.complex128
    else:
        complex_dtype = tf.complex64

    tensor = tf.concat([tensor, tf.reverse(tensor[1:n-1], axis=[0])], 0)
    result = tf.signal.fft(tf.cast(tensor, complex_dtype))
    result = tf.math.real(result)
    return result[:n]

def idct1d(tensor):
    # Performs the inverse discrete cosine transform
    n = tf.shape(tensor)[0]
    if tensor.dtype == tf.float64:
        complex_dtype = tf.complex128
    else:
        complex_dtype = tf.complex64    
    tensor = tf.concat([tensor, tf.reverse(tensor[1:], axis=[0])], 0)
    result = tf.signal.ifft(tf.cast(tensor, complex_dtype))
    result = tf.math.real(result)
    return result[:n]

def kde(data, n=2**14, MIN=None, MAX=None):
    if MIN is None or MAX is None:
        min_val, max_val = tf.reduce_min(data), tf.reduce_max(data)
        Range = max_val - min_val
        MIN, MAX = min_val - Range / 2, max_val + Range / 2

    MIN = tf.cast(MIN, data.dtype)
    MAX = tf.cast(MAX, data.dtype)
    # Set up the grid over which the density estimate is computed
    R = MAX - MIN
    dx = R / (n - 1)
    xmesh = MIN + dx * tf.cast(tf.range(n), data.dtype)

    N = tf.size(tf.unique(data)[0])

    # Bin the data uniformly using the defined grid
    initial_data = tf.histogram_fixed_width(data, [MIN, MAX], nbins=n)
    initial_data = initial_data / tf.reduce_sum(initial_data)
    a = tf.cast(dct1d(initial_data), data.dtype)  # Discrete cosine transform of initial data

    # Optimal bandwidth selection
    I = tf.cast(tf.range(1, n), data.dtype) ** 2
    a2 = tf.cast((a[1:] / 2) ** 2, data.dtype)

    # Define the fixed-point equation
    def fixed_point(t, N, I, a2):
        l = 7
        t = tf.cast(t, a2.dtype)
        N = tf.cast(N, a2.dtype)
        I = tf.cast(I, a2.dtype)
        #a2 = tf.cast(a2, a2.dtype)
        f = 2 * np.pi ** (2 * l) * tf.reduce_sum(I ** l * a2 * tf.exp(-I * np.pi ** 2 * t))
        for s in tf.range(l - 1, 1, -1):
            s = tf.cast(s, a2.dtype)
            pi_tf = tf.constant(np.pi, dtype=a2.dtype)
            K0 = tf.exp(tf.math.lgamma(s + 1) - tf.math.lgamma(s / 2 + 1) - 0.5 * tf.math.log(2 * pi_tf))

            #K0 = tf.exp(tf.math.lgamma(s + 1) - tf.math.lgamma(s / 2 + 1) - 0.5 * tf.math.log(2 * np.pi))           
            const = (1 + (0.5 ** (s + 0.5))) / 3
            time = (2 * const * K0 / N / f) ** (2 / (3 + 2 * s))
            f = 2 * np.pi ** (2 * s) * tf.reduce_sum(I ** s * a2 * tf.exp(-I * np.pi ** 2 * time))
        out = t - (2 * N * np.sqrt(np.pi) * f) ** (-2 / 5)
        return out

    # Find the root of the fixed-point equation
    t_starT = tfp.math.find_root_chandrupatla(
        objective_fn=lambda t: fixed_point(t, N, I, a2),
        low=tf.constant(0.0, dtype=a2.dtype), 
        high=tf.constant(1.0, dtype=a2.dtype)
    )
    t_star = tf.cast(t_starT.estimated_root, a2.dtype)

    # Smooth the DCT of initial data using t_star
    a_t = a * tf.exp(-tf.range(n, dtype=a2.dtype) ** 2 * tf.constant(np.pi, dtype=a2.dtype) ** 2 * t_star / 2)
    # Apply the inverse DCT
    density = idct1d(tf.cast(a_t, R.dtype)) / R

    # Bandwidth
    # bandwidth = tf.sqrt(t_star) * R

    return density, xmesh

######### INTERPOLATION

@tf.function
def interp1d_np(x,xref,yref):
    # 1-D linear interpolation python function
    y = tf.numpy_function(np.interp, [x,xref,yref], tf.float64) #np.interp(x,xref,yref)
    y = tf.cast(y,x.dtype)
    return y

@tf.function
def interp1d_near(x,xref,yref):
    # Nearest neighbor interpolation, it uses a python function
    def inter(x,xref,yref):
        f1 = interp1d(xref, yref, kind='nearest')
        y = f1(x)
        return y
    y = tf.numpy_function(inter, [x,xref,yref], x.dtype) #np.interp(x,xref,yref)
    return y

@tf.function
def interp1d_lin(x,xref,yref):
    # Linear interpolation, it uses a python function
    def inter(x,xref,yref):
        f1 = interp1d(xref, yref, kind='linear')
        y = f1(x)
        return y
    y = tf.numpy_function(inter, [x,xref,yref], x.dtype) #np.interp(x,xref,yref)
    return y

@tf.function
def interp_pdf(x,xref,yref):
    # Interpolate the pdf on the given reference based on nearest neighbor
    x = check_bound(x,xref)
    inter = interp1d_lin(x, xref, yref)
    return inter

@tf.function
def interp_pdf_near(x,xref,yref):
    # Interpolate the pdf on the given reference based on nearest neighbor
    x = check_bound(x,xref)
    inter = interp1d_near(x, xref, yref)
    return inter

#############################################################################################

############################## BUG TENSORFLOW  ##############################################
### Added to is_finished  (num_iterations>=max_iterations)

import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import dtype_util

RootSearchResults = collections.namedtuple(
    'RootSearchResults',
    [
        # A tensor containing the last position explored. If the search was
        # successful, this position is a root of the objective function.
        'estimated_root',
        # A tensor containing the value of the objective function at the last
        # position explored. If the search was successful, then this is close
        # to 0.
        'objective_at_estimated_root',
        # The number of iterations performed.
        'num_iterations',
    ])

@tf.function
def secant_root1(objective_fn,
                initial_position,
                next_position=None,
                value_at_position=None,
                position_tolerance=1e-8,
                value_tolerance=1e-8,
                max_iterations=50,
                stopping_policy_fn=tf.reduce_all,
                validate_args=False,
                name=None):
    r"""Finds root(s) of a function of single variable using the secant method.
    The [secant method](https://en.wikipedia.org/wiki/Secant_method) is a
    root-finding algorithm that uses a succession of roots of secant lines to
    better approximate a root of a function. The secant method can be thought of
    as a finite-difference approximation of Newton's method.
    Args:
    objective_fn: Python callable for which roots are searched. It must be a
      callable of a single variable. `objective_fn` must return a `Tensor` of
      the same shape and dtype as `initial_position`.
    initial_position: `Tensor` or Python float representing the starting
      position. The function will search for roots in the neighborhood of each
      point. The shape of `initial_position` should match that of the input to
      `objective_fn`.
    next_position: Optional `Tensor` representing the next position in the
      search. If specified, this argument must broadcast with the shape of
      `initial_position` and have the same dtype. It will be used to compute the
      first step to take when searching for roots. If not specified, a default
      value will be used instead.
      Default value: `initial_position * (1 + 1e-4) + sign(initial_position) *
        1e-4`.
    value_at_position: Optional `Tensor` or Pyhon float representing the value
      of `objective_fn` at `initial_position`. If specified, this argument must
      have the same shape and dtype as `initial_position`. If not specified, the
      value will be evaluated during the search.
      Default value: None.
    position_tolerance: Optional `Tensor` representing the tolerance for the
      estimated roots. If specified, this argument must broadcast with the shape
      of `initial_position` and have the same dtype.
      Default value: `1e-8`.
    value_tolerance: Optional `Tensor` representing the tolerance used to check
      for roots. If the absolute value of `objective_fn` is smaller than
      `value_tolerance` at a given position, then that position is considered a
      root for the function. If specified, this argument must broadcast with the
      shape of `initial_position` and have the same dtype.
      Default value: `1e-8`.
    max_iterations: Optional `Tensor` or Python integer specifying the maximum
      number of steps to perform for each initial position. Must broadcast with
      the shape of `initial_position`.
      Default value: `50`.
    stopping_policy_fn: Python `callable` controlling the algorithm termination.
      It must be a callable accepting a `Tensor` of booleans with the shape of
      `initial_position` (each denoting whether the search is finished for each
      starting point), and returning a scalar boolean `Tensor` (indicating
      whether the overall search should stop). Typical values are
      `tf.reduce_all` (which returns only when the search is finished for all
      points), and `tf.reduce_any` (which returns as soon as the search is
      finished for any point).
      Default value: `tf.reduce_all` (returns only when the search is finished
        for all points).
    validate_args: Python `bool` indicating whether to validate arguments such
      as `position_tolerance`, `value_tolerance`, and `max_iterations`.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
    Returns:
    root_search_results: A Python `namedtuple` containing the following items:
      estimated_root: `Tensor` containing the last position explored. If the
        search was successful within the specified tolerance, this position is
        a root of the objective function.
      objective_at_estimated_root: `Tensor` containing the value of the
        objective function at `position`. If the search was successful within
        the specified tolerance, then this is close to 0.
      num_iterations: The number of iterations performed.
    Raises:
    ValueError: if a non-callable `stopping_policy_fn` is passed.
    #### Examples
    ```python
    import tensorflow as tf
    import tensorflow_probability as tfp
    tf.enable_eager_execution()
    # Example 1: Roots of a single function from two different starting points.
    f = lambda x: (63 * x**5 - 70 * x**3 + 15 * x) / 8.
    x = tf.constant([-1, 10], dtype=tf.float64)
    tfp.math.secant_root(objective_fn=f, initial_position=x))
    # ==> RootSearchResults(
      estimated_root=array([-0.90617985, 0.90617985]),
      objective_at_estimated_root=array([-4.81727769e-10, 7.44957651e-10]),
      num_iterations=array([ 7, 24], dtype=int32))
    tfp.math.secant_root(objective_fn=f,
                       initial_position=x,
                       stopping_policy_fn=tf.reduce_any)
    # ==> RootSearchResults(
      estimated_root=array([-0.90617985, 3.27379206]),
      objective_at_estimated_root=array([-4.81727769e-10, 2.66058312e+03]),
      num_iterations=array([7, 8], dtype=int32))
    # Example 2: Roots of a multiplex function from a single starting point.
    def f(x):
    return tf.constant([0., 63. / 8], dtype=tf.float64) * x**5 \
        + tf.constant([5. / 2, -70. / 8], dtype=tf.float64) * x**3 \
        + tf.constant([-3. / 2, 15. / 8], dtype=tf.float64) * x
    x = tf.constant([-1, -1], dtype=tf.float64)
    tfp.math.secant_root(objective_fn=f, initial_position=x)
    # ==> RootSearchResults(
      estimated_root=array([-0.77459667, -0.90617985]),
      objective_at_estimated_root=array([-7.81339438e-11, -4.81727769e-10]),
      num_iterations=array([7, 7], dtype=int32))
    # Example 3: Roots of a multiplex function from two starting points.
    def f(x):
    return tf.constant([0., 63. / 8], dtype=tf.float64) * x**5 \
        + tf.constant([5. / 2, -70. / 8], dtype=tf.float64) * x**3 \
        + tf.constant([-3. / 2, 15. / 8], dtype=tf.float64) * x
    x = tf.constant([[-1, -1], [10, 10]], dtype=tf.float64)
    tfp.math.secant_root(objective_fn=f, initial_position=x)
    # ==> RootSearchResults(
      estimated_root=array([
          [-0.77459667, -0.90617985],
          [ 0.77459667, 0.90617985]]),
      objective_at_estimated_root=array([
          [-7.81339438e-11, -4.81727769e-10],
          [6.66025013e-11, 7.44957651e-10]]),
      num_iterations=array([
          [7, 7],
          [16, 24]], dtype=int32))
    ```
    """
    
    if not callable(stopping_policy_fn):
        raise ValueError('stopping_policy_fn must be callable')


    position = tf.convert_to_tensor(
          initial_position,
          name='position',
          )

    
    value_at_position = tf.convert_to_tensor(
    value_at_position or objective_fn(position),
    name='value_at_position',
    dtype=dtype_util.base_dtype(position.dtype))

    zero = tf.zeros_like(position)
    position_tolerance = tf.convert_to_tensor(
      position_tolerance, name='position_tolerance', dtype=position.dtype)
    value_tolerance = tf.convert_to_tensor(
          value_tolerance, name='value_tolerance', dtype=position.dtype)

    num_iterations = tf.zeros_like(position, dtype=tf.int32)
    max_iterations = tf.convert_to_tensor(max_iterations, dtype=tf.int32)
    max_iterations = tf.broadcast_to(
        max_iterations, name='max_iterations', shape=position.shape)

    # Compute the step from `next_position` if present. This covers the case where
    # a user has two starting points, which bound the root or has a specific step
    # size in mind.
    if next_position is None:
        epsilon = tf.constant(1e-4, dtype=position.dtype, shape=position.shape)
        step = position * epsilon + tf.sign(position) * epsilon
    else:
        step = next_position - initial_position

    finished = tf.constant(False, shape=position.shape)

    # Negate `stopping_condition` to determine if the search should continue.
    # This means, in particular, that tf.reduce_*all* will return only when the
    # search is finished for *all* starting points.
    def _should_continue(position, value_at_position, num_iterations, step,
                       finished):
        """Indicates whether the overall search should continue.
        Args:
          position: `Tensor` containing the current root estimates.
          value_at_position: `Tensor` containing the value of `objective_fn` at
            `position`.
          num_iterations: `Tensor` containing the current iteration index for each
            point.
          step: `Tensor` containing the size of the step to take for each point.
          finished: `Tensor` indicating for which points the search is finished.
        Returns:
          A boolean value indicating whether the overall search should continue.
        """
        #del position, value_at_position, num_iterations, step  # Unused
        return ~tf.convert_to_tensor(
            stopping_policy_fn(finished), name='should_stop', dtype=tf.bool)

    # For each point in `position`, the search is stopped if either:
    # (1) A root has been found
    # (2) f(position) == f(position + step)
    # (3) The maximum number of iterations has been reached
    # In case (2), the search may be stopped both before the desired tolerance is
    # achieved (or even a root is found), and the maximum number of iterations is
    # reached.
    def _body(position, value_at_position, num_iterations, step, finished):
        """Performs one iteration of the secant root-finding algorithm.
        Args:
          position: `Tensor` containing the current root estimates.
          value_at_position: `Tensor` containing the value of `objective_fn` at
            `position`.
          num_iterations: `Tensor` containing the current iteration index for each
            point.
          step: `Tensor` containing the size of the step to take for each point.
          finished: `Tensor` indicating for which points the search is finished.
        Returns:
          The `Tensor`s to use for the next iteration of the algorithm.
        """

        # True if the search was already finished, or (1) or (3) just became true.
        was_finished = finished | (num_iterations >= max_iterations) | (
            tf.abs(step) < position_tolerance) | (
                tf.abs(value_at_position) < value_tolerance)
        # Compute the next position and the value at that point.
        next_position = tf.where(was_finished, position, position + step)
        value_at_next_position = tf.where(was_finished, value_at_position,
                                          objective_fn(next_position))
        # True if the search was already finished, or (2) just became true.
        is_finished = tf.equal(value_at_position, value_at_next_position) | (num_iterations >= max_iterations)
        # Use the mid-point between the last two positions if (2) just became true.
        next_position = tf.where(is_finished & ~was_finished,
                                 (position + next_position) * 0.5, next_position)
        # Once finished, stop updating the iteration index and set the step to zero.
        num_iterations = tf.where(is_finished, num_iterations, num_iterations + 1)
        next_step = tf.where(
            is_finished, zero, step * value_at_next_position /
            (value_at_position - value_at_next_position))
        #if tf.math.equal(was_finished,True):
        #    is_finished = True

        return (next_position, value_at_next_position, num_iterations, next_step,
                is_finished)

    with tf.name_scope(name or 'secant_root'):

        assertions = []
        if validate_args:
              assertions += [
              tf.Assert(
                  tf.reduce_all(position_tolerance > zero), [position_tolerance]),
              tf.Assert(tf.reduce_all(value_tolerance > zero), [value_tolerance]),
              tf.Assert(
                  tf.reduce_all(max_iterations >= num_iterations),
                  [max_iterations]),
          ]

    with tf.control_dependencies(assertions):
          root, value_at_root, num_iterations, _, _ = tf.while_loop(
              cond=_should_continue,
              body=_body,
              loop_vars=[
                  position, value_at_position, num_iterations, step, finished
              ])

    return RootSearchResults(
              estimated_root=root,
              objective_at_estimated_root=value_at_root,
              num_iterations=num_iterations)