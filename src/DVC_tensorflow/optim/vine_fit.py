import tensorflow as tf
from utils.prob_op import biv_norm
from optim.bandwidth import bandwidth_mul
from optim.nadam import fit_ban
from optim.nadam import fit_banLL2
from sklearn.model_selection import KFold
from time import perf_counter
from utils.dataset_op import *
from param.copula_fit import *

########################## NON-PARAMETRIC FITTING #############################

def optimization(grid_dict, data_dict,par_dict):

    grid_u = grid_dict['grid_u']
    grid_s = grid_dict['grid_s']
    grid_x = grid_dict['grid_x']
    adu11, adu22 = grid_u.diff()
    step_s = grid_s.step_grid()
    min_s = grid_s.min_grid()
    max_s = grid_s.max_grid()
    
    data_s = data_dict['data_s']
    data_x = data_dict['data_x']
    
    n_cop = par_dict['n_cop']
    batch = tf.convert_to_tensor(par_dict['batch'])
    max_iter = par_dict['max_iter']
    lr = tf.convert_to_tensor(par_dict['lr'],data_x.dtype)
    conv_tol = tf.convert_to_tensor(par_dict['conv_tol'],data_x.dtype)
    opt_method  = par_dict['opt_method']
    
    ## Bivariate normal
    x1_s, x2_s = grid_s.axis()
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM[...,tf.newaxis]
    NORM = tf.tile(NORM,[1, 1, n_cop])

    ## Compute bandwidth
    bw = bandwidth_mul(data_x,2,n_cop)
    
    ## Split data_x and data_s for CV
    train_ind, test_ind = kfold(data_x, 5)

    data_s_train = data_split(data_s,train_ind)
    data_s_test = data_split(data_s,test_ind)

    data_x_train = data_split(data_x,train_ind)
    data_x_test = data_split(data_x,test_ind)
        
    norm_flag = tf.constant(False,dtype=tf.bool)
    max_iter = tf.convert_to_tensor(max_iter)
    lr = tf.convert_to_tensor(lr)
    conv_tol = tf.convert_to_tensor(conv_tol)
    
    start_time = perf_counter()
    
    if opt_method == 'LL1':
        a = tf.random.uniform(shape=[n_cop], minval=1e-1, maxval=1.9, dtype=bw.dtype)
        pos_trace = tf.random.uniform(shape=[n_cop], minval=1e-1, maxval=1.9, dtype=bw.dtype)
        opt1, opt2, opt3, opt4 = fit_ban(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, 
                                          n_cop, batch, NORM, norm_flag, pos_trace,max_iter[0], conv_tol[0], lr[0])
    elif opt_method == 'LL2':
        a = tf.random.uniform(shape=[2,n_cop], minval=1e-1, maxval=1.9, dtype=bw.dtype)
        pos_trace = tf.random.uniform(shape=[2,n_cop], minval=1e-1, maxval=1.9, dtype=bw.dtype)
        opt1, opt2, opt3, opt4 = fit_banLL2(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, 
                                          n_cop, batch, NORM, norm_flag, pos_trace,max_iter[0], conv_tol[0], lr[0])
    

    optim1 = {'optim': opt1.numpy(), 'error': opt2.numpy(), 'num_iter': opt3.numpy(), 'Convergence': opt4.numpy()}
    
    #print('opt1',optim1)
    
    time_fit = perf_counter() - start_time
    print('time_fit:', time_fit)
    
    norm_flag = tf.constant(True,dtype=tf.bool)

    pos_trace = opt1    
    a = opt1 - lr[1]
    
    start_time = perf_counter()
    
    if opt_method == 'LL1':
        opt1, opt2, opt3, opt4 = fit_ban(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, 
                                      n_cop, batch, NORM, norm_flag, pos_trace,max_iter[1], conv_tol[1], lr[1])
    elif opt_method == 'LL2':
        opt1, opt2, opt3, opt4 = fit_banLL2(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, 
                                      n_cop, batch, NORM, norm_flag, pos_trace,max_iter[1], conv_tol[1], lr[1])
    

    optim2 = {'optim': opt1.numpy(), 'error': opt2.numpy(), 'num_iter': opt3.numpy(), 'Convergence': opt4.numpy()}
    #print('opt2',optim2)
    
    time_fit = perf_counter() - start_time
    #print('time_fit2:', time_fit)
    return opt1 #optim1, optim2

########################## PARAMETRIC FITTING #############################

def parametric_fit(u, families, n_cop):
    u = tf.convert_to_tensor(u)
    u = check_bound3(u,tf.constant(1-1e-7,u.dtype),tf.constant(-1+1e-7,u.dtype))

    theta = []
    logp = []
    aic = []
    for j in range(0,len(families),1):
        fam = families[j]
        
        if fam == 'ind':
            theta_est = []
            for i in range(0,n_cop,1):
                theta_est.append([])
            theta.append(theta_est)
            p = tf.constant([[1]],u.dtype)
            p = tf.tile(p,[1,n_cop])
            err = -tf.math.reduce_sum(tf.math.log(p),[0])
            err = err.numpy()
            logp.append(err)
            aic1 = 2*np.shape(theta_est)[0]+2*err
            aic.append(aic1)
        
        if fam == 'gaussian':
            pos_trace = tf.constant([0.5], dtype = u.dtype)
            pos_trace = tf.tile(pos_trace,[n_cop])
            lr = tf.constant(0.005,u.dtype)
            conv_tol = tf.constant(1e-3,u.dtype)
            max_iter = tf.constant(100,tf.int32)
            if np.shape(u)[2] > 1:
                max_iter = tf.constant(200,tf.int32)
            a = pos_trace + lr

            theta_est, err, n_iter, conv_flag = fit_gaussian(u, a, pos_trace, conv_tol, lr, max_iter, n_cop)
            theta_est = theta_est.numpy()
            err = err.numpy()
            n_iter = n_iter.numpy()
            conv_flag = conv_flag.numpy()
            theta.append(theta_est)
            logp.append(err)
            aic1 = 2*np.shape(theta_est)[0]+2*err
            aic.append(aic1)

        if fam == 'student':
            n_cop = n_cop.numpy()
            pos_trace = tf.constant([0.5,3], dtype = u.dtype)
            pos_trace = tf.tile(pos_trace,[n_cop])
            pos_trace = tf.reshape(pos_trace,[n_cop,2])
            lr = tf.constant(0.1,u.dtype)
            conv_tol = tf.constant(5e-1,u.dtype)
            max_iter = tf.constant(100,tf.int32)
            if np.shape(u)[2] > 1:
                max_iter = tf.constant(200,tf.int32)
            a = pos_trace + lr

            theta_est, err, n_iter, conv_flag = fit_student(u, a, pos_trace, conv_tol, lr, max_iter, n_cop)
            theta_est = theta_est.numpy()
            err = err.numpy()
            n_iter = n_iter.numpy()
            conv_flag = conv_flag.numpy()
            theta.append(theta_est)
            logp.append(err)
            aic1 = 2*np.shape(theta_est)[1]+2*err
            aic.append(aic1)

        if (fam == 'clayton') | (fam == 'claytonrot90'):
            pos_trace = tf.constant([3], dtype = u.dtype)
            pos_trace = tf.tile(pos_trace,[n_cop])
            lr = tf.constant(0.2,u.dtype)
            conv_tol = tf.constant(1e-3,u.dtype)
            max_iter = tf.constant(200,tf.int32)
            if np.shape(u)[2] > 1:
                max_iter = tf.constant(200,tf.int32)
            a = pos_trace + lr

            if fam == 'clayton':
                theta_est, err, n_iter, conv_flag = fit_clayton(u, a, pos_trace, conv_tol, lr, max_iter, n_cop)
            if fam == 'claytonrot90':
                theta_est, err, n_iter, conv_flag = fit_claytonrot90(u, a, pos_trace, conv_tol, lr, max_iter, n_cop)
            theta_est = theta_est.numpy()
            err = err.numpy()
            n_iter = n_iter.numpy()
            conv_flag = conv_flag.numpy()
            theta.append(theta_est)
            logp.append(err)
            aic1 = 2*np.shape(theta_est)[0]+2*err
    
            aic.append(aic1)
    
    aic2 = []
    theta2 = []
    logp2 = []
    for i in range(0,n_cop,1):
        aic22 = []
        theta22 = []
        logp22 = []
        for j in range(0,len(families),1):
            aic22.append(aic[j][i])
            theta22.append(theta[j][i])
            logp22.append(logp[j][i])
        aic2.append(aic22)
        theta2.append(theta22)
        logp2.append(logp22)
    return aic2, theta2, logp2
