# File: src/DVC_tensorflow/classes/.ipynb_checkpoints/objects-checkpoint.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from utils.tensor_op import *
from utils.prob_op import *
from utils.interpolation import *
from pre_proc.transformation import *
from grid.grid_op import *
from grid.grid_class import *
from evalu.cop_eval import *
from evalu.vine_eval import *
from vine_tree.tree_op import *
from optim.vine_fit import *
from param.cond_copula import *

## VINE OBJ BIN
from utils.dataset_op import create_bins

class copula_obj(object):
    """Copula object.
    """
    def __init__(self, opt_bw):
        """Create a copula object.
        Args:
            opt_bw: Optimal fitted bandwidth.
        """
        self.opt_bw = opt_bw
        self.pd_grid_uv = None
        self.cdf = None

class cop_par_obj(object):
    """Marginal object.
    """
    def __init__(self, family, theta):
        """Create a marginal object.
        Args:
            ker: Kernel of the marginal.
            family: Type of the marginal.
            min: Min of the range of the marginal.
            max: Max of the range of the marginal.
        """
        self.family = family
        self.theta = theta

class margin_obj(object):
    """Marginal object.
    """
    def __init__(self, dist, theta, is_cont):
        """Create a marginal object.
        Args:
            ker: Kernel of the marginal.
            family: Type of the marginal.
            min: Min of the range of the marginal.
            max: Max of the range of the marginal.
        """
        self.dist = dist
        self.theta = theta
        self.is_cont = is_cont
        self.ker = None

# class margin_obj(object):
#     """Marginal object.
#     """
#     def __init__(self, dist, iscont):
#         """Create a marginal object.
#         Args:
#             ker: Kernel of the marginal.
#             family: Type of the marginal.
#             min: Min of the range of the marginal.
#             max: Max of the range of the marginal.
#         """
#         self.dist = dist
#         self.theta1 = []
#         self.theta2 = []
#         self.iscont = iscont
#         self.ker = []

class vine_obj_bin(object):
    """Vine object.
    """
    def __init__(self, vine_family, families, vine_depth, margin, knots, *args):
        """Create a marginal object.
        Args:
            families: Copula family.
            theta: Correlation factor.
            margin: Margin of the copula.
        """
        self.vine_family = vine_family
        self.families = families
        self.theta1 = []
        self.theta2 = []
        self.rang = None
        self.n_cop = vine_depth
        self.margin = margin
        self.knots = knots
        
        self.ind_vine = []
        for i in range(0,self.n_cop-1,1):
            self.ind_vine.append([])
        
        
        if self.vine_family == 'r-vine':
            self.method = args[0]
            if self.method == 'matrix':
                self.r_matrix = args[1]
                self.E, self.ind_vine, self.nodes, self.matrix_edges = prepare_regular(self.r_matrix)
        
        if (self.vine_family == 'c-vine') | (self.vine_family == 'd-vine'):
            self.r_matrix, self.ind_vine, self.nodes, self.matrix_edges = prepare_vine(self.vine_family, self.n_cop)
                    
        self.Mar_G = None
        self.theta = None
        self.Fp = None
        self.logf = None
        self.copulas = None
        
        self.data_u = None
        self.data_s = None
        self.data_x = None
        
        self.points_u = None
        self.points_s = None
        self.points_x = None
        
        self.grid_u = None
        self.grid_s = None
        self.grid_x = None
        
        self.binning = False
        self.n_bin = 1
        
    def fit(self, x, gen_dict, npc_dict, par_dict, bin_dict):
        
        np_type = x.dtype
        x = tf.convert_to_tensor(x)

        ## Initialization
        self.binning = gen_dict['binning']
        self.parallel = gen_dict['parallel']
        self.param = gen_dict['param']
        self.fitted = gen_dict['fitted']
        
        self.vine_depth = gen_dict['vine_depth']

        d = x.shape[1]
        self.n_cop = d
        
        if self.param == False:
            self.opt_method = npc_dict['opt_method']
            batch_paral = npc_dict['batch_paral']
        else:
            param_families = par_dict['param_families']
        if self.binning == True:
            self.n_bin = bin_dict['n_bin']
        
        ## Batches
        batch_size_cdf = tf.constant(5,tf.int32)
        if np.shape(x)[0] > 5000:
            batch_size_cdf = tf.constant(10,tf.int32)
        elif np.shape(x)[0] > 10000:
            batch_size_cdf = tf.constant(100,tf.int32)
        elif np.shape(x)[0] > 50000:
            batch_size_cdf = tf.constant(200,tf.int32)
        elif np.shape(x)[0] > 100000:
            batch_size_cdf = tf.constant(500,tf.int32)
        elif np.shape(x)[0] > 200000:
            batch_size_cdf = tf.constant(1000,tf.int32)
        elif np.shape(x)[0] > 500000:
            batch_size_cdf = tf.constant(2000,tf.int32)
        
        ## Make grid
        u_1, ex_u = mk_grid(tf.convert_to_tensor(self.knots),np_type)
        trans = Transform(self.n_cop)
        
        ## Grid objects
        self.grid_u = grid_obj(ex_u)
        self.grid_s = grid_obj(trans.forward_u(ex_u))
        
        ## Bivariate normal
        x1_s, x2_s = self.grid_s.axis()
        NORM = biv_norm(x1_s, x2_s)
        self.grid_u.axis()
        self.grid_s.min_grid()
        self.grid_s.max_grid()
        
        ## Create Mar_G, theta and Fp
        self.Mar_G = []
        self.theta_flip = np.zeros([tf.shape(x)[0],self.n_cop,self.n_cop],np_type)
        self.theta = np.zeros([tf.shape(x)[0],self.n_cop,self.n_cop],np_type)
        for i in range(0,self.n_cop,1):
            ccc = self.margin[i].ker
            interp_cdf, mar_s1, mar_p1 = kernel_cdf(ccc,ccc,ex_u)

            self.Mar_G.append([mar_s1, mar_p1])
            self.theta[:,0,i] = interp_cdf.numpy() #interp1d_np(ccc, mar_s1, mar_p1).numpy()
            del ccc, mar_p1, mar_s1
        
        ######################################### FITTING #######################################################
        
        if self.fitted == False:
            self.copulas = []
        self.correlations = []
        self.correlations_bins = []
        self.flip_flag = []
        self.ind_edge_rel = []
        
        for tr in tf.range(0,self.vine_depth-1,1,tf.int32): #d-1
            print('-----------------------------------')
            print('Row theta:',tr.numpy())
            
            if self.fitted == True:
                self.vine_family = 'r-vine'
                self.method = 'matrix'
            
            print('theta:',self.theta[:,tr,:])

            ## Number of copulas in the level
            ## Create object for projections in the other spaces
            
            n_cop = d-1-tr
            trans = Transform(n_cop)
            print('n_cop in the row:',n_cop.numpy())
            
            ###### COMPUTE THE EDGES OF THE VINE LEVEL
            
            if self.vine_family == 'r-vine':
                
                if self.method == 'matrix':   
                    edges_now = self.ind_vine[tr]
                    
                elif (self.method == 'optimal') | (self.method == 'random'):   
                    
                    random = False
                    
                    if (self.method == 'random'):
                        random = True
                        
                    if tr == 0:
                        self.r_matrix = np.zeros([self.n_cop,self.n_cop],np.int32)
                        n = len(self.r_matrix) - 1
                        ind_ee, weights = optimal_tree(self.theta[:,tr,:],self.theta_flip[:,tr,:],self.ind_vine,tr,random)
                        edges_now = ind_ee
                        self.ind_vine[tr] = ind_ee
                        print('opt_tree',ind_ee)

                        edges = []
                        for j in range(0,len(ind_ee),1):
                            edg = ind_ee[len(ind_ee)-1-j]
                            self.r_matrix[n,j] = edg[0] +1
                            self.r_matrix[j,j] = edg[1] +1
                            edges.append({edg[0],edg[1]})

                        edges = np.flip(edges)

                        self.nodes = np.zeros(self.n_cop,np.int32)
                        V = set(range(1,self.n_cop+1))
                        for i in range(0,self.n_cop,1):
                            self.nodes[i]=self.r_matrix[i,i]
                            u_nod = {self.nodes[i]}
                            if u_nod.issubset(V):
                                V.remove(self.nodes[i])
                        self.nodes = np.flip(self.nodes)

                        for elem in V:
                            ind = np.where(self.nodes == 0)
                            self.nodes[self.nodes == 0] = elem
                            self.r_matrix[n-ind[0],n-ind[0]] = elem
                    else:

                        ind_ee, weights = optimal_tree(self.theta[:,tr,:],self.theta_flip[:,tr,:],self.ind_vine,tr,random)
                        print('opt_tree',ind_ee)
                        edges_now = ind_ee
                        self.ind_vine[tr] = ind_ee

            elif (self.vine_family == 'c-vine') | (self.vine_family == 'd-vine'):
                edges_now = self.ind_vine[tr]
            
            
            ######### FROM THETA MATRIX TAKE THE DATA CDF FOR THE COPULA FITTING
            # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
            
            self.data_u = np.zeros([np.shape(self.theta)[0],2,n_cop],np_type)  

            for j in range(0,len(edges_now),1):
                edge = edges_now[j]
                
                ## When tr = 0 there is no parent variable.
                ## After check if has to get the CDF from theta flip
                if tr == 0:
                    self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
                else:
                    parent1, inx1, inx2 = parent_var(tr,self.ind_vine,edge)

                    if self.ind_vine[tr-1][edge[0]][0] != parent1: 
                        self.data_u[:,:,j] = np.concatenate((self.theta_flip[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
                    else:
                        self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
            
            ##### Transform data
            self.data_s = trans.forward_u(self.data_u)
#             self.data_s = check_bound3(self.data_s,tf.constant(3.2-1e-6,x.dtype),tf.constant(-3.2+1e-6,x.dtype))
            self.data_x = trans.forward_s(self.data_s)
            
            ##### Grid on P-Q space
            self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))
            
            ############################### FIT BANDWIDTH  #####################################
            
            if self.fitted == False:
            
                #self.copulas.append([])
                opt_bw = tf.TensorArray(x.dtype,size=n_cop)

                if (tr == 0) | (self.binning == False):

                    if self.parallel == False:

                        if self.param == True:

                            ### NOT BINNING, NOT PARALLEL, PARAMETRIC
                            par_copulas = []
                            tau_values = []
                            n_cop1 = tf.constant(1,tf.int32)
                            for j in range(0,len(edges_now),1):
                                
                                tau, p_value = kendalltau(self.data_u[:,0,j],self.data_u[:,1,j])
                                tau_values.append(tau)

                                start_time = perf_counter()
                                families = param_families # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
                                aic, theta_par, logp = parametric_fit(self.data_u[:,:,j][...,tf.newaxis], families, n_cop1)
                                time_fit_gauss = perf_counter()  - start_time

                                print('aic',aic)
                                print('theta_par',theta_par)

                                ind_fam = np.argmin(aic)
                                ## Gaussian
                                family = families[ind_fam]
                                theta_est = theta_par[0][ind_fam]

                                print('fam_fit',family)
                                print('theta_fit',theta_est)

                                cop_p = cop_par_obj(family,theta_est)
                                par_copulas.append(cop_p)

                            self.copulas.append(par_copulas)
                            self.correlations.append(tau_values)
                        else: #param

                            ### NOT BINNING, NOT PARALLEL, NOT PARAMETRIC
                            opt_bw = tf.TensorArray(x.dtype,size=n_cop)
                            tau_values = []
                            
                            ## Batches
                            batch_size = tf.constant(2,tf.int32)
                            if np.shape(self.data_s)[0] > 10000:
                                batch_size = tf.constant(10,tf.int32)
                            elif np.shape(self.data_s)[0] > 50000:
                                batch_size = tf.constant(20,tf.int32)
                            elif np.shape(self.data_s)[0] > 100000:
                                batch_size = tf.constant(50,tf.int32)
                            elif np.shape(self.data_s)[0] > 200000:
                                batch_size = tf.constant(100,tf.int32)
                            elif np.shape(self.data_s)[0] > 500000:
                                batch_size = tf.constant(200,tf.int32)

                            for i in range(0,n_cop,1):
                                print('col:',i)
                                
                                tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
                                tau_values.append(tau)

                                n_cop1 = tf.constant(1,tf.int32)

                                grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
                                data_dict = {'data_s':self.data_s[:,:,i], 'data_x':self.data_x[:,:,i]}
                                par_dict = {'n_cop':n_cop1, 'batch':batch_size, 'max_iter': [70,100], 'lr':[0.1, 0.03], #lr = 0.1, 0.01
                                            'conv_tol': [1e-5,5e-5], 'opt_method': self.opt_method}  #1e-5

                                opt = optimization(grid_dict, data_dict, par_dict)
                                opt_bw = opt_bw.write(i,opt)

                            opt_bw = opt_bw.stack()

                            bw = bandwidth_mul(self.data_x,2,n_cop)
                            bw1 = np.transpose(np.squeeze(opt_bw))*bw
                            
                            ### Check constraints on the bandwidth
                            bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(1e-2,x.dtype))  ##It was 5e-3 but too low

                            copula = copula_obj(bw1.numpy())
                            self.copulas.append(copula)
                            self.correlations.append(tau_values)

                            print('opt_bw',bw1)

                    else: #Parallel

                        if self.param == True:

                            par_copulas = []
                            tau_values = []
    #                         if n_cop == 1:
    #                             n_cop = 1  ## THIS BECAUSE THERE IS A PROBLEM IN SHAPE 'a' WITH FIT_STUDENT EVEN IF I FORCE IT TO BE THE SAME

                            ### NOT BINNING, PARALLEL, PARAMETRIC
                            start_time = perf_counter()
                            families = param_families # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
                            aic, theta_par, logp = parametric_fit(self.data_u, families, n_cop)
                            time_fit_gauss = perf_counter()  - start_time

                            print('aic',aic)
                            print('theta_par',theta_par)
    
                            for i in range(0,n_cop,1):
                                tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
                                tau_values.append(tau)
                                
                                ind_fam = np.argmin(aic[i])
                                family = families[ind_fam]
                                theta_est = theta_par[i][ind_fam]
                                print('fam_fit',family)
                                print('theta_fit',theta_est)
                                cop_p = cop_par_obj(family,theta_est)
                                par_copulas.append(cop_p)
                            self.copulas.append(par_copulas)
                            self.correlations.append(tau_values)

                        else: #param

                            ### NOT BINNING, PARALLEL, NOT PARAMETRIC
                            n_cop1 = tf.constant(n_cop,tf.int32)

                            batch_size = tf.constant(2,tf.int32)
                            if np.shape(self.data_s)[0]*n_cop1 > 5000:
                                batch_size = tf.constant(10,tf.int32)
                            elif np.shape(self.data_s)[0]*n_cop1 > 10000:
                                batch_size = tf.constant(20,tf.int32)
                            elif np.shape(self.data_s)[0]*n_cop1 > 20000:
                                batch_size = tf.constant(50,tf.int32)
                            elif np.shape(self.data_s)[0]*n_cop1 > 50000:
                                batch_size = tf.constant(100,tf.int32)
                            elif np.shape(self.data_s)[0]*n_cop1 > 100000:
                                batch_size = tf.constant(200,tf.int32)
                            
                            if self.opt_method == 'LL1':
                                opt_bw = np.zeros((1,n_cop1),np_type)
                            else:
                                opt_bw = np.zeros((2,n_cop1),np_type)
                                
                            batch_parallel = batch_paral
                            batch_len1 = n_cop1/batch_parallel
                            batch_len = tf.cast(batch_len1,tf.int32)
                            
                            if batch_len <= 1:
                                batch_len = n_cop1
                                batch_parallel = 1
                            else:
                                while batch_parallel*batch_len < n_cop1:
                                        batch_parallel += 1
                            
                            for j in range(0,batch_parallel,1):

                                grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,batch_len*j:batch_len*(j+1)]}
                                data_dict = {'data_s':self.data_s[:,:,batch_len*j:batch_len*(j+1)], 'data_x':self.data_x[:,:,batch_len*j:batch_len*(j+1)]}
                                par_dict = {'n_cop':tf.shape(self.data_s[:,:,batch_len*j:batch_len*(j+1)])[2], 'batch':batch_size, 'max_iter': [70,100], 'lr':[0.1, 0.03], 
                                                'conv_tol': [1e-5,5e-5], 'opt_method': self.opt_method}  ## 1e-5,5e-5

                                opt = optimization(grid_dict, data_dict, par_dict)
                                print('opt',opt)
                                
                                opt_bw[:,batch_len*j:batch_len*(j+1)] = opt.numpy() #[...,tf.newaxis]
                                
                            opt_bw = tf.convert_to_tensor(opt_bw)
                            
                            bw = bandwidth_mul(self.data_x,2,n_cop)
                            bw1 = np.transpose(np.squeeze(opt_bw))*bw
                            bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(1e-2,x.dtype))  ##It was 5e-3 but too low

                            
                            copula = copula_obj(bw1.numpy())
                            self.copulas.append(copula)
                            
                            tau_values = []
                            for i in range(0,n_cop,1):
                                tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
                                tau_values.append(tau)
                            self.correlations.append(tau_values)

                            print('opt_bw',bw1)

                else: #Binning

                    if self.parallel == False:

                        if self.param == True:

                            ### BINNING, NOT PARALLEL, PARAMETRIC

                            par_copulas = []
                            tau_values = []
                            tau_val_bin = []
                            n_cop1 = 1
                            for j in range(0,len(edges_now),1):
                                
                                tau, p_value = kendalltau(self.data_u[:,0,j],self.data_u[:,1,j])
                                tau_values.append(tau)
                                print('Tau value before binning: ',tau)
                                
                                ind_now = edges_now[j]
                                parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                                ind1 = parent11

                                #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                                if tr == 1:
                                    bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                else:
                                    ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                                    parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                                    if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                        bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                        val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                        val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                    else:
                                        bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                        val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                        val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)
                                 
                                
                                bin_copulas = []
                                tau_binned = []
                                for bb in range(0,self.n_bin,1):
                                    print('bin:',bb)
                                    mask = np.where(val_to_bin == bb)
                                    u_bin = self.data_u[mask[0],:,j]
                                    
                                    ### CDF FORCE UNIFORM
                                    vv_bin_new = u_bin
                                    for zz in range(0,2,1):
                                        vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(u_bin[:,zz],u_bin[:,zz],self.grid_u.ex)
                                    u_bin = vv_bin_new[...,tf.newaxis]
                                    ###
                                    
                                    tau, p_value = kendalltau(u_bin[:,0,0],u_bin[:,1,0])
                                    print('Tau value bin -',bb, '- is: ', tau)
                                    tau_binned.append(tau)
                                    corr = stats.pearsonr(u_bin[:,0,0],u_bin[:,1,0])
                                    print('Corr value  UV space: ',corr[0])
#                                     tau_binned.append(corr[0])

                                    start_time = perf_counter()
                                    families = param_families # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
                                    aic, theta_par, logp = parametric_fit(u_bin, families, n_cop1)
                                    time_fit_gauss = perf_counter()  - start_time

                                    print('aic',aic)
                                    print('theta_par',theta_par)

                                    ind_fam = np.argmin(aic)

                                    cop_p = cop_par_obj(families[ind_fam],theta_par[0][ind_fam])
                                    bin_copulas.append(cop_p)

                                    print('fam_fit',families[ind_fam])
                                    print('theta_fit',theta_par[0][ind_fam])
                                    print('--------------------')

                                par_copulas.append(bin_copulas)
                                tau_val_bin.append(tau_binned)

                            self.copulas.append(par_copulas)
                            self.correlations.append(tau_values)
                            self.correlations_bins.append(tau_val_bin)

                        else: #param

                            ### BINNING, NOT PARALLEL, NOT PARAMETRIC
                            n = len(self.r_matrix)-1
                            tau_values = []
                            tau_val_bin = []
                            
                            for i in range(0,n_cop,1):
                                print('col:',i)
                                
                                tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
                                tau_values.append(tau)
                                print('Tau value before binning: ',tau)
                                
                                tau_binned = []
                                opt_bin = tf.TensorArray(x.dtype,size=self.n_bin)
#                                 parent = self.r_matrix[n-tr,n-2-i] - 1 # Because first node starts from 1 (and not from 0)
#                                 bins = create_bins(self.theta[:,0,parent],self.n_bin)
#                                 val_to_bin = np.digitize(self.theta[:,0,parent], bins) -1
                                ind_now = edges_now[i]  #j
                                parent, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
#                                 print('ind_now',ind_now)
#                                 print('par',parent11)
#                                 print('prev',ind_vine[tr-1][ind_now[0]],ind_vine[tr-1][ind_now[1]])
                                ind1 = parent
                                
                                 #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                                if tr == 1:
                                    bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                else:
                                    ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                                    parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                                    if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                        bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                        val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                        val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                    else:
                                        bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                        val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                        val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)

                                for bb in range(0,self.n_bin,1):
                                    print('bin:',bb)
                                    mask = tf.where(tf.equal(val_to_bin,bb))
                                    n_cop1 = tf.constant(1,tf.int32)
                                    
                                    data_u_bin = tf.gather_nd(self.data_u[:,:,i],mask)
                                    tau, p_value = kendalltau(data_u_bin[:,0],data_u_bin[:,1])
                                    tau_binned.append(tau)
                                    print('Tau value bin -',bb, '- is: ', tau)
                                    
                                    ### CDF FORCE UNIFORM
                                    data_u_bin_new = np.zeros(np.shape(data_u_bin),np_type)
                                    for zz in range(0,2,1):
                                        data_u_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],data_u_bin[:,zz],self.grid_u.ex)
                                    data_u_bin = data_u_bin_new[...,np.newaxis]
                                    ###

                                    trans = Transform(1)
                                    data_s_bin = trans.forward_u(data_u_bin)#[:,:,0]
                                    data_x_bin = trans.forward_s(data_s_bin)

#                                     data_s_bin = tf.gather_nd(self.data_s[:,:,i],mask)
#                                     data_x_bin = tf.gather_nd(self.data_x[:,:,i],mask)

                                    grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
                                    data_dict = {'data_s':data_s_bin, 'data_x':data_x_bin}
                                    par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,70], 'lr':[0.1, 0.03],
                                                'conv_tol': [1e-4,1e-4], 'opt_method': self.opt_method}

                                    opt = optimization(grid_dict, data_dict, par_dict)
                                    
                                    bw = bandwidth_mul(data_x_bin,2,n_cop1)
                                    bw1 = opt*bw
                                    bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(1e-2,x.dtype))  ##It was 5e-3 but too low
                                    
                                    opt_bin = opt_bin.write(bb,bw1)
#                                     opt_bin = opt_bin.write(bb,opt)
                                opt_bin = opt_bin.stack()
#                                 opt_bin = tf.reshape(opt_bin,[tf.shape(opt)[0],self.n_bin])
                                opt_bin = tf.reshape(opt_bin,[2,self.n_bin])

                                print(opt_bin)
                                tau_val_bin.append(tau_binned)

                                opt_bw = opt_bw.write(i,opt_bin)
                            opt_bw = opt_bw.stack()
#                             opt_bw = tf.reshape(opt_bw,[tf.shape(opt)[0],n_cop,self.n_bin])
                            opt_bw = tf.reshape(opt_bw,[2,n_cop,self.n_bin])

#                             bw = bandwidth_mul(self.data_x,2,n_cop)
                            
#                             bw1 = np.zeros((2,n_cop,self.n_bin),np_type)
#                             for i in range(0,n_cop,1):
#                                 bw1[:,i,:] = opt_bw[:,i,:]*bw[:,i][...,np.newaxis]
                            
#                             ### If bw < 5e-3 gives nan
#                             bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(5e-3,x.dtype))
                            bw1 = opt_bw
    
                            copula = copula_obj(bw1.numpy())
                            self.copulas.append(copula)
                            self.correlations.append(tau_values)
                            self.correlations_bins.append(tau_val_bin)

                            print('opt_bw',bw1)

                    else: #parallel

                        if self.param == True:
                            print('Miss to implement')
                        else:
                            tau_values = []
                            tau_val_bin = []
                            
                            len_bin = tf.shape(self.theta)[0]/self.n_bin
                            len_bin = tf.cast(len_bin,tf.int32)
#                             print('len bin',len_bin)
                            data_s_bin = np.zeros([len_bin,tf.shape(self.data_s)[1],n_cop,self.n_bin],np_type)
                            data_x_bin = np.zeros([len_bin,tf.shape(self.data_s)[1],n_cop,self.n_bin],np_type)

                            for i in range(0,n_cop,1):
                                
                                tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
                                tau_values.append(tau)
                                print('Tau value before binning: ',tau)
                                
                                tau_binned = []
                                ind_now = edges_now[i] #j
                                parent, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                                ind1 = parent
                                
                                 #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                                if tr == 1:
                                    bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                else:
                                    ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                                    parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                                    if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                        bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                        val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                        val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                    else:
                                        bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                        val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                        val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)
                        
                                for bb in range(0,self.n_bin,1):
                                    mask = tf.where(tf.equal(val_to_bin,bb))
#                                     print('mask bin',np.shape(mask))
                                    
                                    ##### FIXED FOR THE UNIFORMITY
                                    data_u_bin = tf.gather_nd(self.data_u[:,:,i],mask)

                                    ### CDF FORCE UNIFORM
                                    data_u_bin_new = np.zeros(np.shape(data_u_bin),np_type)
                                    for zz in range(0,2,1):
                                        data_u_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],data_u_bin[:,zz],self.grid_u.ex)
                                    data_u_bin = data_u_bin_new[...,np.newaxis]
                                    ###

                                    trans = Transform(1)
                                    data_s_bin[:,:,i,bb] = trans.forward_u(data_u_bin)[:,:,0]
                                    data_x_bin[:,:,i,bb] = trans.forward_s(data_s_bin[:,:,i,bb][...,tf.newaxis])[:,:,0]
                                    
#                                     data_s_bin[:,:,i,bb] = tf.gather_nd(self.data_s[:,:,i],mask)
#                                     data_x_bin[:,:,i,bb] = tf.gather_nd(self.data_x[:,:,i],mask)
                                    
                                    u_bin = tf.gather_nd(self.data_u[:,:,i],mask)
                                    tau, p_value = kendalltau(u_bin[:,0],u_bin[:,1])
                                    tau_binned.append(tau)
                                    print('Tau value bin -',bb, '- is: ', tau)
                                    corr = stats.pearsonr(u_bin[:,0],u_bin[:,1])
                                    print('Corr value  UV space: ',corr[0])
                                tau_val_bin.append(tau_binned)


                            opt_bw = np.zeros((2,n_cop,self.n_bin),np_type)
                            batch_parallel = batch_paral
                            batch_len1 = n_cop/batch_parallel
                            batch_len = tf.cast(batch_len1,tf.int32)

                            #                             if tf.cast(batch_len1,x.dtype) > tf.cast(batch_len,x.dtype):
                            if batch_len <= 1:
                                batch_len = n_cop
                                batch_parallel = 1
                            else:
                                while batch_parallel*batch_len < n_cop:
                                        batch_parallel += 1

                            for j in range(0,batch_parallel,1):

                                for bb in range(0,self.n_bin,1):
                                    ## UPDATE THETA
                                    n_batch = tf.shape(data_s_bin[:,:,batch_len*j:batch_len*(j+1),bb])[2]

                                    grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,batch_len*j:batch_len*(j+1)]}
                                    data_dict = {'data_s':data_s_bin[:,:,batch_len*j:batch_len*(j+1),bb], 'data_x':data_x_bin[:,:,batch_len*j:batch_len*(j+1),bb]}
                                    par_dict = {'n_cop':n_batch, 'batch':batch_size, 
                                                'max_iter': [70,100], 'lr':[0.1, 0.03], 'conv_tol': [1e-4,1e-4], 'opt_method': self.opt_method}

                                    opt = optimization(grid_dict, data_dict, par_dict)

                                    bw = bandwidth_mul(data_x_bin[:,:,batch_len*j:batch_len*(j+1),bb],2,n_batch)

                                    bw1 = opt*bw
                                    bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(1e-2,x.dtype))

                                    opt_bw[:,batch_len*j:batch_len*(j+1),bb] = bw1.numpy()
                            
                            print('opt_bw',opt_bw)
                            copula = copula_obj(opt_bw)
                            self.copulas.append(copula)
                            self.correlations.append(tau_values)
                            self.correlations_bins.append(tau_val_bin)
                   
            ##############################  UPDATE THETA #####################################
            
            n = np.shape(self.r_matrix)[0] -1
            
            #### if optimal or random, flip_flag = [True,False,True,False,...] in order to evaluate all possible orders
            #### Otherwise just stores when to flip based on the parent variable
            ## Flip_flap stores boolean if flipped or not
            ## ind_edge_rel1 refers to the index of the copula
            
            flip_flag1 = []
            ind_edge_rel1 = []
            parent_all = []
            if (self.vine_family == 'r-vine'):
                if (self.method == 'optimal') | (self.method == 'random'):
                    for j in range(0,len(edges_now),1):
                        edge = edges_now[j]
                        flip_flag1.append(True)
                        flip_flag1.append(False)
                        ind_edge_rel1.append(j)
                        ind_edge_rel1.append(j)
                        parent_all.append([edge[0],edge[1]])
                else:
                    flip_flag1, ind_edge_rel1, parent_all = flip_check_all(self.ind_vine, tr, self.binning, self.n_bin)
            else:
                flip_flag1, ind_edge_rel1, parent_all = flip_check_all(self.ind_vine, tr, self.binning, self.n_bin)
            
            ## vv_s is another variable which stores the data_s taking into account also the flipping
            
            vv_u = np.zeros((np.shape(self.data_u)[0],np.shape(self.data_u)[1],len(ind_edge_rel1)),self.data_u.dtype)
            vv_s = np.zeros((np.shape(self.data_u)[0],np.shape(self.data_u)[1],len(ind_edge_rel1)),self.data_u.dtype)

            for j in range(0,len(ind_edge_rel1),1):
                ind_edge = ind_edge_rel1[j]
                edge = edges_now[ind_edge]

                if self.param == True:
                    if (tr==0) | (self.binning == False):

                        cop_p = self.copulas[tr][ind_edge]

                        if flip_flag1[j] == True:
                            vv = self.data_u[:,:,ind_edge]
                            self.theta_flip[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                        else:
                            vv = np.flip(self.data_u[:,:,ind_edge],1)
                            self.theta[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))

                    else: #binning

                        parent11, inx1, inx2 = parent_var(tr,self.ind_vine,edge)
                        ind1 = parent11
        
                        
                        #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                        if tr == 1:
                            bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                            val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                            val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                        else:
                            ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                            if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                            else:
                                bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)

                        flip_flag_bin = []
#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
                        
                        for bb in range(0,self.n_bin,1):

                            cop_p = self.copulas[tr][ind_edge][bb]

                            mask = np.where(val_to_bin == bb)

                            if flip_flag1[j] == True:
                                vv = self.data_u[:,:,ind_edge]
                                vv_bin = vv[mask[0],:]
                                
                                ### CDF FORCE UNIFORM
                                vv_bin_new = vv_bin
                                for zz in range(0,2,1):
                                    vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],self.grid_u.ex)
                                vv_bin = vv_bin_new
                                ###
#                                 print('flip')
#                                 print('bin-',bb,',bef: ',vv_bin)
                                self.theta_flip[mask[0],tr+1,ind_edge] = copulaccdf(cop_p,vv_bin) 
#                                 print('bin-',bb,': ',self.theta_flip[mask[0],tr+1,ind_edge])
                            else:
                                vv = np.flip(self.data_u[:,:,ind_edge],1)
                                vv_bin = vv[mask[0],:]
                                
                                ### CDF FORCE UNIFORM
                                vv_bin_new = vv_bin
                                for zz in range(0,2,1):
                                    vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],self.grid_u.ex)
                                vv_bin = vv_bin_new
                                ###
#                                 print('no')
#                                 print('bin-',bb,',bef: ',vv_bin)
                                self.theta[mask[0],tr+1,ind_edge] = copulaccdf(cop_p,vv_bin)
#                                 print('bin-',bb,': ',self.theta[mask[0],tr+1,ind_edge])
                else: #param
                    
                    if flip_flag1[j] == True:
                        vv_u[:,:,j] = np.flip(self.data_u[:,:,ind_edge],1)
                        vv_s[:,:,j] = np.flip(self.data_s[:,:,ind_edge],1) #self.data_s[:,:,j] #Flip cambia per npc
                    else:
                        vv_u[:,:,j] = self.data_u[:,:,ind_edge]
                        vv_s[:,:,j] = self.data_s[:,:,ind_edge] #np.flip(self.data_s[:,:,j],1)

            self.flip_flag.append(flip_flag1)
            self.ind_edge_rel.append(ind_edge_rel1)

            
            if self.param == False:
                
                n_eval = len(self.ind_edge_rel[tr])
                self.data_u = vv_u[:,:,:n_eval]
                self.data_s = vv_s[:,:,:n_eval]
                trans = Transform(n_eval)
                self.data_x = trans.forward_s(self.data_s)
                grid_x = trans.forward_s(self.grid_s.ex)
                
                del vv_s
                
                if (tr == 0) | (self.binning == False):
                    
                    batch_size = tf.constant(2,tf.int32)
                    if np.shape(self.data_s)[0]*n_eval > 5000:
                        batch_size = tf.constant(10,tf.int32)
                    elif np.shape(self.data_s)[0]*n_eval > 10000:
                        batch_size = tf.constant(20,tf.int32)
                    elif np.shape(self.data_s)[0]*n_eval > 20000:
                        batch_size = tf.constant(50,tf.int32)
                    elif np.shape(self.data_s)[0]*n_eval > 50000:
                        batch_size = tf.constant(100,tf.int32)
                    elif np.shape(self.data_s)[0]*n_eval > 100000:
                        batch_size = tf.constant(200,tf.int32)

                    grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':grid_x}
                    data_dict = {'data_s':self.data_s, 'data_x':self.data_x, 'theta':self.theta, 'theta_flip':self.theta_flip}
                    par_dict = {'copulas': self.copulas[tr], 'n_eval':tf.convert_to_tensor(n_eval), 'batch':batch_size, 'batch_cdf':batch_size_cdf, 'tr':tr,
                               'ind_edge_rel': self.ind_edge_rel[tr], 'flip_flag': self.flip_flag[tr]}

                    self.copulas[tr].pd_grid_uv, self.copulas[tr].cdf, self.theta, self.theta_flip = evaluate_fit(data_dict, grid_dict, par_dict)
                    
                else: #binning
                    
                    len_bin = tf.shape(self.theta)[0]/self.n_bin
                    len_bin = tf.cast(len_bin,tf.int32)
                    data_s_bin = np.empty([len_bin,tf.shape(self.data_s)[1],n_eval,self.n_bin],np_type)
                    data_x_bin = np.empty([len_bin,tf.shape(self.data_s)[1],n_eval,self.n_bin],np_type)

                    for i in range(0,n_eval,1):
                        ind_edge = self.ind_edge_rel[tr][i]
#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
#                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
                        ind_now = edges_now[ind_edge]
                        parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                        ind1 = parent11
#                                 print('ind_now',ind_now)
#                                 print('par',parent11)
#                                 print('prev',ind_vine[tr-1][ind_now[0]],ind_vine[tr-1][ind_now[1]])

                        #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                        if tr == 1:
                            bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                            val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                            val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                        else:
                            ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                            if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                            else:
                                bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)
                        
                        for bb in range(0,self.n_bin,1):
                            mask = tf.where(tf.equal(val_to_bin,bb))
                            
                            ##### FIXED FOR THE UNIFORMITY
                            data_u_bin = tf.gather_nd(self.data_u[:,:,i],mask)
                            
                            ### CDF FORCE UNIFORM
                            data_u_bin_new = np.zeros(np.shape(data_u_bin),np_type)
                            for zz in range(0,2,1):
                                data_u_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],data_u_bin[:,zz],self.grid_u.ex)
                            data_u_bin = data_u_bin_new[...,np.newaxis]
                            ###
                            
                            trans = Transform(1)
                            data_s_bin[:,:,i,bb] = trans.forward_u(data_u_bin)[:,:,0]
                            data_x_bin[:,:,i,bb] = trans.forward_s(data_s_bin[:,:,i,bb][...,tf.newaxis])[:,:,0]
                            
#                             data_s_bin[:,:,i,bb] = tf.gather_nd(self.data_s[:,:,i],mask)
#                             data_x_bin[:,:,i,bb] = tf.gather_nd(self.data_x[:,:,i],mask)
                
                    self.copulas[tr].pd_grid_uv = np.zeros([self.knots,self.knots,n_eval,self.n_bin],np_type)
                    self.copulas[tr].cdf = np.zeros([self.knots,self.knots,n_eval,self.n_bin],np_type)
                    
                    for bb in range(0,self.n_bin,1):
                        ## UPDATE THETA

                        grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':grid_x}
                        data_dict = {'data_s':data_s_bin[:,:,:,bb], 'data_x':data_x_bin[:,:,:,bb]} #],'bin':bb
                        par_dict = {'bw': tf.convert_to_tensor(self.copulas[tr].opt_bw[:,:,bb]), 'n_cop':tf.convert_to_tensor(n_eval), 'batch':tf.constant(2,tf.int32), 'tr':tr, 'ind_edge_rel':self.ind_edge_rel[tr]}

                        self.copulas[tr].pd_grid_uv[:,:,:,bb], self.copulas[tr].cdf[:,:,:,bb] = evaluate_fit_bin(data_dict, grid_dict, par_dict)
                    
                    interp_cdf_bin = np.zeros([tf.shape(self.theta)[0],n_eval],np_type)
                    for i in range(0,n_eval,1):
                        print('col:',i)
                        ind_edge = self.ind_edge_rel[tr][i]
#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)

#                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
                        ind_now = edges_now[ind_edge]
                        parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                        ind1 = parent11

                         #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                        if tr == 1:
                            bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                            val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                            val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                        else:
                            ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                            if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                            else:
                                bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)

                        for bb in range(0,self.n_bin,1):
#                             print('bin:',bb)
                            
                            mask = tf.where(tf.equal(val_to_bin,bb))

                            ## Update theta  
                            ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin[:,:,i,bb],self.grid_s.min,self.grid_s.max,self.copulas[tr].cdf[:,:,i,bb],axis=-2)
                            ccdf_data = tf.squeeze(ccdf_data)
                                
#                             u_bin = tf.gather_nd(self.data_u[:,:,i],mask)
                        
#                             print('bin-',bb,',bef: ',u_bin)
#                             print('bin-',bb,',bef: ',ccdf_data)
                            interp_cdf, mar_s, mar_p = kernel_cdf(ccdf_data,ccdf_data,self.grid_u.ex)
#                             print('bin-',bb,': ',interp_cdf)
                            
                            if self.flip_flag[tr][i] == True:
                                self.theta_flip[mask,tr+1,ind_edge] = tf.reshape(interp_cdf,[tf.shape(interp_cdf)[0],1])
                            else:
                                self.theta[mask,tr+1,ind_edge] = tf.reshape(interp_cdf,[tf.shape(interp_cdf)[0],1])
        
        ### After finding the optimal or the random vine, it stores the connection in the r_matrix
        if self.vine_depth == self.n_cop:
            if self.vine_family == 'r-vine':
                if (self.method == 'optimal') | (self.method == 'random'):
                    self.r_matrix, self.E, self.nodes = prepare_optimal(self.n_cop,self.ind_vine)
        
        return
    
    
        ################################ EVALUATION ##############################################################################################
    def evaluation(self, points):
        
        d = self.n_cop
        
        ## Create Fp
        self.Fp = np.zeros([tf.shape(points)[0],self.n_cop,self.n_cop],self.data_u.dtype)
        self.Fp_flip = np.zeros([tf.shape(points)[0],self.n_cop,self.n_cop],self.data_u.dtype)
        ### Create logf
        logf = tf.zeros([tf.shape(points)[0],tf.shape(points)[1],self.vine_depth],points.dtype)
        self.logf_flip = tf.zeros(tf.shape(points),points.dtype)
        
        for i in range(0,d,1):
            interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(self.margin[i].ker,points[:,i],self.grid_u.ex)
            self.Fp[:,0,i] = interp_cdf_poi.numpy()
            
            den1,mden1 = kernel_pdf2(self.margin[i].ker)      
            inter = interp_pdf(points[:,i], mden1, den1) #interp1d_np
            
            # Product of pdf is the sum of logarithm - Product of pdf margingales evaluated on copula samples
            logf = update_tensor(logf,tf.math.log(inter),i,0)
           
            del den1,mden1, inter, interp_cdf_poi #,logf_tmp
            
        self.logf = logf.numpy()
        
        for tr in range(0,self.vine_depth-1,1): #d-1
            print('Row theta:',tr)
            
            if self.vine_family == 'r-vine':
                    if (self.method == 'optimal') | (self.method == 'random'):
                        self.flip_flag[tr], self.ind_edge_rel[tr], parent_all = flip_check_all(self.ind_vine, tr, self.binning, self.n_bin)
            
            # Number of copuals to evaluate and create Transform object
            
            n_eval = len(self.ind_edge_rel[tr])
            trans = Transform(n_eval)
            print('n to eval in the row:',n_eval)
            
            ## Edges of the vine
            edges_now = self.ind_vine[tr]
            
            ######### FROM THETA MATRIX TAKE THE DATA CDF FOR THE COPULA FITTING
            # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
            
            self.data_u = np.zeros([np.shape(self.theta)[0],2,n_eval],self.data_u.dtype)
            self.points_u = np.zeros([np.shape(self.Fp)[0],2,n_eval],self.data_u.dtype)
            for j in range(0,n_eval,1):
                edge = edges_now[self.ind_edge_rel[tr][j]]
                if tr == 0:
                    self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
                    self.points_u[:,:,j] = np.concatenate((self.Fp[:,tr,edge[0]][...,np.newaxis],self.Fp[:,tr,edge[1]][...,np.newaxis]),1)
                else:
                    parent1, inx1, inx2 = parent_var(tr,self.ind_vine,edge)
                    
                    if self.ind_vine[tr-1][edge[0]][0] != parent1: 
                        self.data_u[:,:,j] = np.concatenate((self.theta_flip[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
                        self.points_u[:,:,j] = np.concatenate((self.Fp_flip[:,tr,edge[0]][...,np.newaxis],self.Fp[:,tr,edge[1]][...,np.newaxis]),1)
                    else:
                        self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
                        self.points_u[:,:,j] = np.concatenate((self.Fp[:,tr,edge[0]][...,np.newaxis],self.Fp[:,tr,edge[1]][...,np.newaxis]),1)
                
                if self.param == False:
                    if self.flip_flag[tr][j] == True:
                        self.data_u[:,:,j] = np.flip(self.data_u[:,:,j],1)
                        self.points_u[:,:,j] = np.flip(self.points_u[:,:,j],1)
            
            ### Transform data
            self.data_s = trans.forward_u(self.data_u)
#             self.data_s = check_bound3(self.data_s,tf.constant(3.2-1e-6,points.dtype),tf.constant(-3.2+1e-6,points.dtype))
            self.data_x = trans.forward_s(self.data_s)
            
            ### Transform points
            self.points_s = trans.forward_u(self.points_u)
#             self.points_s = check_bound3(self.points_s,tf.constant(3.2-1e-6,points.dtype),tf.constant(-3.2+1e-6,points.dtype))
            self.points_x = trans.forward_s(self.points_s)
            
            ### Grid on P-Q space
            self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))
            
            n = np.shape(self.r_matrix)[0] -1
            
            if self.param == False:
                
                pd_grid_uv = self.copulas[tr].pd_grid_uv
                cdf1 = self.copulas[tr].cdf

                if (tr==0) | (self.binning == False):

                    for j in range(0,n_eval,1):
                        
                        ind_edge = self.ind_edge_rel[tr][j]
                        
                        ind_pd = j
                        if self.vine_family == 'r-vine':
                            if (self.method == 'optimal') | (self.method == 'random'):
                                ind_pd = self.ind_edge_rel[tr][j]*2
                                if self.flip_flag[tr][j] == True:
                                    ind_pd = ind_pd
                                else:
                                    ind_pd = ind_pd + 1
                        
                        batch_size = tf.constant(2,tf.int32)
                        if np.shape(self.data_s)[0] > 5000:
                            batch_size = tf.constant(10,tf.int32)
                        elif np.shape(self.data_s)[0] > 10000:
                            batch_size = tf.constant(20,tf.int32)
                        elif np.shape(self.data_s)[0] > 20000:
                            batch_size = tf.constant(50,tf.int32)
                        elif np.shape(self.data_s)[0] > 50000:
                            batch_size = tf.constant(100,tf.int32)
                        elif np.shape(self.data_s)[0] > 100000:
                            batch_size = tf.constant(200,tf.int32)

                        ccdf_data = tfp.math.batch_interp_regular_nd_grid(self.data_s[:,:,j],self.grid_s.min,self.grid_s.max,cdf1[:,:,ind_pd],axis=-2)
                        
                        pd_points, ccdf_points = evaluate_points(self.points_s[:,:,j], batch_size, self.grid_s, cdf1[:,:,ind_pd], pd_grid_uv[:,:,ind_pd])   

                        # Update logf
                        logftr = tf.math.log(pd_points) 
                        
                        self.logf[:,ind_edge,tr+1] = tf.squeeze(logftr).numpy() #update_tensor(logf,logftr,j,tr+1)

                        # Update Fp
                        
                        interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data,ccdf_points,self.grid_u.ex)

                        if self.flip_flag[tr][j] == False:
                            self.Fp[:,tr+1,ind_edge] = interp_cdf_poi
                        else:
                            self.Fp_flip[:,tr+1,ind_edge] = interp_cdf_poi

                else: #binning

                    batch_size = tf.constant(1,tf.int32)

                    len_bin = tf.shape(self.theta)[0]/self.n_bin
                    len_bin = tf.cast(len_bin,tf.int32)
                    data_s_bin = np.zeros([len_bin,tf.shape(self.data_s)[1],n_eval,self.n_bin],self.data_u.dtype)
                    len_bin1 = tf.shape(self.Fp)[0]/self.n_bin   
                    len_bin1 = tf.cast(len_bin1,tf.int32)
                    points_s_bin = []

                    for i in range(0,n_eval,1):
                        ind_edge = self.ind_edge_rel[tr][i]

#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)

#                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
#                         val_to_bin1 = np.digitize(self.Fp[:,0,ind1], bins) -1
                        ind_now = edges_now[ind_edge]
                        parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                        ind1 = parent11
#                                 print('ind_now',ind_now)
#                                 print('par',parent11)
#                                 print('prev',ind_vine[tr-1][ind_now[0]],ind_vine[tr-1][ind_now[1]])
                        
                        if tr == 1:
                            bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                            val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                            val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                            val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                        else:
                            ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  
                            if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                            else:
                                bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)
                                val_to_bin1 = np.digitize(self.Fp_flip[:,tr-1,ind1], bins) -1

                        points_s_bin1 = []
                        for bb in range(0,self.n_bin,1):

                            mask = tf.where(tf.equal(val_to_bin,bb))
                            mask1 = tf.where(tf.equal(val_to_bin1,bb))
                            
                            ##### FIXED FOR THE UNIFORMITY
                            data_u_bin = tf.gather_nd(self.data_u[:,:,i],mask)
                            points_u_bin = tf.gather_nd(self.points_u[:,:,i],mask1)
                            
                            ### CDF FORCE UNIFORM
                            data_u_bin_new = np.zeros(np.shape(data_u_bin),self.data_u.dtype)
                            points_u_bin_new = np.zeros(np.shape(points_u_bin),self.data_u.dtype)
                            for zz in range(0,2,1):
                                data_u_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],data_u_bin[:,zz],self.grid_u.ex)
                                points_u_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],points_u_bin[:,zz],self.grid_u.ex)
                            data_u_bin = data_u_bin_new[...,np.newaxis]
                            points_u_bin = points_u_bin_new #[...,np.newaxis]
                            ###
                            
                            trans = Transform(1)
                            data_s_bin[:,:,i,bb] = trans.forward_u(data_u_bin)[:,:,0]
                            points_s_bin1.append(trans.forward_u(points_u_bin))#[:,:,0]

#                             data_s_bin[:,:,i,bb] = tf.gather_nd(self.data_s[:,:,i],mask)
#                             points_s_bin1.append(tf.gather_nd(self.points_s[:,:,i],mask1))

                        points_s_bin.append(points_s_bin1)

                    log_f_bin = np.zeros([tf.shape(self.logf)[0],n_eval],self.data_u.dtype)
                    Fp_bin = np.zeros([tf.shape(self.Fp)[0],n_eval],self.data_u.dtype)
                    for i in range(0,n_eval,1):

                        ind_edge = self.ind_edge_rel[tr][i]
                        
                        ind_now = edges_now[ind_edge]
                        parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                        ind1 = parent11
#                                 print('ind_now',ind_now)
#                                 print('par',parent11)
#                                 print('prev',ind_vine[tr-1][ind_now[0]],ind_vine[tr-1][ind_now[1]])
                        
                        if tr == 1:
                            bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                            val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                        else:
                            ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  
                            if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                            else:
                                bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                val_to_bin1 = np.digitize(self.Fp_flip[:,tr-1,ind1], bins) -1

                        for bb in range(0,self.n_bin,1):
                            mask1 = tf.where(tf.equal(val_to_bin1,bb))

                            ## Update theta  
                            ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin[:,:,i,bb],self.grid_s.min,self.grid_s.max,self.copulas[tr].cdf[:,:,i,bb],axis=-2)
                            ccdf_data = tf.squeeze(ccdf_data)
                
                            pd_points, ccdf_points = evaluate_points(points_s_bin[i][bb], batch_size, self.grid_s, cdf1[:,:,i,bb], pd_grid_uv[:,:,i,bb]) 

                            ## Update logf
                            logftr = tf.math.log(pd_points) 
                
                            self.logf[tf.squeeze(mask1),ind_edge,tr+1] = tf.squeeze(logftr).numpy()

                            ## Update Fp
                            interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data,ccdf_points,self.grid_u.ex)
                            Fp_bin[mask1,i] = tf.reshape(interp_cdf_poi,[tf.shape(interp_cdf_poi)[0],1])

                        Fp_bin1 = tf.squeeze(Fp_bin[:,i])

                        if self.flip_flag[tr][i] == True:
                            self.Fp_flip[:,tr+1,ind_edge] = Fp_bin1
                        else:
                            self.Fp[:,tr+1,ind_edge] = Fp_bin1

            else:

                if (tr==0) | (self.binning == False):
                    
                    for j in range(0,len(self.ind_edge_rel[tr]),1):

                        ind_edge = self.ind_edge_rel[tr][j]

                        cop_p = self.copulas[tr][ind_edge]

                        if self.flip_flag[tr][j] == True:
                            vv = self.points_u[:,:,j]
                            self.Fp_flip[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                        else:
                            vv = np.flip(self.points_u[:,:,j],1)
                            if (self.vine_family == 'c-vine') & (cop_p.family == 'ind') & (j == 0):
                                    vv = self.points_u[:,:,j]
                            self.Fp[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                            
                        
#                         vv = np.flip(vv,1)   Does not change the order in the copulapdf it seems at least for gaussian
                        vv = vv[...,np.newaxis]
                        pd_points = np.squeeze(copulapdf(cop_p,vv))

                        # Update logf
                        logftr = tf.math.log(pd_points)
            
                        self.logf[:,ind_edge,tr+1] = np.squeeze(logftr) #.numpy()

                else:
                    
                    log_f_bin = np.zeros([tf.shape(self.logf)[0],n_eval],self.data_u.dtype)
                    for j in range(0,len(self.ind_edge_rel[tr]),1):
                        ind_edge = self.ind_edge_rel[tr][j]

#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
#                         val_to_bin = np.digitize(self.Fp[:,0,ind1], bins) -1
                        ind_now = edges_now[ind_edge]
                        parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                        ind1 = parent11
#                                 print('ind_now',ind_now)
#                                 print('par',parent11)
#                                 print('prev',ind_vine[tr-1][ind_now[0]],ind_vine[tr-1][ind_now[1]])
                        
                        if tr == 1:
                            bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                            val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                            val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                            val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                        else:
                            ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  
                            if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                            else:
                                bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)
                                val_to_bin1 = np.digitize(self.Fp_flip[:,tr-1,ind1], bins) -1
                        
                        for bb in range(0,self.n_bin,1):

                            cop_p = self.copulas[tr][ind_edge][bb]

                            mask = np.where(val_to_bin == bb)
                            mask1 = np.where(val_to_bin1 == bb)
                            data_u_bin = self.data_u[mask[0],:,j]

                            if self.flip_flag[tr][j] == True:

                                vv = self.points_u[:,:,j]
                                vv_bin = vv[mask[0],:]
                                
                                ### CDF FORCE UNIFORM
                                vv_bin_new = vv_bin
                                for zz in range(0,2,1):
                                    vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],vv_bin[:,zz],self.grid_u.ex)
                                vv_bin = vv_bin_new
                                ###
                                self.Fp_flip[mask[0],tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv_bin))
                            else:
                                vv = np.flip(self.points_u[:,:,j],1)
                                if (self.vine_family == 'c-vine') & (cop_p.family == 'ind') & (j == 0):
                                    vv = self.points_u[:,:,j]
        
                                vv_bin = vv[mask[0],:]
                                
                                ### CDF FORCE UNIFORM
                                vv_bin_new = vv_bin
                                for zz in range(0,2,1):
                                    vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],vv_bin[:,zz],self.grid_u.ex)
                                vv_bin = vv_bin_new
                                ###
                                
                                self.Fp[mask[0],tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv_bin))

                            vv_bin = vv_bin[...,np.newaxis]
                            pd_points = np.squeeze(copulapdf(cop_p,vv_bin))

                            ## Update logf
                            logftr = tf.math.log(pd_points) 
                            
                            self.logf[mask[0],ind_edge,tr+1] = logftr
                            
            
        logp = tf.math.reduce_sum(self.logf[:,:,0],1)
        logp_copula = tf.zeros(tf.shape(self.logf[:,0,0]),points.dtype)
        
        for i in tf.range(1,self.vine_depth,1,tf.int32):
#             unn, ind_un = np.unique(vine.ind_edge_rel[0],return_index=True)  ## Return unique index of ind_edge_rel
            logp = logp + tf.math.reduce_sum(self.logf[:,:,i],1)
            logp_copula = logp_copula + tf.math.reduce_sum(self.logf[:,:,i],1)

        #print('logp',logp)
        logp = tf.cast(logp,tf.float64)
        logp_copula = tf.cast(logp_copula,tf.float64)
        p = tf.exp(logp)
        p_copula = tf.exp(logp_copula)
        return p, p_copula

# class vine_obj_bin(object):
#     """Vine object.
#     """
#     def __init__(self, vine_family, families, vine_depth, margin, knots, *args):
#         """Create a marginal object.
#         Args:
#             families: Copula family.
#             theta: Correlation factor.
#             margin: Margin of the copula.
#         """
#         self.vine_family = vine_family
#         self.families = families
#         self.theta1 = []
#         self.theta2 = []
#         self.rang = None
#         self.n_cop = vine_depth
#         self.margin = margin
#         self.knots = knots
        
#         self.ind_vine = []
#         for i in range(0,self.n_cop-1,1):
#             self.ind_vine.append([])
        
        
#         if self.vine_family == 'r-vine':
#             self.method = args[0]
#             if self.method == 'matrix':
#                 self.r_matrix = args[1]
#                 self.E, self.ind_vine, self.nodes, self.matrix_edges = prepare_regular(self.r_matrix)
        
#         if (self.vine_family == 'c-vine') | (self.vine_family == 'd-vine'):
#             self.r_matrix, self.ind_vine, self.nodes, self.matrix_edges = prepare_vine(self.vine_family, self.n_cop)
                    
#         self.Mar_G = None
#         self.theta = None
#         self.Fp = None
#         self.logf = None
#         self.copulas = None
        
#         self.data_u = None
#         self.data_s = None
#         self.data_x = None
        
#         self.points_u = None
#         self.points_s = None
#         self.points_x = None
        
#         self.grid_u = None
#         self.grid_s = None
#         self.grid_x = None
        
#         self.binning = False
#         self.n_bin = 1
        
#     def fit(self, x, gen_dict, npc_dict, par_dict, bin_dict):
        
#         np_type = x.dtype
#         x = tf.convert_to_tensor(x)

#         ## Initialization
#         self.binning = gen_dict['binning']
#         self.parallel = gen_dict['parallel']
#         self.param = gen_dict['param']
#         self.fitted = gen_dict['fitted']
        
#         self.vine_depth = gen_dict['vine_depth']

#         d = x.shape[1]
#         self.n_cop = d
        
#         if self.param == False:
#             self.opt_method = npc_dict['opt_method']
#             batch_paral = npc_dict['batch_paral']
#         else:
#             param_families = par_dict['param_families']
#         if self.binning == True:
#             self.n_bin = bin_dict['n_bin']
        
#         ## Batches
#         batch_size_cdf = tf.constant(5,tf.int32)
#         if np.shape(x)[0] > 5000:
#             batch_size_cdf = tf.constant(10,tf.int32)
#         elif np.shape(x)[0] > 10000:
#             batch_size_cdf = tf.constant(100,tf.int32)
#         elif np.shape(x)[0] > 50000:
#             batch_size_cdf = tf.constant(200,tf.int32)
#         elif np.shape(x)[0] > 100000:
#             batch_size_cdf = tf.constant(500,tf.int32)
#         elif np.shape(x)[0] > 200000:
#             batch_size_cdf = tf.constant(1000,tf.int32)
#         elif np.shape(x)[0] > 500000:
#             batch_size_cdf = tf.constant(2000,tf.int32)
        
#         ## Make grid
#         u_1, ex_u = mk_grid(tf.convert_to_tensor(self.knots),np_type)
#         trans = Transform(self.n_cop)
        
#         ## Grid objects
#         self.grid_u = grid_obj(ex_u)
#         self.grid_s = grid_obj(trans.forward_u(ex_u))
        
#         ## Bivariate normal
#         x1_s, x2_s = self.grid_s.axis()
#         NORM = biv_norm(x1_s, x2_s)
#         self.grid_u.axis()
#         self.grid_s.min_grid()
#         self.grid_s.max_grid()
        
#         ## Create Mar_G, theta and Fp
#         self.Mar_G = []
#         self.theta_flip = np.zeros([tf.shape(x)[0],self.n_cop,self.n_cop],np_type)
#         self.theta = np.zeros([tf.shape(x)[0],self.n_cop,self.n_cop],np_type)
#         for i in range(0,self.n_cop,1):
#             ccc = self.margin[i].ker
#             interp_cdf, mar_s1, mar_p1 = kernel_cdf(ccc,ccc,ex_u)

#             self.Mar_G.append([mar_s1, mar_p1])
#             self.theta[:,0,i] = interp_cdf.numpy() #interp1d_np(ccc, mar_s1, mar_p1).numpy()
#             del ccc, mar_p1, mar_s1
        
#         ######################################### FITTING #######################################################
        
#         if self.fitted == False:
#             self.copulas = []
#         self.correlations = []
#         self.correlations_bins = []
#         self.flip_flag = []
#         self.ind_edge_rel = []
        
#         for tr in tf.range(0,self.vine_depth-1,1,tf.int32): #d-1
#             print('-----------------------------------')
#             print('Row theta:',tr.numpy())
            
#             if self.fitted == True:
#                 self.vine_family = 'r-vine'
#                 self.method = 'matrix'
            
#             print('theta:',self.theta[:,tr,:])

#             ## Number of copulas in the level
#             ## Create object for projections in the other spaces
            
#             n_cop = d-1-tr
#             trans = Transform(n_cop)
#             print('n_cop in the row:',n_cop.numpy())
            
#             ###### COMPUTE THE EDGES OF THE VINE LEVEL
            
#             if self.vine_family == 'r-vine':
                
#                 if self.method == 'matrix':   
#                     edges_now = self.ind_vine[tr]
                    
#                 elif (self.method == 'optimal') | (self.method == 'random'):   
                    
#                     random = False
                    
#                     if (self.method == 'random'):
#                         random = True
                        
#                     if tr == 0:
#                         self.r_matrix = np.zeros([self.n_cop,self.n_cop],np.int32)
#                         n = len(self.r_matrix) - 1
#                         ind_ee, weights = optimal_tree(self.theta[:,tr,:],self.theta_flip[:,tr,:],self.ind_vine,tr,random)
#                         edges_now = ind_ee
#                         self.ind_vine[tr] = ind_ee
#                         print('opt_tree',ind_ee)

#                         edges = []
#                         for j in range(0,len(ind_ee),1):
#                             edg = ind_ee[len(ind_ee)-1-j]
#                             self.r_matrix[n,j] = edg[0] +1
#                             self.r_matrix[j,j] = edg[1] +1
#                             edges.append({edg[0],edg[1]})

#                         edges = np.flip(edges)

#                         self.nodes = np.zeros(self.n_cop,np.int32)
#                         V = set(range(1,self.n_cop+1))
#                         for i in range(0,self.n_cop,1):
#                             self.nodes[i]=self.r_matrix[i,i]
#                             u_nod = {self.nodes[i]}
#                             if u_nod.issubset(V):
#                                 V.remove(self.nodes[i])
#                         self.nodes = np.flip(self.nodes)

#                         for elem in V:
#                             ind = np.where(self.nodes == 0)
#                             self.nodes[self.nodes == 0] = elem
#                             self.r_matrix[n-ind[0],n-ind[0]] = elem
#                     else:

#                         ind_ee, weights = optimal_tree(self.theta[:,tr,:],self.theta_flip[:,tr,:],self.ind_vine,tr,random)
#                         print('opt_tree',ind_ee)
#                         edges_now = ind_ee
#                         self.ind_vine[tr] = ind_ee

#             elif (self.vine_family == 'c-vine') | (self.vine_family == 'd-vine'):
#                 edges_now = self.ind_vine[tr]
            
            
#             ######### FROM THETA MATRIX TAKE THE DATA CDF FOR THE COPULA FITTING
#             # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
            
#             self.data_u = np.zeros([np.shape(self.theta)[0],2,n_cop],np_type)  

#             for j in range(0,len(edges_now),1):
#                 edge = edges_now[j]
                
#                 ## When tr = 0 there is no parent variable.
#                 ## After check if has to get the CDF from theta flip
#                 if tr == 0:
#                     self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
#                 else:
#                     parent1, inx1, inx2 = parent_var(tr,self.ind_vine,edge)

#                     if self.ind_vine[tr-1][edge[0]][0] != parent1: 
#                         self.data_u[:,:,j] = np.concatenate((self.theta_flip[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
#                     else:
#                         self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
            
#             ##### Transform data
#             self.data_s = trans.forward_u(self.data_u)
# #             self.data_s = check_bound3(self.data_s,tf.constant(3.2-1e-6,x.dtype),tf.constant(-3.2+1e-6,x.dtype))
#             self.data_x = trans.forward_s(self.data_s)
            
#             ##### Grid on P-Q space
#             self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))
            
#             ############################### FIT BANDWIDTH  #####################################
            
#             if self.fitted == False:
            
#                 #self.copulas.append([])
#                 opt_bw = tf.TensorArray(x.dtype,size=n_cop)

#                 if (tr == 0) | (self.binning == False):

#                     if self.parallel == False:

#                         if self.param == True:

#                             ### NOT BINNING, NOT PARALLEL, PARAMETRIC
#                             par_copulas = []
#                             tau_values = []
#                             n_cop1 = tf.constant(1,tf.int32)
#                             for j in range(0,len(edges_now),1):
                                
#                                 tau, p_value = kendalltau(self.data_u[:,0,j],self.data_u[:,1,j])
#                                 tau_values.append(tau)

#                                 start_time = perf_counter()
#                                 families = param_families # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
#                                 aic, theta_par, logp = parametric_fit(self.data_u[:,:,j][...,tf.newaxis], families, n_cop1)
#                                 time_fit_gauss = perf_counter()  - start_time

#                                 print('aic',aic)
#                                 print('theta_par',theta_par)

#                                 ind_fam = np.argmin(aic)
#                                 ## Gaussian
#                                 family = families[ind_fam]
#                                 theta_est = theta_par[0][ind_fam]

#                                 print('fam_fit',family)
#                                 print('theta_fit',theta_est)

#                                 cop_p = cop_par_obj(family,theta_est)
#                                 par_copulas.append(cop_p)

#                             self.copulas.append(par_copulas)
#                             self.correlations.append(tau_values)
#                         else: #param

#                             ### NOT BINNING, NOT PARALLEL, NOT PARAMETRIC
#                             opt_bw = tf.TensorArray(x.dtype,size=n_cop)
#                             tau_values = []
                            
#                             ## Batches
#                             batch_size = tf.constant(2,tf.int32)
#                             if np.shape(self.data_s)[0] > 10000:
#                                 batch_size = tf.constant(10,tf.int32)
#                             elif np.shape(self.data_s)[0] > 50000:
#                                 batch_size = tf.constant(20,tf.int32)
#                             elif np.shape(self.data_s)[0] > 100000:
#                                 batch_size = tf.constant(50,tf.int32)
#                             elif np.shape(self.data_s)[0] > 200000:
#                                 batch_size = tf.constant(100,tf.int32)
#                             elif np.shape(self.data_s)[0] > 500000:
#                                 batch_size = tf.constant(200,tf.int32)

#                             for i in range(0,n_cop,1):
#                                 print('col:',i)
                                
#                                 tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
#                                 tau_values.append(tau)

#                                 n_cop1 = tf.constant(1,tf.int32)

#                                 grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
#                                 data_dict = {'data_s':self.data_s[:,:,i], 'data_x':self.data_x[:,:,i]}
#                                 par_dict = {'n_cop':n_cop1, 'batch':batch_size, 'max_iter': [70,100], 'lr':[0.1, 0.03], #lr = 0.1, 0.01
#                                             'conv_tol': [1e-5,5e-5], 'opt_method': self.opt_method}  #1e-5

#                                 opt = optimization(grid_dict, data_dict, par_dict)
#                                 opt_bw = opt_bw.write(i,opt)

#                             opt_bw = opt_bw.stack()

#                             bw = bandwidth_mul(self.data_x,2,n_cop)
#                             bw1 = np.transpose(np.squeeze(opt_bw))*bw
                            
#                             ### Check constraints on the bandwidth
#                             bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(1e-2,x.dtype))  ##It was 5e-3 but too low

#                             copula = copula_obj(bw1.numpy())
#                             self.copulas.append(copula)
#                             self.correlations.append(tau_values)

#                             print('opt_bw',bw1)

#                     else: #Parallel

#                         if self.param == True:

#                             par_copulas = []
#                             tau_values = []
#     #                         if n_cop == 1:
#     #                             n_cop = 1  ## THIS BECAUSE THERE IS A PROBLEM IN SHAPE 'a' WITH FIT_STUDENT EVEN IF I FORCE IT TO BE THE SAME

#                             ### NOT BINNING, PARALLEL, PARAMETRIC
#                             start_time = perf_counter()
#                             families = param_families # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
#                             aic, theta_par, logp = parametric_fit(self.data_u, families, n_cop)
#                             time_fit_gauss = perf_counter()  - start_time

#                             print('aic',aic)
#                             print('theta_par',theta_par)
    
#                             for i in range(0,n_cop,1):
#                                 tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
#                                 tau_values.append(tau)
                                
#                                 ind_fam = np.argmin(aic[i])
#                                 family = families[ind_fam]
#                                 theta_est = theta_par[i][ind_fam]
#                                 print('fam_fit',family)
#                                 print('theta_fit',theta_est)
#                                 cop_p = cop_par_obj(family,theta_est)
#                                 par_copulas.append(cop_p)
#                             self.copulas.append(par_copulas)
#                             self.correlations.append(tau_values)

#                         else: #param

#                             ### NOT BINNING, PARALLEL, NOT PARAMETRIC
#                             n_cop1 = tf.constant(n_cop,tf.int32)

#                             batch_size = tf.constant(2,tf.int32)
#                             if np.shape(self.data_s)[0]*n_cop1 > 5000:
#                                 batch_size = tf.constant(10,tf.int32)
#                             elif np.shape(self.data_s)[0]*n_cop1 > 10000:
#                                 batch_size = tf.constant(20,tf.int32)
#                             elif np.shape(self.data_s)[0]*n_cop1 > 20000:
#                                 batch_size = tf.constant(50,tf.int32)
#                             elif np.shape(self.data_s)[0]*n_cop1 > 50000:
#                                 batch_size = tf.constant(100,tf.int32)
#                             elif np.shape(self.data_s)[0]*n_cop1 > 100000:
#                                 batch_size = tf.constant(200,tf.int32)
                            
#                             if self.opt_method == 'LL1':
#                                 opt_bw = np.zeros((1,n_cop1),np_type)
#                             else:
#                                 opt_bw = np.zeros((2,n_cop1),np_type)
                                
#                             batch_parallel = batch_paral
#                             batch_len1 = n_cop1/batch_parallel
#                             batch_len = tf.cast(batch_len1,tf.int32)
                            
#                             if batch_len <= 1:
#                                 batch_len = n_cop1
#                                 batch_parallel = 1
#                             else:
#                                 while batch_parallel*batch_len < n_cop1:
#                                         batch_parallel += 1
                            
#                             for j in range(0,batch_parallel,1):

#                                 grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,batch_len*j:batch_len*(j+1)]}
#                                 data_dict = {'data_s':self.data_s[:,:,batch_len*j:batch_len*(j+1)], 'data_x':self.data_x[:,:,batch_len*j:batch_len*(j+1)]}
#                                 par_dict = {'n_cop':tf.shape(self.data_s[:,:,batch_len*j:batch_len*(j+1)])[2], 'batch':batch_size, 'max_iter': [70,100], 'lr':[0.1, 0.03], 
#                                                 'conv_tol': [1e-5,5e-5], 'opt_method': self.opt_method}  ## 1e-5

#                                 opt = optimization(grid_dict, data_dict, par_dict)
#                                 print('opt',opt)
                                
#                                 opt_bw[:,batch_len*j:batch_len*(j+1)] = opt.numpy() #[...,tf.newaxis]
                                
#                             opt_bw = tf.convert_to_tensor(opt_bw)
                            
#                             bw = bandwidth_mul(self.data_x,2,n_cop)
#                             bw1 = np.transpose(np.squeeze(opt_bw))*bw
#                             bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(1e-2,x.dtype))  ##It was 5e-3 but too low

                            
#                             copula = copula_obj(bw1.numpy())
#                             self.copulas.append(copula)
                            
#                             tau_values = []
#                             for i in range(0,n_cop,1):
#                                 tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
#                                 tau_values.append(tau)
#                             self.correlations.append(tau_values)

#                             print('opt_bw',bw1)

#                 else: #Binning

#                     if self.parallel == False:

#                         if self.param == True:

#                             ### BINNING, NOT PARALLEL, PARAMETRIC

#                             par_copulas = []
#                             tau_values = []
#                             tau_val_bin = []
#                             n_cop1 = 1
#                             for j in range(0,len(edges_now),1):
                                
#                                 tau, p_value = kendalltau(self.data_u[:,0,j],self.data_u[:,1,j])
#                                 tau_values.append(tau)
#                                 print('Tau value before binning: ',tau)

#                                 ### BINNING, NOT PARALLEL, PARAMETRIC
#                                 parent11 = self.r_matrix[n-tr+1,n-1-j-tr]
#                                 ind1 = np.where(self.nodes == parent11)
#                                 ind1 = ind1[0][0]

#                                 bin_copulas = []
#                                 tau_binned = []
#                                 bins = create_bins(self.theta[:,0,ind1],self.n_bin)
#                                 val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
#                                 for bb in range(0,self.n_bin,1):
#                                     print('bin:',bb)
#                                     mask = np.where(val_to_bin == bb)
#                                     u_bin = self.data_u[mask[0],:,j][...,tf.newaxis]
                                    
#                                     tau, p_value = kendalltau(u_bin[:,0,0],u_bin[:,1,0])
#                                     print('Tau value bin -',bb, '- is: ', tau)
#                                     corr = stats.pearsonr(u_bin[:,0,0],u_bin[:,1,0])
#                                     print('Corr value  UV space: ',corr[0])
#                                     tau_binned.append(corr[0])

#                                     start_time = perf_counter()
#                                     families = param_families # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
#                                     aic, theta_par, logp = parametric_fit(u_bin, families, n_cop1)
#                                     time_fit_gauss = perf_counter()  - start_time

#                                     print('aic',aic)
#                                     print('theta_par',theta_par)

#                                     ind_fam = np.argmin(aic)

#                                     cop_p = cop_par_obj(families[ind_fam],theta_par[0][ind_fam])
#                                     bin_copulas.append(cop_p)

#                                     print('fam_fit',families[ind_fam])
#                                     print('theta_fit',theta_par[0][ind_fam])
#                                     print('--------------------')

#                                 par_copulas.append(bin_copulas)
#                                 tau_val_bin.append(tau_binned)

#                             self.copulas.append(par_copulas)
#                             self.correlations.append(tau_values)
#                             self.correlations_bins.append(tau_val_bin)

#                         else: #param

#                             ### BINNING, NOT PARALLEL, NOT PARAMETRIC
#                             n = len(self.r_matrix)-1
#                             tau_values = []
#                             tau_val_bin = []
                            
#                             for i in range(0,n_cop,1):
#                                 print('col:',i)
                                
#                                 tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
#                                 tau_values.append(tau)
#                                 print('Tau value before binning: ',tau)
                                
#                                 tau_binned = []
#                                 opt_bin = tf.TensorArray(x.dtype,size=self.n_bin)
#                                 parent = self.r_matrix[n-tr,n-2-i] - 1 # Because first node starts from 1 (and not from 0)
#                                 bins = create_bins(self.theta[:,0,parent],self.n_bin)
#                                 val_to_bin = np.digitize(self.theta[:,0,parent], bins) -1

#                                 for bb in range(0,self.n_bin,1):
#                                     print('bin:',bb)
#                                     mask = tf.where(tf.equal(val_to_bin,bb))
#                                     n_cop1 = tf.constant(1,tf.int32)
                                    
#                                     u_bin = tf.gather_nd(self.data_u[:,:,i],mask)
#                                     tau, p_value = kendalltau(u_bin[:,0],u_bin[:,1])
#                                     tau_binned.append(tau)
#                                     print('Tau value bin -',bb, '- is: ', tau)

#                                     data_s_bin = tf.gather_nd(self.data_s[:,:,i],mask)
#                                     data_x_bin = tf.gather_nd(self.data_x[:,:,i],mask)

#                                     grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
#                                     data_dict = {'data_s':data_s_bin, 'data_x':data_x_bin}
#                                     par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,70], 'lr':[0.1, 0.03],
#                                                 'conv_tol': [1e-4,1e-4], 'opt_method': self.opt_method}

#                                     opt = optimization(grid_dict, data_dict, par_dict)
                                    
#                                     bw = bandwidth_mul(data_x_bin,2,n_cop1)
#                                     bw1 = opt*bw
#                                     bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(1e-2,x.dtype))  ##It was 5e-3 but too low
                                    
#                                     opt_bin = opt_bin.write(bb,bw1)
# #                                     opt_bin = opt_bin.write(bb,opt)
#                                 opt_bin = opt_bin.stack()
# #                                 opt_bin = tf.reshape(opt_bin,[tf.shape(opt)[0],self.n_bin])
#                                 opt_bin = tf.reshape(opt_bin,[2,self.n_bin])

#                                 print(opt_bin)
#                                 tau_val_bin.append(tau_binned)

#                                 opt_bw = opt_bw.write(i,opt_bin)
#                             opt_bw = opt_bw.stack()
# #                             opt_bw = tf.reshape(opt_bw,[tf.shape(opt)[0],n_cop,self.n_bin])
#                             opt_bw = tf.reshape(opt_bw,[2,n_cop,self.n_bin])

# #                             bw = bandwidth_mul(self.data_x,2,n_cop)
                            
# #                             bw1 = np.zeros((2,n_cop,self.n_bin),np_type)
# #                             for i in range(0,n_cop,1):
# #                                 bw1[:,i,:] = opt_bw[:,i,:]*bw[:,i][...,np.newaxis]
                            
# #                             ### If bw < 5e-3 gives nan
# #                             bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(5e-3,x.dtype))
#                             bw1 = opt_bw
    
#                             copula = copula_obj(bw1.numpy())
#                             self.copulas.append(copula)
#                             self.correlations.append(tau_values)
#                             self.correlations_bins.append(tau_val_bin)

#                             print('opt_bw',bw1)

#                     else: #parallel

#                         if self.param == True:
#                             print('Miss to implement')
#                         else:
#                             tau_values = []
#                             tau_val_bin = []
                            
#                             len_bin = tf.shape(self.theta)[0]/self.n_bin
#                             len_bin = tf.cast(len_bin,tf.int32)
#                             data_s_bin = np.zeros([len_bin,tf.shape(self.data_s)[1],n_cop,self.n_bin],np_type)
#                             data_x_bin = np.zeros([len_bin,tf.shape(self.data_s)[1],n_cop,self.n_bin],np_type)

#                             for i in range(0,n_cop,1):
                                
#                                 tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
#                                 tau_values.append(tau)
#                                 print('Tau value before binning: ',tau)
                                
#                                 tau_binned = []
#                                 parent = self.r_matrix[n-tr,n-2-i] - 1 # Because first node starts from 1 (and not from 0)
#                                 bins = create_bins(self.theta[:,0,parent],self.n_bin)
#                                 val_to_bin = np.digitize(self.theta[:,0,parent], bins) -1

#                                 for bb in range(0,self.n_bin,1):
#                                     mask = tf.where(tf.equal(val_to_bin,bb))
#                                     data_s_bin[:,:,i,bb] = tf.gather_nd(self.data_s[:,:,i],mask)
#                                     data_x_bin[:,:,i,bb] = tf.gather_nd(self.data_x[:,:,i],mask)
                                    
#                                     u_bin = tf.gather_nd(self.data_u[:,:,i],mask)
#                                     tau, p_value = kendalltau(u_bin[:,0],u_bin[:,1])
#                                     tau_binned.append(tau)
#                                     print('Tau value bin -',bb, '- is: ', tau)
#                                     tau, p_value = stats.kendalltau(u_bin[:,0],u_bin[:,1])
#                                     print('Tau value bin -',bb, '- is: ', tau)
#                                     corr = stats.pearsonr(u_bin[:,0],u_bin[:,1])
#                                     print('Corr value  UV space: ',corr[0])
#                                 tau_val_bin.append(corr[0])


#                             opt_bw = np.zeros((2,n_cop,self.n_bin),np_type)
#                             batch_parallel = batch_paral
#                             batch_len1 = n_cop/batch_parallel
#                             batch_len = tf.cast(batch_len1,tf.int32)

#                             #                             if tf.cast(batch_len1,x.dtype) > tf.cast(batch_len,x.dtype):
#                             if batch_len <= 1:
#                                 batch_len = n_cop
#                                 batch_parallel = 1
#                             else:
#                                 while batch_parallel*batch_len < n_cop:
#                                         batch_parallel += 1

#                             for j in range(0,batch_parallel,1):

#                                 for bb in range(0,self.n_bin,1):
#                                     ## UPDATE THETA
#                                     n_batch = tf.shape(data_s_bin[:,:,batch_len*j:batch_len*(j+1),bb])[2]

#                                     grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,batch_len*j:batch_len*(j+1)]}
#                                     data_dict = {'data_s':data_s_bin[:,:,batch_len*j:batch_len*(j+1),bb], 'data_x':data_x_bin[:,:,batch_len*j:batch_len*(j+1),bb]}
#                                     par_dict = {'n_cop':n_batch, 'batch':batch_size, 
#                                                 'max_iter': [70,100], 'lr':[0.1, 0.03], 'conv_tol': [1e-4,1e-2], 'opt_method': self.opt_method}

#                                     opt = optimization(grid_dict, data_dict, par_dict)

#                                     bw = bandwidth_mul(data_x_bin[:,:,batch_len*j:batch_len*(j+1),bb],2,n_batch)

#                                     bw1 = opt*bw
#                                     bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(1e-2,x.dtype))

#                                     opt_bw[:,batch_len*j:batch_len*(j+1),bb] = bw1.numpy()
                            
#                             print('opt_bw',opt_bw)
#                             copula = copula_obj(opt_bw)
#                             self.copulas.append(copula)
#                             self.correlations.append(tau_values)
#                             self.correlations_bins.append(tau_val_bin)
                   
#             ##############################  UPDATE THETA #####################################
            
#             n = np.shape(self.r_matrix)[0] -1
            
#             #### if optimal or random, flip_flag = [True,False,True,False,...] in order to evaluate all possible orders
#             #### Otherwise just stores when to flip based on the parent variable
#             ## Flip_flap stores boolean if flipped or not
#             ## ind_edge_rel1 refers to the index of the copula
            
#             flip_flag1 = []
#             ind_edge_rel1 = []
#             parent_all = []
#             if (self.vine_family == 'r-vine'):
#                 if (self.method == 'optimal') | (self.method == 'random'):
#                     for j in range(0,len(edges_now),1):
#                         edge = edges_now[j]
#                         flip_flag1.append(True)
#                         flip_flag1.append(False)
#                         ind_edge_rel1.append(j)
#                         ind_edge_rel1.append(j)
#                         parent_all.append([edge[0],edge[1]])
#                 else:
#                     flip_flag1, ind_edge_rel1, parent_all = flip_check_all(self.ind_vine, tr, self.binning, self.n_bin)
#             else:
#                 flip_flag1, ind_edge_rel1, parent_all = flip_check_all(self.ind_vine, tr, self.binning, self.n_bin)
            
#             ## vv_s is another variable which stores the data_s taking into account also the flipping
            
#             vv_s = np.zeros((np.shape(self.data_u)[0],np.shape(self.data_u)[1],len(ind_edge_rel1)),self.data_u.dtype)

#             for j in range(0,len(ind_edge_rel1),1):
#                 ind_edge = ind_edge_rel1[j]
#                 edge = edges_now[ind_edge]

#                 if self.param == True:
#                     if (tr==0) | (self.binning == False):

#                         cop_p = self.copulas[tr][ind_edge]

#                         if flip_flag1[j] == True:
#                             vv = self.data_u[:,:,ind_edge]
#                             self.theta_flip[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
#                         else:
#                             vv = np.flip(self.data_u[:,:,ind_edge],1)
#                             self.theta[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))

#                     else: #binning

#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         flip_flag_bin = []
#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
#                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
#                         for bb in range(0,self.n_bin,1):

#                             cop_p = self.copulas[tr][ind_edge][bb]

#                             mask = np.where(val_to_bin == bb)

#                             if flip_flag1[j] == True:
#                                 vv = self.data_u[:,:,ind_edge]
#                                 vv_bin = vv[mask[0],:]
#                                 self.theta_flip[mask[0],tr+1,ind_edge] = copulaccdf(cop_p,vv_bin) 
#                             else:
#                                 vv = np.flip(self.data_u[:,:,ind_edge],1)
#                                 vv_bin = vv[mask[0],:]
#                                 self.theta[mask[0],tr+1,ind_edge] = copulaccdf(cop_p,vv_bin)
#                 else: #param
                    
#                     if flip_flag1[j] == True:
#                         vv_s[:,:,j] = np.flip(self.data_s[:,:,ind_edge],1) #self.data_s[:,:,j] #Flip cambia per npc
#                     else:
#                         vv_s[:,:,j] = self.data_s[:,:,ind_edge] #np.flip(self.data_s[:,:,j],1)

#             self.flip_flag.append(flip_flag1)
#             self.ind_edge_rel.append(ind_edge_rel1)

            
#             if self.param == False:
                
#                 n_eval = len(self.ind_edge_rel[tr])
#                 self.data_s = vv_s[:,:,:n_eval]
#                 trans = Transform(n_eval)
#                 self.data_x = trans.forward_s(self.data_s)
#                 grid_x = trans.forward_s(self.grid_s.ex)
                
#                 del vv_s
                
#                 if (tr == 0) | (self.binning == False):
                    
#                     batch_size = tf.constant(2,tf.int32)
#                     if np.shape(self.data_s)[0]*n_eval > 5000:
#                         batch_size = tf.constant(10,tf.int32)
#                     elif np.shape(self.data_s)[0]*n_eval > 10000:
#                         batch_size = tf.constant(20,tf.int32)
#                     elif np.shape(self.data_s)[0]*n_eval > 20000:
#                         batch_size = tf.constant(50,tf.int32)
#                     elif np.shape(self.data_s)[0]*n_eval > 50000:
#                         batch_size = tf.constant(100,tf.int32)
#                     elif np.shape(self.data_s)[0]*n_eval > 100000:
#                         batch_size = tf.constant(200,tf.int32)

#                     grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':grid_x}
#                     data_dict = {'data_s':self.data_s, 'data_x':self.data_x, 'theta':self.theta, 'theta_flip':self.theta_flip}
#                     par_dict = {'copulas': self.copulas[tr], 'n_eval':tf.convert_to_tensor(n_eval), 'batch':batch_size, 'batch_cdf':batch_size_cdf, 'tr':tr,
#                                'ind_edge_rel': self.ind_edge_rel[tr], 'flip_flag': self.flip_flag[tr]}

#                     self.copulas[tr].pd_grid_uv, self.copulas[tr].cdf, self.theta, self.theta_flip = evaluate_fit(data_dict, grid_dict, par_dict)
                    
#                 else: #binning
                    
#                     len_bin = tf.shape(self.theta)[0]/self.n_bin
#                     data_s_bin = np.empty([len_bin,tf.shape(self.data_s)[1],n_eval,self.n_bin],np_type)
#                     data_x_bin = np.empty([len_bin,tf.shape(self.data_s)[1],n_eval,self.n_bin],np_type)

#                     for i in range(0,n_eval,1):
#                         ind_edge = self.ind_edge_rel[tr][i]
#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
#                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
                        
#                         for bb in range(0,self.n_bin,1):
#                             mask = tf.where(tf.equal(val_to_bin,bb))
#                             data_s_bin[:,:,i,bb] = tf.gather_nd(self.data_s[:,:,i],mask)
#                             data_x_bin[:,:,i,bb] = tf.gather_nd(self.data_x[:,:,i],mask)
                
#                     self.copulas[tr].pd_grid_uv = np.zeros([self.knots,self.knots,n_eval,self.n_bin],np_type)
#                     self.copulas[tr].cdf = np.zeros([self.knots,self.knots,n_eval,self.n_bin],np_type)
                    
#                     for bb in range(0,self.n_bin,1):
#                         ## UPDATE THETA

#                         grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':grid_x}
#                         data_dict = {'data_s':data_s_bin[:,:,:,bb], 'data_x':data_x_bin[:,:,:,bb]} #],'bin':bb
#                         par_dict = {'bw': tf.convert_to_tensor(self.copulas[tr].opt_bw[:,:,bb]), 'n_cop':tf.convert_to_tensor(n_eval), 'batch':tf.constant(2,tf.int32), 'tr':tr, 'ind_edge_rel':self.ind_edge_rel[tr]}

#                         self.copulas[tr].pd_grid_uv[:,:,:,bb], self.copulas[tr].cdf[:,:,:,bb] = evaluate_fit_bin(data_dict, grid_dict, par_dict)
                    
#                     interp_cdf_bin = np.zeros([tf.shape(self.theta)[0],n_eval],np_type)
#                     for i in range(0,n_eval,1):
#                         print('col:',i)
#                         ind_edge = self.ind_edge_rel[tr][i]
#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)

#                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1

#                         for bb in range(0,self.n_bin,1):
# #                             print('bin:',bb)
                            
#                             mask = tf.where(tf.equal(val_to_bin,bb))

#                             ## Update theta  
#                             ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin[:,:,i,bb],self.grid_s.min,self.grid_s.max,self.copulas[tr].cdf[:,:,i,bb],axis=-2)
#                             ccdf_data = tf.squeeze(ccdf_data)
                            
#                             interp_cdf, mar_s, mar_p = kernel_cdf(ccdf_data,ccdf_data,self.grid_u.ex)
                            
#                             if self.flip_flag[tr][i] == True:
#                                 self.theta_flip[mask,tr+1,ind_edge] = tf.reshape(interp_cdf,[tf.shape(interp_cdf)[0],1])
#                             else:
#                                 self.theta[mask,tr+1,ind_edge] = tf.reshape(interp_cdf,[tf.shape(interp_cdf)[0],1])
        
#         ### After finding the optimal or the random vine, it stores the connection in the r_matrix
#         if self.vine_depth == self.n_cop:
#             if self.vine_family == 'r-vine':
#                 if (self.method == 'optimal') | (self.method == 'random'):
#                     self.r_matrix, self.E, self.nodes = prepare_optimal(self.n_cop,self.ind_vine)
        
#         return
    
    
#         ################################ EVALUATION #############################################
#     def evaluation(self, points):
        
#         d = self.n_cop
        
#         ## Create Fp
#         self.Fp = np.zeros([tf.shape(points)[0],self.n_cop,self.n_cop],self.data_u.dtype)
#         self.Fp_flip = np.zeros([tf.shape(points)[0],self.n_cop,self.n_cop],self.data_u.dtype)
#         ### Create logf
#         logf = tf.zeros([tf.shape(points)[0],tf.shape(points)[1],self.vine_depth],points.dtype)
#         self.logf_flip = tf.zeros(tf.shape(points),points.dtype)
        
#         for i in range(0,d,1):
#             interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(self.margin[i].ker,points[:,i],self.grid_u.ex)
#             self.Fp[:,0,i] = interp_cdf_poi.numpy()
            
#             den1,mden1 = kernel_pdf2(self.margin[i].ker)      
#             inter = interp_pdf(points[:,i], mden1, den1) #interp1d_np
            
#             # Product of pdf is the sum of logarithm - Product of pdf margingales evaluated on copula samples
#             logf = update_tensor(logf,tf.math.log(inter),i,0)
           
#             del den1,mden1, inter, interp_cdf_poi #,logf_tmp
            
#         self.logf = logf.numpy()
        
#         for tr in range(0,self.vine_depth-1,1): #d-1
#             print('Row theta:',tr)
            
#             if self.vine_family == 'r-vine':
#                     if (self.method == 'optimal') | (self.method == 'random'):
#                         self.flip_flag[tr], self.ind_edge_rel[tr], parent_all = flip_check_all(self.ind_vine, tr, self.binning, self.n_bin)
            
#             # Number of copuals to evaluate and create Transform object
            
#             n_eval = len(self.ind_edge_rel[tr])
#             trans = Transform(n_eval)
#             print('n to eval in the row:',n_eval)
            
#             ## Edges of the vine
#             edges_now = self.ind_vine[tr]
            
#             ######### FROM THETA MATRIX TAKE THE DATA CDF FOR THE COPULA FITTING
#             # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
            
#             self.data_u = np.zeros([np.shape(self.theta)[0],2,n_eval],self.data_u.dtype)
#             self.points_u = np.zeros([np.shape(self.Fp)[0],2,n_eval],self.data_u.dtype)
#             for j in range(0,n_eval,1):
#                 edge = edges_now[self.ind_edge_rel[tr][j]]
#                 if tr == 0:
#                     self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
#                     self.points_u[:,:,j] = np.concatenate((self.Fp[:,tr,edge[0]][...,np.newaxis],self.Fp[:,tr,edge[1]][...,np.newaxis]),1)
#                 else:
#                     parent1, inx1, inx2 = parent_var(tr,self.ind_vine,edge)
                    
#                     if self.ind_vine[tr-1][edge[0]][0] != parent1: 
#                         self.data_u[:,:,j] = np.concatenate((self.theta_flip[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
#                         self.points_u[:,:,j] = np.concatenate((self.Fp_flip[:,tr,edge[0]][...,np.newaxis],self.Fp[:,tr,edge[1]][...,np.newaxis]),1)
#                     else:
#                         self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
#                         self.points_u[:,:,j] = np.concatenate((self.Fp[:,tr,edge[0]][...,np.newaxis],self.Fp[:,tr,edge[1]][...,np.newaxis]),1)
                
#                 if self.param == False:
#                     if self.flip_flag[tr][j] == True:
#                         self.data_u[:,:,j] = np.flip(self.data_u[:,:,j],1)
#                         self.points_u[:,:,j] = np.flip(self.points_u[:,:,j],1)
            
#             ### Transform data
#             self.data_s = trans.forward_u(self.data_u)
# #             self.data_s = check_bound3(self.data_s,tf.constant(3.2-1e-6,points.dtype),tf.constant(-3.2+1e-6,points.dtype))
#             self.data_x = trans.forward_s(self.data_s)
            
#             ### Transform points
#             self.points_s = trans.forward_u(self.points_u)
# #             self.points_s = check_bound3(self.points_s,tf.constant(3.2-1e-6,points.dtype),tf.constant(-3.2+1e-6,points.dtype))
#             self.points_x = trans.forward_s(self.points_s)
            
#             ### Grid on P-Q space
#             self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))
            
#             n = np.shape(self.r_matrix)[0] -1
            
#             if self.param == False:
                
#                 pd_grid_uv = self.copulas[tr].pd_grid_uv
#                 cdf1 = self.copulas[tr].cdf

#                 if (tr==0) | (self.binning == False):

#                     for j in range(0,n_eval,1):
                        
#                         ind_edge = self.ind_edge_rel[tr][j]
                        
#                         ind_pd = j
#                         if self.vine_family == 'r-vine':
#                             if (self.method == 'optimal') | (self.method == 'random'):
#                                 ind_pd = self.ind_edge_rel[tr][j]*2
#                                 if self.flip_flag[tr][j] == True:
#                                     ind_pd = ind_pd
#                                 else:
#                                     ind_pd = ind_pd + 1
                        
#                         batch_size = tf.constant(2,tf.int32)
#                         if np.shape(self.data_s)[0] > 5000:
#                             batch_size = tf.constant(10,tf.int32)
#                         elif np.shape(self.data_s)[0] > 10000:
#                             batch_size = tf.constant(20,tf.int32)
#                         elif np.shape(self.data_s)[0] > 20000:
#                             batch_size = tf.constant(50,tf.int32)
#                         elif np.shape(self.data_s)[0] > 50000:
#                             batch_size = tf.constant(100,tf.int32)
#                         elif np.shape(self.data_s)[0] > 100000:
#                             batch_size = tf.constant(200,tf.int32)

#                         ccdf_data = tfp.math.batch_interp_regular_nd_grid(self.data_s[:,:,j],self.grid_s.min,self.grid_s.max,cdf1[:,:,ind_pd],axis=-2)
                        
#                         pd_points, ccdf_points = evaluate_points(self.points_s[:,:,j], batch_size, self.grid_s, cdf1[:,:,ind_pd], pd_grid_uv[:,:,ind_pd])   

#                         # Update logf
#                         logftr = tf.math.log(pd_points) 
                        
#                         self.logf[:,ind_edge,tr+1] = tf.squeeze(logftr).numpy() #update_tensor(logf,logftr,j,tr+1)

#                         # Update Fp
                        
#                         interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data,ccdf_points,self.grid_u.ex)

#                         if self.flip_flag[tr][j] == False:
#                             self.Fp[:,tr+1,ind_edge] = interp_cdf_poi
#                         else:
#                             self.Fp_flip[:,tr+1,ind_edge] = interp_cdf_poi

#                 else: #binning

#                     batch_size = tf.constant(1,tf.int32)

#                     len_bin = tf.shape(self.theta)[0]/self.n_bin
#                     data_s_bin = np.zeros([len_bin,tf.shape(self.data_s)[1],n_eval,self.n_bin],self.data_u.dtype)
#                     len_bin1 = tf.shape(self.Fp)[0]/self.n_bin                 
#                     points_s_bin = []

#                     for i in range(0,n_eval,1):
#                         ind_edge = self.ind_edge_rel[tr][i]

#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)

#                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
#                         val_to_bin1 = np.digitize(self.Fp[:,0,ind1], bins) -1

#                         points_s_bin1 = []
#                         for bb in range(0,self.n_bin,1):

#                             mask = tf.where(tf.equal(val_to_bin,bb))
#                             mask1 = tf.where(tf.equal(val_to_bin1,bb))

#                             data_s_bin[:,:,i,bb] = tf.gather_nd(self.data_s[:,:,i],mask)
#                             points_s_bin1.append(tf.gather_nd(self.points_s[:,:,i],mask1))

#                         points_s_bin.append(points_s_bin1)

#                     log_f_bin = np.zeros([tf.shape(self.logf)[0],n_eval],self.data_u.dtype)
#                     Fp_bin = np.zeros([tf.shape(self.Fp)[0],n_eval],self.data_u.dtype)
#                     for i in range(0,n_eval,1):

#                         ind_edge = self.ind_edge_rel[tr][i]
#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
#                         val_to_bin1 = np.digitize(self.Fp[:,0,ind1], bins) -1

#                         for bb in range(0,self.n_bin,1):
#                             mask1 = tf.where(tf.equal(val_to_bin1,bb))

#                             ## Update theta  
#                             ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin[:,:,i,bb],self.grid_s.min,self.grid_s.max,self.copulas[tr].cdf[:,:,i,bb],axis=-2)
#                             ccdf_data = tf.squeeze(ccdf_data)
                
#                             pd_points, ccdf_points = evaluate_points(points_s_bin[i][bb], batch_size, self.grid_s, cdf1[:,:,i,bb], pd_grid_uv[:,:,i,bb]) 

#                             ## Update logf
#                             logftr = tf.math.log(pd_points) 
                
#                             self.logf[tf.squeeze(mask1),ind_edge,tr+1] = tf.squeeze(logftr).numpy()

#                             ## Update Fp
#                             interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data,ccdf_points,self.grid_u.ex)
#                             Fp_bin[mask1,i] = tf.reshape(interp_cdf_poi,[tf.shape(interp_cdf_poi)[0],1])

#                         Fp_bin1 = tf.squeeze(Fp_bin[:,i])

#                         if self.flip_flag[tr][i] == True:
#                             self.Fp_flip[:,tr+1,ind_edge] = Fp_bin1
#                         else:
#                             self.Fp[:,tr+1,ind_edge] = Fp_bin1

#             else:

#                 if (tr==0) | (self.binning == False):
                    
#                     for j in range(0,len(self.ind_edge_rel[tr]),1):

#                         ind_edge = self.ind_edge_rel[tr][j]

#                         cop_p = self.copulas[tr][ind_edge]

#                         if self.flip_flag[tr][j] == True:
#                             vv = self.points_u[:,:,j]
#                             self.Fp_flip[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
#                         else:
#                             vv = np.flip(self.points_u[:,:,j],1)
#                             if (self.vine_family == 'c-vine') & (cop_p.family == 'ind') & (j == 0):
#                                     vv = self.points_u[:,:,j]
#                             self.Fp[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                            
                        
# #                         vv = np.flip(vv,1)   Does not change the order in the copulapdf it seems at least for gaussian
#                         vv = vv[...,np.newaxis]
#                         pd_points = np.squeeze(copulapdf(cop_p,vv))

#                         # Update logf
#                         logftr = tf.math.log(pd_points)
            
#                         self.logf[:,ind_edge,tr+1] = np.squeeze(logftr) #.numpy()

#                 else:
                    
#                     log_f_bin = np.zeros([tf.shape(self.logf)[0],n_eval],self.data_u.dtype)
#                     for j in range(0,len(self.ind_edge_rel[tr]),1):
#                         ind_edge = self.ind_edge_rel[tr][j]

#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
#                         val_to_bin = np.digitize(self.Fp[:,0,ind1], bins) -1
#                         for bb in range(0,self.n_bin,1):

#                             cop_p = self.copulas[tr][ind_edge][bb]

#                             mask = np.where(val_to_bin == bb)

#                             if self.flip_flag[tr][j] == True:

#                                 vv = self.points_u[:,:,j]
#                                 vv_bin = vv[mask[0],:]
#                                 self.Fp_flip[mask[0],tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv_bin))
#                             else:
#                                 vv = np.flip(self.points_u[:,:,j],1)
#                                 if (self.vine_family == 'c-vine') & (cop_p.family == 'ind') & (j == 0):
#                                     vv = self.points_u[:,:,j]
        
#                                 vv_bin = vv[mask[0],:]
#                                 self.Fp[mask[0],tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv_bin))

#                             vv_bin = vv_bin[...,np.newaxis]
#                             pd_points = np.squeeze(copulapdf(cop_p,vv_bin))

#                             ## Update logf
#                             logftr = tf.math.log(pd_points) 
                            
#                             self.logf[mask[0],ind_edge,tr+1] = logftr
                            
            
#         logp = tf.math.reduce_sum(self.logf[:,:,0],1)
#         logp_copula = tf.zeros(tf.shape(self.logf[:,0,0]),points.dtype)
        
#         for i in tf.range(1,self.vine_depth,1,tf.int32):
#             logp = logp + tf.math.reduce_sum(self.logf[:,:,i],1)
#             logp_copula = logp_copula + tf.math.reduce_sum(self.logf[:,:,i],1)

#         #print('logp',logp)
#         logp = tf.cast(logp,tf.float64)
#         logp_copula = tf.cast(logp_copula,tf.float64)
#         p = tf.exp(logp)
#         p_copula = tf.exp(logp_copula)
#         return p, p_copula


# class vine_obj_bin(object):
#     """Vine object.
#     """
#     def __init__(self, vine_family, families, vine_depth, margin, knots, *args):
#         """Create a marginal object.
#         Args:
#             families: Copula family.
#             theta: Correlation factor.
#             margin: Margin of the copula.
#         """
#         self.vine_family = vine_family
#         self.families = families
#         self.theta1 = []
#         self.theta2 = []
#         self.rang = None
#         self.n_cop = vine_depth
#         self.margin = margin
#         self.knots = knots
        
#         self.ind_vine = []
#         for i in range(0,self.n_cop-1,1):
#             self.ind_vine.append([])
        
        
#         if self.vine_family == 'r-vine':
#             self.method = args[0]
#             if self.method == 'matrix':
#                 self.r_matrix = args[1]
#                 self.E, self.ind_vine, self.nodes, self.matrix_edges = prepare_regular(self.r_matrix)
# #                 self.E, self.ind_vine, self.nodes, self.matrix_edges = prepare_regular(self.r_matrix)
# #                 self.E = build_edges(self.r_matrix)
        
#         if self.vine_family == 'c-vine':
# #             self.r_matrix = np.tril(np.tile(np.array(range(self.n_cop,0,-1)),(self.n_cop,1)).T)
# #             self.E, self.ind_vine, self.nodes, self.matrix_edges = prepare_regular(self.r_matrix)
            
#             self.r_matrix, self.ind_vine, self.nodes, self.matrix_edges = prepare_vine(self.vine_family, self.n_cop)
# #             self.E = build_edges(self.r_matrix)
        
#         if self.vine_family == 'd-vine':
# #             self.r_matrix = np.zeros((self.n_cop,self.n_cop),np.int32)
# #             for i in range(0,self.n_cop,1):
# #                 self.r_matrix[i,i] = self.n_cop-i #-1
# #             for j in range(0,self.n_cop-1,1):
# #                 c = 1 #0
# #                 for i in range(j+1,self.n_cop,1):
# #                     self.r_matrix[i,j] = c
# #                     c += 1
# # #             self.E = build_edges(self.r_matrix)
# #             self.E, self.ind_vine, self.nodes, self.matrix_edges = prepare_regular(self.r_matrix)
    
#             self.r_matrix, self.ind_vine, self.nodes, self.matrix_edges = prepare_vine(self.vine_family, self.n_cop)
            
# #         self.E, self.ind_vine, self.nodes, self.matrix_edges = prepare_regular(self.r_matrix)
                    
#         self.Mar_G = None
#         self.theta = None
#         self.Fp = None
#         self.logf = None
#         self.copulas = None
        
#         self.data_u = None
#         self.data_s = None
#         self.data_x = None
        
#         self.points_u = None
#         self.points_s = None
#         self.points_x = None
        
#         self.grid_u = None
#         self.grid_s = None
#         self.grid_x = None
        
#         self.binning = False
#         self.n_bin = 1
# #         self.err_trace = tf.Variable(initial_value=tf.ones([1],x.dtype),dtype=x.dtype, trainable=False)
# #         self.pos_trace = tf.Variable(initial_value=tf.ones([1],x.dtype),dtype=x.dtype, trainable=False)
# #         self.iter_err = tf.Variable(1,dtype=tf.int32, trainable=False)
# #         self.a = tf.Variable(initial_value=tf.random.uniform(shape=[1], minval=2e-3, maxval=2, dtype=x.dtype), trainable=False)
        
#     def fit(self, x, gen_dict, npc_dict, par_dict, bin_dict):
        
#         np_type = x.dtype
#         x = tf.convert_to_tensor(x)

#         ## Initialization
#         self.binning = gen_dict['binning']
#         self.parallel = gen_dict['parallel']
#         self.param = gen_dict['param']
        
#         self.vine_depth = gen_dict['vine_depth']
# #         d = self.n_cop
#         d = x.shape[1]
#         self.n_cop = d
        
#         if self.param == False:
#             self.opt_method = npc_dict['opt_method']
#         else:
#             param_families = par_dict['param_families']
#         if self.binning == True:
#             self.n_bin = bin_dict['n_bin']
        
#         ## Make grid
#         u_1, ex_u = mk_grid(tf.convert_to_tensor(self.knots),np_type)
#         trans = Transform(self.n_cop)
        
#         ## Grid objects
#         self.grid_u = grid_obj(ex_u)
#         self.grid_s = grid_obj(trans.forward_u(ex_u))
        
#         ## Bivariate normal
#         x1_s, x2_s = self.grid_s.axis()
#         NORM = biv_norm(x1_s, x2_s)
        
#         ## Create Mar_G, theta and Fp
#         self.Mar_G = []
#         self.theta_flip = np.zeros([tf.shape(x)[0],self.n_cop,self.n_cop],np_type)
#         self.theta = np.zeros([tf.shape(x)[0],self.n_cop,self.n_cop],np_type)
#         for i in range(0,self.n_cop,1):
#             ccc = self.margin[i].ker
#             mar_p1, mar_s1 = kernel_cdf(ccc, ex_u)
#             self.Mar_G.append([mar_s1, mar_p1])
#             self.theta[:,0,i] = interp1d_np(ccc, mar_s1, mar_p1).numpy()
#             del ccc, mar_p1, mar_s1
        
#         ############### FITTING ####################
#         self.copulas = []
#         self.flip_flag = []
#         self.ind_edge_rel = []
# #         family2 = []
# #         theta2 = []
        
#         for tr in tf.range(0,self.vine_depth-1,1,tf.int32): #d-1
#             print('-----------------------------------')
#             print('Row theta:',tr.numpy())
            
#             print('theta:',self.theta[:,tr,:])
#             print('theta shape:',tf.shape(self.theta[:,tr,:]))

#             # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
#             n_cop = d-1-tr
#             trans = Transform(n_cop)
#             print('n_cop in the row:',n_cop.numpy())
            
#             #### EDGES OF THE VINE
#             if self.vine_family == 'r-vine':
#                 if self.method == 'matrix':   
#                     edges_now = self.ind_vine[tr]
#                 elif (self.method == 'optimal') | (self.method == 'random'):   
#                     random = False
#                     if (self.method == 'random'):
#                         random = True
#                     if tr == 0:
                        
#                         self.r_matrix = np.zeros([self.n_cop,self.n_cop],np.int32)
#                         n = len(self.r_matrix) - 1
#                         ind_ee, weights = optimal_tree(self.theta[:,tr,:],self.theta_flip[:,tr,:],self.ind_vine,tr,random)
#                         edges_now = ind_ee
#                         self.ind_vine[tr] = ind_ee
#                         print('opt_tree',ind_ee)

#                         edges = []
#                         for j in range(0,len(ind_ee),1):
#                             edg = ind_ee[len(ind_ee)-1-j]
#                             self.r_matrix[n,j] = edg[0] +1
#                             self.r_matrix[j,j] = edg[1] +1
#                             edges.append({edg[0],edg[1]})
                        
#                         edges = np.flip(edges)
                        
# #                         self.r_matrix = np.zeros([self.n_cop,self.n_cop],np.int32)
# #                         n = len(self.r_matrix) - 1
# #                         ind_ee, weights = optimal_tree(self.theta[:,tr,:],self.theta_flip[:,tr,:],self.ind_vine,tr)
# #                         self.ind_vine[tr] = ind_ee
# #                         edges_now = ind_ee
# #                         edges = []
# #                         cc = 0
# #                         for edg in ind_ee:
# #                             self.r_matrix[cc,cc] = edg[1] +1
# #                             self.r_matrix[n,cc] = edg[0] +1
# #                             edges.append({edg[0],edg[1]})
# #                             cc += 1
# #                         print('opt_tree',ind_ee)
                        
#                         self.nodes = np.zeros(self.n_cop,np.int32)
#                         V = set(range(1,self.n_cop+1))
#                         for i in range(0,self.n_cop,1):
#                             self.nodes[i]=self.r_matrix[i,i]
#                             u_nod = {self.nodes[i]}
#                             if u_nod.issubset(V):
#                                 V.remove(self.nodes[i])
#                         self.nodes = np.flip(self.nodes)

#                         for elem in V:
#                             ind = np.where(self.nodes == 0)
#                             self.nodes[self.nodes == 0] = elem
#                             self.r_matrix[n-ind[0],n-ind[0]] = elem
                        
#                     else:
                        
#                         ind_ee, weights = optimal_tree(self.theta[:,tr,:],self.theta_flip[:,tr,:],self.ind_vine,tr,random)
#                         print('opt_tree',ind_ee)
#                         edges_now = ind_ee
#                         self.ind_vine[tr] = ind_ee
#                         to_remove = set()

#                         for j in range(0,len(ind_ee),1):
#                             edg = ind_ee[j]
#                             if tr == 1:
#                                 parent = edges[edg[0]].intersection(edges[edg[1]])
#                                 u_union = edges[edg[0]].union(edges[edg[1]])
#                             else:
#                                 parent = u_union1[edg[0]].intersection(u_union1[edg[1]])
#                                 u_union = u_union1[edg[0]].union(u_union1[edg[1]])
#                             diff = u_union - parent
#                             diff1 = diff - to_remove
                            
#                             for jj in range(0,len(ind_ee),1):
#                                 inxx = {self.r_matrix[jj,jj]-1}
#                                 if (inxx.issubset(diff1)) & ({self.r_matrix[n-tr+1,jj]-1}.issubset(parent)):
#                                     ind1 = jj
#                                     for inx1 in inxx:
#                                         to_remove.add(inx1)
#                             diff.remove(r_matrix[ind1,ind1]-1)
                            
#                             for elem in diff:
#                                 self.r_matrix[n-tr,ind1] = elem +1
#                             u_union = set()
#                             parent = set()
#                             diff = set()
                            
# #                             diff.remove(self.r_matrix[j,j]-1)
# #                             for elem in diff:
# #                                 self.r_matrix[n-tr,j] = elem +1
                        
#                         if tr > 1:
#                             u_union2 = u_union1
#                         u_union1 = []
#                         for j in range(0,len(ind_ee),1):
#                             edg = ind_ee[j]
#                             if tr == 1:
#                                 u_union1.append(edges[edg[0]].union(edges[edg[1]]))
#                             else:
#                                 u_union1.append(u_union2[edg[0]].union(u_union2[edg[1]]))
                        
# #                         ind_ee, weights = optimal_tree(self.theta[:,tr,:],self.theta_flip[:,tr,:],self.ind_vine,tr)
# #                         self.ind_vine[tr] = ind_ee
# #                         edges_now = ind_ee
# #                         n = len(self.r_matrix) - 1
# #                         cc = 0
# #                         for edg in ind_ee:
# #                             parent = edges[edg[0]].intersection(edges[edg[1]])
# #                             for par in parent:
# #                                 self.r_matrix[n-tr,cc] = par +1
# #                             cc += 1
# #                         edges = []
# #                         cc = 0
# #                         for edg in ind_ee:
# #                             edges.append({edg[0],edg[1]})
# #                             cc += 1
# #                         print('opt_tree',ind_ee)
#             elif (self.vine_family == 'c-vine') | (self.vine_family == 'd-vine'):
# #                 ind_ee = edges_index(self.E,self.r_matrix,tr)
#                 edges_now = self.ind_vine[tr]
            
            
# #             ro = 0
# #             for ind in ind_ee:
# #                 data_u[:,0,ro] = self.theta[:,tr,ind[0]]
# #                 data_u[:,1,ro] = self.theta[:,tr,ind[1]]
# #                 ro += 1
            
#             self.data_u = np.zeros([np.shape(self.theta)[0],2,n_cop],np_type)  

#             for j in range(0,len(edges_now),1):
#                 edge = edges_now[j]
#                 if tr == 0:
#                     self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
#                 else:
#                     parent1, inx1, inx2 = parent_var(tr,self.ind_vine,edge)

#                     if self.ind_vine[tr-1][edge[0]][0] != parent1: 
#                         self.data_u[:,:,j] = np.concatenate((self.theta_flip[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
#                     else:
#                         self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
            
#             ## Transform data
#             self.data_s = trans.forward_u(self.data_u)
#             self.data_x = trans.forward_s(self.data_s)
            
#             ## Grid on P-Q space
#             self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))

            
#             #self.copulas.append([])
#             opt_bw = tf.TensorArray(x.dtype,size=n_cop)
            
#             if (tr == 0) | (self.binning == False):
                
#                 if self.parallel == False:
                    
#                     if self.param == True:
                        
#                         ### NOT BINNING, NOT PARALLEL, PARAMETRIC
#                         par_copulas = []
#                         n_cop1 = tf.constant(1,tf.int32)
#                         for j in range(0,len(edges_now),1):

#                             start_time = perf_counter()
#                             families = param_families # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
#                             aic, theta_par, logp = parametric_fit(self.data_u[:,:,j][...,tf.newaxis], families, n_cop1)
#                             time_fit_gauss = perf_counter()  - start_time

#                             print('aic',aic)
#                             print('theta_par',theta_par)

#                             ind_fam = np.argmin(aic)
#                             ## Gaussian
#                             family = families[ind_fam]
#                             theta_est = theta_par[0][ind_fam]

#                             print('fam_fit',family)
#                             print('theta_fit',theta_est)
                            
#                             cop_p = cop_par_obj(family,theta_est)
#                             par_copulas.append(cop_p)
                            
#                         self.copulas.append(par_copulas)
#                     else: #param
                        
#                         ### NOT BINNING, NOT PARALLEL, NOT PARAMETRIC
#                         opt_bw = tf.TensorArray(x.dtype,size=n_cop)
                
#                         for i in range(0,n_cop,1):
#                             print('col:',i)

#                             n_cop1 = tf.constant(1,tf.int32)

#                             grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
#                             data_dict = {'data_s':self.data_s[:,:,i], 'data_x':self.data_x[:,:,i]}
#                             par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,100], 'lr':[0.05, 0.01], #lr = 0.1, 0.01
#                                         'conv_tol': [0.000001,0.0000001], 'opt_method': self.opt_method}

#                             opt = optimization(grid_dict, data_dict, par_dict)
#                             opt_bw = opt_bw.write(i,opt)

#                         opt_bw = opt_bw.stack()
                        
#                         copula = copula_obj(opt_bw)
#                         self.copulas.append(copula)
                        
#                         print('opt_bw',opt_bw)
                
#                 else: #Parallel
                    
#                     if self.param == True:
                        
#                         par_copulas = []
# #                         if n_cop == 1:
# #                             n_cop = 1  ## THIS BECAUSE THERE IS A PROBLEM IN SHAPE 'a' WITH FIT_STUDENT EVEN IF I FORCE IT TO BE THE SAME

#                         ### NOT BINNING, PARALLEL, PARAMETRIC
#                         start_time = perf_counter()
#                         families = param_families # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
#                         aic, theta_par, logp = parametric_fit(self.data_u, families, n_cop)
#                         time_fit_gauss = perf_counter()  - start_time

#                         print('aic',aic)
#                         print('theta_par',theta_par)
                        
#                         for i in range(0,n_cop,1):
#                             ind_fam = np.argmin(aic[i])
#                             family = families[ind_fam]
#                             theta_est = theta_par[i][ind_fam]
#                             print('fam_fit',family)
#                             print('theta_fit',theta_est)
#                             cop_p = cop_par_obj(family,theta_est)
#                             par_copulas.append(cop_p)
#                         self.copulas.append(par_copulas)
                        
#                     else: #param
                        
#                         ### NOT BINNING, PARALLEL, NOT PARAMETRIC
#                         n_cop1 = tf.constant(n_cop,tf.int32)
                        
#                         batch_size = tf.constant(2,tf.int32)
#                         if np.shape(self.data_s)[0]*n_cop1 > 5000:
#                             batch_size = tf.constant(10,tf.int32)
#                         elif np.shape(self.data_s)[0]*n_cop1 > 10000:
#                             batch_size = tf.constant(20,tf.int32)
#                         elif np.shape(self.data_s)[0]*n_cop1 > 20000:
#                             batch_size = tf.constant(50,tf.int32)
#                         elif np.shape(self.data_s)[0]*n_cop1 > 50000:
#                             batch_size = tf.constant(100,tf.int32)
#                         elif np.shape(self.data_s)[0]*n_cop1 > 100000:
#                             batch_size = tf.constant(200,tf.int32)

#                         grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex}
#                         data_dict = {'data_s':self.data_s, 'data_x':self.data_x}
#                         par_dict = {'n_cop':n_cop1, 'batch':batch_size, 'max_iter': [85,125], 'lr':[0.1, 0.01], 
#                                         'conv_tol': [0.000001,0.0000001], 'opt_method': self.opt_method}

#                         opt = optimization(grid_dict, data_dict, par_dict)
#                         opt_bw = opt[...,tf.newaxis]
                        
#                         copula = copula_obj(opt_bw)
#                         self.copulas.append(copula)
                        
#                         print('opt_bw',opt_bw)
                        
#             else: #Binning
                
#                 if self.parallel == False:
                    
#                     if self.param == True:
                        
#                         ### NOT BINNING, NOT PARALLEL, PARAMETRIC

#                         par_copulas = []
#                         n_cop1 = 1
#                         for j in range(0,len(edges_now),1):
                        
#                             ### BINNING, NOT PARALLEL, PARAMETRIC
#                             parent11 = self.r_matrix[n-tr+1,n-1-j-tr]
#                             ind1 = np.where(self.nodes == parent11)
#                             ind1 = ind1[0][0]

#                             bin_copulas = []
#                             bins = create_bins(self.theta[:,0,ind1],self.n_bin)
#                             val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
#                             for bb in range(0,self.n_bin,1):
#                                 print('bin:',bb)
#                                 mask = np.where(val_to_bin == bb)
#                                 u_bin = self.data_u[mask[0],:,j][...,tf.newaxis]

#                                 start_time = perf_counter()
#                                 families = param_families # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
#                                 aic, theta_par, logp = parametric_fit(u_bin, families, n_cop1)
#                                 time_fit_gauss = perf_counter()  - start_time

#                                 print('aic',aic)
#                                 print('theta_par',theta_par)

#                                 ind_fam = np.argmin(aic)
                                
#                                 cop_p = cop_par_obj(families[ind_fam],theta_par[0][ind_fam])
#                                 bin_copulas.append(cop_p)

#                                 print('fam_fit',families[ind_fam])
#                                 print('theta_fit',theta_par[0][ind_fam])
#                                 print('--------------------')
                            
#                             par_copulas.append(bin_copulas)
                        
#                         self.copulas.append(par_copulas)
                    
#                     else: #param
                        
#                         ### BINNING, NOT PARALLEL, NOT PARAMETRIC
#                         n = len(self.r_matrix)-1

#                         for i in range(0,n_cop,1):
#                             print('col:',i)

#                             opt_bin = tf.TensorArray(x.dtype,size=self.n_bin)
#                             parent = self.r_matrix[n-tr,n-2-i] - 1 # Because first node starts from 1 (and not from 0)
#                             bins = create_bins(self.theta[:,0,parent],self.n_bin)
#                             val_to_bin = np.digitize(self.theta[:,0,parent], bins) -1

#                             for bb in range(0,self.n_bin,1):
#                                 print('bin:',bb)
#                                 mask = tf.where(tf.equal(val_to_bin,bb))
#                                 n_cop1 = tf.constant(1,tf.int32)
                                
#                                 data_s_bin = tf.gather_nd(self.data_s[:,:,i],mask)
#                                 data_x_bin = tf.gather_nd(self.data_x[:,:,i],mask)

#                                 grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
#                                 data_dict = {'data_s':data_s_bin, 'data_x':data_x_bin}
#                                 par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,100], 'lr':[0.1, 0.01],
#                                             'conv_tol': [0.0001,0.0001], 'opt_method': self.opt_method}

#                                 opt = optimization(grid_dict, data_dict, par_dict)
#                                 opt_bin = opt_bin.write(bb,opt)
#                             opt_bin = opt_bin.stack()
#                             opt_bin = tf.reshape(opt_bin,[tf.shape(opt)[0],self.n_bin])
                            
#                             print(opt_bin)
                
#                             opt_bw = opt_bw.write(i,opt_bin)
#                         opt_bw = opt_bw.stack()
#                         opt_bw = tf.reshape(opt_bw,[tf.shape(opt)[0],n_cop,self.n_bin])
                        
#                         copula = copula_obj(opt_bw)
#                         self.copulas.append(copula)
                        
#                         print('opt_bw',opt_bw)
                        
#                 else: #parallel
                    
#                     if self.param == True:
#                         print('Miss to implement')
#                     else:
#                         print('Miss to implement')
                    
#             ##############################  UPDATE THETA #####################################
            
#             n = np.shape(self.r_matrix)[0] -1
# #             if tr < n-1:  #n-1
            
#             #######################
#             flip_flag1 = []
#             ind_edge_rel1 = []
#             parent_all = []
#             if (self.vine_family == 'r-vine'):
#                 if (self.method == 'optimal') | (self.method == 'random'):
#                     for j in range(0,len(edges_now),1):
#                         edge = edges_now[j]
#                         flip_flag1.append(True)
#                         flip_flag1.append(False)
#                         ind_edge_rel1.append(j)
#                         ind_edge_rel1.append(j)
#                         parent_all.append([edge[0],edge[1]])
#                 else:
#                     flip_flag1, ind_edge_rel1, parent_all = flip_check_all(self.ind_vine, tr, self.binning, self.n_bin)
#             else:
#                 flip_flag1, ind_edge_rel1, parent_all = flip_check_all(self.ind_vine, tr, self.binning, self.n_bin)
                
#             vv_s = np.zeros((np.shape(self.data_u)[0],np.shape(self.data_u)[1],len(ind_edge_rel1)),self.data_u.dtype)

#             for j in range(0,len(ind_edge_rel1),1):
#                 ind_edge = ind_edge_rel1[j]
#                 edge = edges_now[ind_edge]

#                 if self.param == True:
#                     if (tr==0) | (self.binning == False):

#                         cop_p = self.copulas[tr][ind_edge]

#                         if flip_flag1[j] == True:
#                             vv = self.data_u[:,:,ind_edge]
#                             self.theta_flip[:,tr+1,ind_edge] = copulaccdf(cop_p,vv) 
#                         else:
#                             vv = np.flip(self.data_u[:,:,ind_edge],1)
#                             self.theta[:,tr+1,ind_edge] = copulaccdf(cop_p,vv)

#                     else: #binning

#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         flip_flag_bin = []
#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
#                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
#                         for bb in range(0,self.n_bin,1):

#                             cop_p = self.copulas[tr][ind_edge][bb]

#                             mask = np.where(val_to_bin == bb)

#                             if flip_flag1[j] == True:
#                                 vv = self.data_u[:,:,ind_edge]
#                                 vv_bin = vv[mask[0],:]
#                                 self.theta_flip[mask[0],tr+1,ind_edge] = copulaccdf(cop_p,vv_bin) 
#                             else:
#                                 vv = np.flip(self.data_u[:,:,ind_edge],1)
#                                 vv_bin = vv[mask[0],:]
#                                 self.theta[mask[0],tr+1,ind_edge] = copulaccdf(cop_p,vv_bin)
#                 else: #param
                    
#                     if flip_flag1[j] == True:
#                         vv_s[:,:,j] = np.flip(self.data_s[:,:,ind_edge],1) #self.data_s[:,:,j] #Flip cambia per npc
#                     else:
#                         vv_s[:,:,j] = self.data_s[:,:,ind_edge] #np.flip(self.data_s[:,:,j],1)

#             self.flip_flag.append(flip_flag1)
#             self.ind_edge_rel.append(ind_edge_rel1)
# #             print('ind-edge',ind_edge_rel1)
# #             print('flip_flag',flip_flag1)
            
#             if self.param == False:
                
#                 n_eval = len(self.ind_edge_rel[tr])
#                 self.data_s = vv_s[:,:,:n_eval]
#                 trans = Transform(n_eval)
#                 self.data_x = trans.forward_s(self.data_s)
#                 grid_x = trans.forward_s(self.grid_s.ex)
                
#                 if (tr == 0) | (self.binning == False):
                    
#                     batch_size = tf.constant(2,tf.int32)
#                     if n_eval > 10:
#                         batch_size = tf.constant(10,tf.int32)
#                     elif n_eval > 30:
#                         batch_size = tf.constant(20,tf.int32)

#                     grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':grid_x}
#                     data_dict = {'data_s':self.data_s, 'data_x':self.data_x, 'theta':self.theta, 'theta_flip':self.theta_flip}
#                     par_dict = {'copulas': self.copulas[tr], 'n_eval':tf.convert_to_tensor(n_eval), 'batch':batch_size, 'tr':tr,
#                                'ind_edge_rel': self.ind_edge_rel[tr], 'flip_flag': self.flip_flag[tr]}

#                     self.copulas[tr].pd_grid_uv, self.copulas[tr].cdf, self.theta, self.theta_flip = evaluate_fit(data_dict, grid_dict, par_dict)
                    
#                 else: #binning
                    
#                     len_bin = tf.shape(self.theta)[0]/self.n_bin
#                     data_s_bin = np.empty([len_bin,tf.shape(self.data_s)[1],n_eval,self.n_bin],np_type)
#                     data_x_bin = np.empty([len_bin,tf.shape(self.data_s)[1],n_eval,self.n_bin],np_type)

#                     for i in range(0,n_eval,1):
#                         ind_edge = self.ind_edge_rel[tr][i]
#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
#                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
                        
#                         for bb in range(0,self.n_bin,1):
#                             mask = tf.where(tf.equal(val_to_bin,bb))
#                             data_s_bin[:,:,i,bb] = tf.gather_nd(self.data_s[:,:,i],mask)
#                             data_x_bin[:,:,i,bb] = tf.gather_nd(self.data_x[:,:,i],mask)
                
#                     self.copulas[tr].pd_grid_uv = np.zeros([self.knots,self.knots,n_eval,self.n_bin],np_type)
#                     self.copulas[tr].cdf = np.zeros([self.knots,self.knots,n_eval,self.n_bin],np_type)
                    
#                     for bb in range(0,self.n_bin,1):
#                         ## UPDATE THETA

#                         grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':grid_x}
#                         data_dict = {'data_s':data_s_bin[:,:,:,bb], 'data_x':data_x_bin[:,:,:,bb],'bin':bb}
#                         par_dict = {'copulas': self.copulas[tr], 'n_cop':tf.convert_to_tensor(n_eval), 'batch':tf.constant(2,tf.int32), 'tr':tr, 'ind_edge_rel':self.ind_edge_rel[tr]}

#                         self.copulas[tr].pd_grid_uv[:,:,:,bb], self.copulas[tr].cdf[:,:,:,bb] = evaluate_fit_bin(data_dict, grid_dict, par_dict)
                    
#                     interp_cdf_bin = np.zeros([tf.shape(self.theta)[0],n_eval],np_type)
#                     for i in range(0,n_eval,1):
#                         print('col:',i)
#                         ind_edge = self.ind_edge_rel[tr][i]
#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)

#                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1

#                         for bb in range(0,self.n_bin,1):
#                             print('bin:',bb)
                            
#                             mask = tf.where(tf.equal(val_to_bin,bb))

#                             ## Update theta  
#                             ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin[:,:,i,bb],self.grid_s.min,self.grid_s.max,self.copulas[tr].cdf[:,:,i,bb],axis=-2)
#                             ccdf_data = tf.squeeze(ccdf_data)
#                             mar_p1, mar_s1 = kernel_cdf(ccdf_data, self.grid_u.ex)

#                             interp_cdf = interp1d_np(ccdf_data, mar_s1, mar_p1)
#                             if self.flip_flag[tr][i] == True:
#                                 self.theta_flip[mask,tr+1,ind_edge] = tf.reshape(interp_cdf,[tf.shape(interp_cdf)[0],1])
#                             else:
#                                 self.theta[mask,tr+1,ind_edge] = tf.reshape(interp_cdf,[tf.shape(interp_cdf)[0],1])
                    
                    
#         return
    
    
#         ################################ EVALUATION #############################################
#     def evaluation(self, points):
        
#         d = self.n_cop
        
#         ## Create Fp
#         self.Fp = np.zeros([tf.shape(points)[0],self.n_cop,self.n_cop],self.data_u.dtype)
#         self.Fp_flip = np.zeros([tf.shape(points)[0],self.n_cop,self.n_cop],self.data_u.dtype)
#         for i in range(0,self.n_cop,1):
#             self.Fp[:,0,i] = interp1d_np(points[:,i], self.Mar_G[i][0], self.Mar_G[i][1]).numpy()

        
#         ### Create logf
#         en = tf.TensorArray(points.dtype,size=self.n_cop)
#         logf = tf.zeros(tf.shape(points),points.dtype)
#         self.logf_flip = tf.zeros(tf.shape(points),points.dtype)
#         for i in tf.range(0,d,1,tf.int32):
#             den1,mden1 = kernel_pdf2(self.margin[i].ker)      
#             inter = interp_pdf(points[:,i], mden1, den1) #interp1d_np
#             # Product of pdf is the sum of logarithm - Product of pdf margingales evaluated on copula samples
#             #logf = logf + tf.math.log(inter)
#             logf_tmp = logf[:,0] + tf.math.log(inter)
#             logf = update_tensor2D(logf,0,logf_tmp)

#             m_diff = mden1[1:] - mden1[:-1]
#             m_diff = tf.concat([m_diff, tf.expand_dims(m_diff[-1], 0)], 0)

#             # log 2 of the pdf on the grid
#             log_pd = tf.py_function(np.log2, [den1], den1.dtype)
#             log_pd = replace_inf(log_pd, tf.constant(den1.dtype.min,den1.dtype))
#             en = en.write(i,- tf.math.reduce_sum(den1*log_pd*tf.transpose(m_diff),0))  #log_pd
#             #vec = tf.linspace(tf.math.reduce_min(vine1.margin[i].ker),tf.math.reduce_max(vine1.margin[i].ker),100)
#             #inter_en = interp1d_np(vec, mden1, den1)
#             #en = en.write(i,tf.py_function(stats.entropy, [inter_en,vec], vec.dtype))
#             del den1,mden1,logf_tmp,m_diff,log_pd
#         en = en.stack()
#         self.logf = logf.numpy()
        
#         for tr in range(0,self.vine_depth-1,1): #d-1
#             print('Row theta:',tr)
            
#             if self.vine_family == 'r-vine':
#                     if self.method == 'optimal':
#                         self.flip_flag[tr], self.ind_edge_rel[tr], parent_all = flip_check_all(self.ind_vine,tr, self.binning, self.n_bin)
            
#             # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
#             n_eval = len(self.ind_edge_rel[tr])#d-1-tr
#             trans = Transform(n_eval)
#             print('n to eval in the row:',n_eval)
            
#             self.data_u = np.zeros([np.shape(self.theta)[0],2,n_eval],self.data_u.dtype)
#             self.points_u = np.zeros([np.shape(self.Fp)[0],2,n_eval],self.data_u.dtype)
            
#             #### EDGES OF THE VINE
#             edges_now = self.ind_vine[tr]
            
#             for j in range(0,n_eval,1):
#                 edge = edges_now[self.ind_edge_rel[tr][j]]
#                 if tr == 0:
#                     self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
#                     self.points_u[:,:,j] = np.concatenate((self.Fp[:,tr,edge[0]][...,np.newaxis],self.Fp[:,tr,edge[1]][...,np.newaxis]),1)
#                 else:
#                     parent1, inx1, inx2 = parent_var(tr,self.ind_vine,edge)
                    
#                     if self.ind_vine[tr-1][edge[0]][0] != parent1: 
#                         self.data_u[:,:,j] = np.concatenate((self.theta_flip[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
#                         self.points_u[:,:,j] = np.concatenate((self.Fp_flip[:,tr,edge[0]][...,np.newaxis],self.Fp[:,tr,edge[1]][...,np.newaxis]),1)
#                     else:
#                         self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
#                         self.points_u[:,:,j] = np.concatenate((self.Fp[:,tr,edge[0]][...,np.newaxis],self.Fp[:,tr,edge[1]][...,np.newaxis]),1)
                
#                 if self.param == False:
#                     if self.flip_flag[tr][j] == True:
#                         self.data_u[:,:,j] = np.flip(self.data_u[:,:,j],1)
#                         self.points_u[:,:,j] = np.flip(self.points_u[:,:,j],1)
            
#             ## Transform data
#             self.data_s = trans.forward_u(self.data_u)
#             self.data_x = trans.forward_s(self.data_s)
            
#             ## Transform points
#             self.points_s = trans.forward_u(self.points_u)
#             self.points_x = trans.forward_s(self.points_s)
            
#             ## Grid on P-Q space
#             self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))
            
            
#             n = np.shape(self.r_matrix)[0] -1
            
            
#             if self.param == False:
                
#                 pd_grid_uv = self.copulas[tr].pd_grid_uv
#                 cdf1 = self.copulas[tr].cdf


#                 if (tr==0) | (self.binning == False):

#                     for j in range(0,n_eval,1):

#                         batch_size = tf.constant(2,tf.int32)

#                         ccdf_data = tfp.math.batch_interp_regular_nd_grid(self.data_s[:,:,j],self.grid_s.min,self.grid_s.max,cdf1[:,:,j],axis=-2)
#                         mar_p1, mar_s1 = kernel_cdf(ccdf_data, self.grid_u.ex)

#                         pd_points, ccdf_points = evaluate_points(self.points_s[:,:,j], batch_size, self.grid_s, cdf1[:,:,j], pd_grid_uv[:,:,j])    

#                         # Update logf
#                         logftr = tf.math.log(pd_points) 
#                         logf_tmp = self.logf[:,tr+1] + tf.squeeze(logftr)
#                         self.logf = update_tensor2D(self.logf,tr+1,logf_tmp)

#                         # Update Fp
#                         interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
#                         if self.flip_flag[tr][j] == False:
#                             self.Fp[:,tr+1,self.ind_edge_rel[tr][j]] = interp_cdf_poi
#                         else:
#                             self.Fp_flip[:,tr+1,self.ind_edge_rel[tr][j]] = interp_cdf_poi

#                 else: #binning

#                     batch_size = tf.constant(1,tf.int32)

#                     len_bin = tf.shape(self.theta)[0]/self.n_bin
#                     data_s_bin = np.zeros([len_bin,tf.shape(self.data_s)[1],n_eval,self.n_bin],self.data_u.dtype)
#                     len_bin1 = tf.shape(self.Fp)[0]/self.n_bin                 
#                     points_s_bin = []

#                     for i in range(0,n_eval,1):
#                         ind_edge = self.ind_edge_rel[tr][i]

#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)

#                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
#                         val_to_bin1 = np.digitize(self.Fp[:,0,ind1], bins) -1

#                         points_s_bin1 = []
#                         for bb in range(0,self.n_bin,1):

#                             mask = tf.where(tf.equal(val_to_bin,bb))
#                             mask1 = tf.where(tf.equal(val_to_bin1,bb))

#                             data_s_bin[:,:,i,bb] = tf.gather_nd(self.data_s[:,:,i],mask)
#                             points_s_bin1.append(tf.gather_nd(self.points_s[:,:,i],mask1))

#                         points_s_bin.append(points_s_bin1)

#                     log_f_bin = np.zeros([tf.shape(self.logf)[0],n_eval],self.data_u.dtype)
#                     Fp_bin = np.zeros([tf.shape(self.Fp)[0],n_eval],self.data_u.dtype)
#                     for i in range(0,n_eval,1):
#                         print('col:',i)
#                         ind_edge = self.ind_edge_rel[tr][i]
#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]
#                         print('ind1',ind1)

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
#                         val_to_bin1 = np.digitize(self.Fp[:,0,ind1], bins) -1

# #                             parent = self.r_matrix[n-tr,n-2-i] - 1 # Because first node starts from 1 (and not from 0)

# #                             bins = create_bins(self.theta[:,0,parent],n_bin)
# #                             val_to_bin1 = np.digitize(self.Fp[:,0,parent], bins) -1

#                         for bb in range(0,self.n_bin,1):
#                             print('bin:',bb)
#                             mask1 = tf.where(tf.equal(val_to_bin1,bb))

# #                                 print('shape bin',np.shape(points_s_bin[i][bb]))

#                             ## Update theta  
#                             ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin[:,:,i,bb],self.grid_s.min,self.grid_s.max,self.copulas[tr].cdf[:,:,i,bb],axis=-2)
#                             ccdf_data = tf.squeeze(ccdf_data)
#                             mar_p1, mar_s1 = kernel_cdf(ccdf_data, self.grid_u.ex)
#                             pd_points, ccdf_points = evaluate_points(points_s_bin[i][bb], batch_size, self.grid_s, cdf1[:,:,i,bb], pd_grid_uv[:,:,i,bb]) 

#                             # Update logf
#                             logftr = tf.math.log(pd_points) 
#                             logf_tmp = tf.gather_nd(self.logf[:,tr+1],mask1)
# #                             print('logf_tmp',logf_tmp)
#                             logf_tmp = logf_tmp + tf.squeeze(logftr)

#                             log_f_bin[mask1,i] = tf.reshape(logf_tmp,[tf.shape(logf_tmp)[0],1])

#                             # Update Fp
#                             interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
#                             Fp_bin[mask1,i] = tf.reshape(interp_cdf_poi,[tf.shape(interp_cdf_poi)[0],1])

#                         log_f_bin1 = tf.squeeze(log_f_bin[:,i])
#                         Fp_bin1 = tf.squeeze(Fp_bin[:,i])
#                         self.logf = update_tensor2D(self.logf,tr+1,log_f_bin1)

#                         if self.flip_flag[tr][i] == True:
#                             self.Fp_flip[:,tr+1,ind_edge] = Fp_bin1
#                         else:
#                             self.Fp[:,tr+1,ind_edge] = Fp_bin1

#             else:

#                 if (tr==0) | (self.binning == False):
                    
#                     for j in range(0,len(self.ind_edge_rel[tr]),1):
# #                         print('j',j)
#                         ind_edge = self.ind_edge_rel[tr][j]
# #                         print('ind_edge', ind_edge)
# #                         print('flip', self.flip_flag[tr][j])

#                         cop_p = self.copulas[tr][ind_edge]
# #                         print('family: ',cop_p.family,'theta: ',cop_p.theta)

#                         if self.flip_flag[tr][j] == True:
#                             vv = self.points_u[:,:,j][...,tf.newaxis]
#                             self.Fp_flip[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
#                         else:
#                             vv = np.flip(self.points_u[:,:,j][...,tf.newaxis],1)
#                             self.Fp[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                            
                        
# #                         vv = np.flip(vv,1)   Does not change the order in the copulapdf it seems at least for gaussian
#                         pd_points = np.squeeze(copulapdf(cop_p,vv))

#                         # Update logf
#                         logftr = tf.math.log(pd_points) 
#                         logf_tmp = self.logf[:,tr+1] + tf.squeeze(logftr)
#                         self.logf = update_tensor2D(self.logf,tr+1,logf_tmp)

#                 else:
                    
#                     log_f_bin = np.zeros([tf.shape(self.logf)[0],n_eval],self.data_u.dtype)
#                     for j in range(0,len(self.ind_edge_rel[tr]),1):
#                         ind_edge = self.ind_edge_rel[tr][j]

#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
#                         val_to_bin = np.digitize(self.Fp[:,0,ind1], bins) -1
#                         for bb in range(0,self.n_bin,1):

#                             cop_p = self.copulas[tr][ind_edge][bb]

#                             mask = np.where(val_to_bin == bb)
# #                             mask = tf.where(tf.equal(val_to_bin1,bb))

#                             if self.flip_flag[tr][j] == True:
#                                 vv = self.points_u[:,:,j][...,tf.newaxis]
#                                 vv_bin = vv[mask[0],:,:]
#                                 self.Fp_flip[mask[0],tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv_bin))
#                             else:
#                                 vv = np.flip(self.points_u[:,:,j][...,tf.newaxis],1)
#                                 vv_bin = vv[mask[0],:,:]
#                                 self.Fp[mask[0],tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv_bin))
                            
# #                             print('vv bin', vv_bin)
# #                             vv_bin = np.flip(vv_bin,1)
#                             pd_points = np.squeeze(copulapdf(cop_p,vv_bin))
# #                             print('pd_points',pd_points)
#                             # Update logf
#                             logftr = tf.math.log(pd_points) 
# #                             print('logftr',logftr)
#                             mask1 = np.reshape(mask,(np.shape(mask[0])[0],1))
#                             logf_tmp = tf.gather_nd(self.logf[:,tr+1],mask1)
#                             logf_tmp = logf_tmp + tf.squeeze(logftr)

#                             log_f_bin[mask[0],ind_edge] = logf_tmp # tf.reshape(logf_tmp,[tf.shape(logf_tmp)[0],1])

#                         log_f_bin1 = tf.squeeze(log_f_bin[:,ind_edge])
#                         self.logf = update_tensor2D(self.logf,tr+1,log_f_bin1)
        
#         logp = self.logf[:,0]
#         logp_copula = tf.zeros(tf.shape(self.logf[:,0]),points.dtype)
#         for i in tf.range(1,self.n_cop,1,tf.int32):
#             #print('loghi',logf[:,i])
#             logp = logp + self.logf[:,i]
#             logp_copula = logp_copula + self.logf[:,i]
#         #print('logp',logp)
#         p = tf.exp(logp)
#         p_copula = tf.exp(logp_copula)
#         return p, p_copula



        
class vine_obj(object):
    """Vine object.
    """
    def __init__(self, vine_family, families, vine_depth, margin, knots, *args):
        """Create a marginal object.
        Args:
            families: Copula family.
            theta: Correlation factor.
            margin: Margin of the copula.
        """
        self.vine_family = vine_family
        self.families = families
        self.theta1 = []
        self.theta2 = []
        self.rang = None
        self.n_cop = vine_depth
        self.margin = margin
        self.knots = knots
        
        if self.vine_family == 'r-vine':
            self.method = args[0]
            if self.method == 'matrix':
                self.r_matrix = args[1]
                self.E = build_edges(self.r_matrix)
        
        if self.vine_family == 'c-vine':
            self.r_matrix = np.tril(np.tile(np.array(range(self.n_cop,0,-1)),(self.n_cop,1)).T)
            self.E = build_edges(self.r_matrix)
                    
        self.n_cop = None
        self.Mar_G = None
        self.theta = None
        self.Fp = None
        self.logf = None
        self.copulas = None
        
        self.data_u = None
        self.data_s = None
        self.data_x = None
        
        self.points_u = None
        self.points_s = None
        self.points_x = None
        
        self.grid_u = None
        self.grid_s = None
        self.grid_x = None
#         self.err_trace = tf.Variable(initial_value=tf.ones([1],x.dtype),dtype=x.dtype, trainable=False)
#         self.pos_trace = tf.Variable(initial_value=tf.ones([1],x.dtype),dtype=x.dtype, trainable=False)
#         self.iter_err = tf.Variable(1,dtype=tf.int32, trainable=False)
#         self.a = tf.Variable(initial_value=tf.random.uniform(shape=[1], minval=2e-3, maxval=2, dtype=x.dtype), trainable=False)
        
    def fit(self, x, parallel, opt_method):
        
        ## Initialization
        d = x.shape[1]
        self.n_cop = d
        
        ## Make grid
        u_1, ex_u = mk_grid(self.knots,x.dtype)
        trans = Transform(self.n_cop)
        
        ## Grid objects
        self.grid_u = grid_obj(ex_u)
        self.grid_s = grid_obj(trans.forward_u(ex_u))
        
        ## Bivariate normal
        x1_s, x2_s = self.grid_s.axis()
        NORM = biv_norm(x1_s, x2_s)
        
        ## Create Mar_G, theta and Fp
        Mar_G = []
        theta = np.empty([x.shape[0],self.n_cop,self.n_cop],x.dtype)
        for i in range(0,self.n_cop,1):
            ccc = self.margin[i].ker
            mar_p1, mar_s1 = kernel_cdf(ccc, ex_u)
            Mar_G.append([mar_s1, mar_p1])
            theta[:,0,i] = interp1d_np(ccc, mar_s1, mar_p1).numpy()
            del ccc, mar_p1, mar_s1
            
        self.Mar_G = Mar_G
        self.theta = theta
        
        ############### FITTING ####################
        self.copulas = []
        
        for tr in tf.range(0,d-1,1,tf.int32): #d-1
            print('Row theta:',tr.numpy())
            
            print('theta:',self.theta[:,tr,:])

            # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
            n_cop = d-1-tr
            trans = Transform(n_cop)
            print('n_cop in the row:',n_cop.numpy())
            
            data_u = np.empty([theta.shape[0],2,n_cop],x.dtype)  
            
            #### EDGES OF THE VINE
            if self.vine_family == 'r-vine':
                if self.method == 'matrix':   
                    ind_ee = edges_index(self.E,self.r_matrix,tr)
                elif self.method == 'optimal':   
                    if tr == 0:
                        ind_ee, weights = optimal_tree(self.theta[:,tr,:])
                    else:
                        ind_ee, weights = optimal_tree(self.theta[:,tr,:-tr])
            elif self.vine_family == 'c-vine':
                ind_ee = edges_index(self.E,self.r_matrix,tr)
            
            ro = 0
            for ind in ind_ee:
                data_u[:,0,ro] = self.theta[:,tr,ind[0]]
                data_u[:,1,ro] = self.theta[:,tr,ind[1]]
                ro += 1
            
            ## Transform data
            self.data_u = data_u
            self.data_s = trans.forward_u(self.data_u)
            self.data_x = trans.forward_s(self.data_s)
            
            ## Grid on P-Q space
            self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))
            
            #print(data_u)
            
            #self.copulas.append([])
            opt_bw = tf.TensorArray(x.dtype,size=n_cop)
            
            if parallel == True:
                n_cop1 = tf.constant(n_cop,tf.int32)

                grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex}
                data_dict = {'data_s':self.data_s, 'data_x':self.data_x}
                par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,100], 'lr':[0.1, 0.01], 
                                'conv_tol': [0.000001,0.0000001], 'opt_method': opt_method}

                opt = optimization(grid_dict, data_dict, par_dict)
                opt_bw = opt            
            elif parallel == False:
                opt_bw = tf.TensorArray(x.dtype,size=n_cop)
            
                for i in range(0,n_cop,1):
                    print('col:',i)

                    n_cop1 = tf.constant(1,tf.int32)

                    grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
                    data_dict = {'data_s':self.data_s[:,:,i], 'data_x':self.data_x[:,:,i]}
                    par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,100], 'lr':[0.1, 0.01], 
                                'conv_tol': [0.000001,0.0000001], 'opt_method': opt_method}
    
                    opt = optimization(grid_dict, data_dict, par_dict)
                    opt_bw = opt_bw.write(i,opt)

                opt_bw = opt_bw.stack()         
            
            copula = copula_obj(opt_bw)
            self.copulas.append(copula)
            
            ## UPDATE THETA
            
            grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x}
            data_dict = {'data_s':self.data_s, 'data_x':self.data_x, 'theta':self.theta}
            par_dict = {'copulas': self.copulas[tr], 'n_cop':tf.convert_to_tensor(n_cop), 'batch':tf.constant(2,tf.int32), 'tr':tr}
            
            self.copulas[tr].pd_grid_uv, self.copulas[tr].cdf, self.theta = evaluate_fit(data_dict, grid_dict, par_dict)
            
        return

    def evaluation(self, points):
        
        d = tf.shape(points)[1]
        
        ## Create Fp
        Fp = np.empty([tf.shape(points)[0],self.n_cop,self.n_cop],self.data_u.dtype)
        for i in range(0,self.n_cop,1):
            Fp[:,0,i] = interp1d_np(points[:,i], self.Mar_G[i][0], self.Mar_G[i][1]).numpy()
        self.Fp = Fp
        
        ### Create logf
        en = tf.TensorArray(points.dtype,size=d)
        logf = tf.zeros(tf.shape(points),points.dtype)
        for i in tf.range(0,d,1,tf.int32):
            den1,mden1 = kernel_pdf2(self.margin[i].ker)      
            inter = interp_pdf(points[:,i], mden1, den1) #interp1d_np
            # Product of pdf is the sum of logarithm - Product of pdf margingales evaluated on copula samples
            #logf = logf + tf.math.log(inter)
            logf_tmp = logf[:,0] + tf.math.log(inter)
            logf = update_tensor2D(logf,0,logf_tmp)

            m_diff = mden1[1:] - mden1[:-1]
            m_diff = tf.concat([m_diff, tf.expand_dims(m_diff[-1], 0)], 0)

            # log 2 of the pdf on the grid
            log_pd = tf.py_function(np.log2, [den1], den1.dtype)
            log_pd = replace_inf(log_pd, tf.constant(den1.dtype.min,den1.dtype))
            en = en.write(i,- tf.math.reduce_sum(den1*log_pd*tf.transpose(m_diff),0))  #log_pd
            #vec = tf.linspace(tf.math.reduce_min(vine1.margin[i].ker),tf.math.reduce_max(vine1.margin[i].ker),100)
            #inter_en = interp1d_np(vec, mden1, den1)
            #en = en.write(i,tf.py_function(stats.entropy, [inter_en,vec], vec.dtype))
            del den1,mden1,logf_tmp,m_diff,log_pd
        en = en.stack()
        self.logf = logf.numpy()
        
        for tr in tf.range(0,d-1,1,tf.int32): #d-1
            print('Row theta:',tr.numpy())
            
            # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
            n_cop = d-1-tr
            trans = Transform(n_cop)
            print('n_cop in the row:',n_cop.numpy())
            
            data_u = np.empty([self.theta.shape[0],2,n_cop],self.data_u.dtype)
            points_u = np.empty([self.Fp.shape[0],2,n_cop],self.data_u.dtype)
            
            #### EDGES OF THE VINE
            if self.vine_family == 'r-vine':
                if self.method == 'matrix':   
                    ind_ee = edges_index(self.E,self.r_matrix,tr)
                elif self.method == 'optimal':   
                    if tr == 0:
                        ind_ee, weights = optimal_tree(self.theta[:,tr,:])
                    else:
                        ind_ee, weights = optimal_tree(self.theta[:,tr,:-tr])
            elif self.vine_family == 'c-vine':
                ind_ee = edges_index(self.E,self.r_matrix,tr)
            
            ro = 0
            for ind in ind_ee:
                data_u[:,0,ro] = self.theta[:,tr,ind[0]]
                data_u[:,1,ro] = self.theta[:,tr,ind[1]]
                points_u[:,0,ro] = self.Fp[:,tr,ind[0]]
                points_u[:,1,ro] = self.Fp[:,tr,ind[1]]
                ro += 1
            
            ## Transform data
            self.data_u = data_u
            self.data_s = trans.forward_u(self.data_u)
            self.data_x = trans.forward_s(self.data_s)
            
            ## Transform points
            self.points_u = points_u
            self.points_s = trans.forward_u(self.points_u)
            self.points_x = trans.forward_s(self.points_s)
            
            ## Grid on P-Q space
            self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))
            
            #print(data_u)
            grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x}
            data_dict = {'data_s':self.data_s, 'data_x':self.data_x, 'theta':self.theta}
            par_dict = {'copulas': self.copulas[tr], 'n_cop':tf.convert_to_tensor(n_cop), 'batch':tf.constant(2,tf.int32), 'tr':tr}
            
#             pd_grid_uv, cdf1, self.theta = evaluate_fit(data_dict, grid_dict, par_dict)
    
            pd_grid_uv = self.copulas[tr].pd_grid_uv
            cdf1 = self.copulas[tr].cdf
    
            batch_size = tf.constant(2,tf.int32)
            
            for i in range(0,n_cop,1):
                
                ccdf_data = tfp.math.batch_interp_regular_nd_grid(self.data_s[:,:,i],self.grid_s.min,self.grid_s.max,cdf1[:,:,i],axis=-2)
                mar_p1, mar_s1 = kernel_cdf(ccdf_data, self.grid_u.ex)
                
                pd_points, ccdf_points = evaluate_points(self.points_s[:,:,i], batch_size, self.grid_s, cdf1[:,:,i], pd_grid_uv[:,:,i])    
                
                # Update logf
                logftr = tf.math.log(pd_points) 
                logf_tmp = self.logf[:,tr+1] + tf.squeeze(logftr)
                self.logf = update_tensor2D(self.logf,tr+1,logf_tmp)

                # Update Fp
                interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
                self.Fp[:,tr+1,i] = interp_cdf_poi
            
        logp = self.logf[:,0]
        logp_copula = tf.zeros(tf.shape(self.logf[:,0]),points.dtype)
        for i in tf.range(1,d,1,tf.int32):
            #print('loghi',logf[:,i])
            logp = logp + self.logf[:,i]
            logp_copula = logp_copula + self.logf[:,i]
        #print('logp',logp)
        p = tf.exp(logp)
        p_copula = tf.exp(logp_copula)
        return p, p_copula
    
    
# class vine_obj_bin(object):
#     """Vine object.
#     """
#     def __init__(self, vine_family, families, vine_depth, margin, knots, *args):
#         """Create a marginal object.
#         Args:
#             families: Copula family.
#             theta: Correlation factor.
#             margin: Margin of the copula.
#         """
#         self.vine_family = vine_family
#         self.families = families
#         self.theta1 = []
#         self.theta2 = []
#         self.rang = None
#         self.n_cop = vine_depth
#         self.margin = margin
#         self.knots = knots
        
#         if self.vine_family == 'r-vine':
#             self.method = args[0]
#             if self.method == 'matrix':
#                 self.r_matrix = args[1]
#                 self.E = build_edges(self.r_matrix)
        
#         if self.vine_family == 'c-vine':
#             self.r_matrix = np.tril(np.tile(np.array(range(self.n_cop,0,-1)),(self.n_cop,1)).T)
#             self.E = build_edges(self.r_matrix)
        
#         if self.vine_family == 'd-vine':
#             self.r_matrix = np.zeros((self.n_cop,self.n_cop),np.int32)
#             for i in range(0,self.n_cop,1):
#                 self.r_matrix[i,i] = self.n_cop-i #-1
#             for j in range(0,self.n_cop-1,1):
#                 c = 1 #0
#                 for i in range(j+1,self.n_cop,1):
#                     self.r_matrix[i,j] = c
#                     c += 1
#             self.E = build_edges(self.r_matrix)
                    
#         self.n_cop = None
#         self.Mar_G = None
#         self.theta = None
#         self.Fp = None
#         self.logf = None
#         self.copulas = None
        
#         self.data_u = None
#         self.data_s = None
#         self.data_x = None
        
#         self.points_u = None
#         self.points_s = None
#         self.points_x = None
        
#         self.grid_u = None
#         self.grid_s = None
#         self.grid_x = None
        
#         self.binning = False
#         self.n_bin = 1
# #         self.err_trace = tf.Variable(initial_value=tf.ones([1],x.dtype),dtype=x.dtype, trainable=False)
# #         self.pos_trace = tf.Variable(initial_value=tf.ones([1],x.dtype),dtype=x.dtype, trainable=False)
# #         self.iter_err = tf.Variable(1,dtype=tf.int32, trainable=False)
# #         self.a = tf.Variable(initial_value=tf.random.uniform(shape=[1], minval=2e-3, maxval=2, dtype=x.dtype), trainable=False)
        
#     def fit(self, x, parallel, opt_method, binning, n_bin):
        
#         ## Initialization
#         d = x.shape[1]
#         self.n_cop = d
#         self.binning = binning
#         self.n_bin = n_bin
        
#         ## Make grid
#         u_1, ex_u = mk_grid(self.knots,x.dtype)
#         trans = Transform(self.n_cop)
        
#         ## Grid objects
#         self.grid_u = grid_obj(ex_u)
#         self.grid_s = grid_obj(trans.forward_u(ex_u))
        
#         ## Bivariate normal
#         x1_s, x2_s = self.grid_s.axis()
#         NORM = biv_norm(x1_s, x2_s)
        
#         ## Create Mar_G, theta and Fp
#         Mar_G = []
#         theta = np.zeros([x.shape[0],self.n_cop,self.n_cop],x.dtype)
#         for i in range(0,self.n_cop,1):
#             ccc = self.margin[i].ker
#             mar_p1, mar_s1 = kernel_cdf(ccc, ex_u)
#             Mar_G.append([mar_s1, mar_p1])
#             theta[:,0,i] = interp1d_np(ccc, mar_s1, mar_p1).numpy()
#             del ccc, mar_p1, mar_s1
            
#         self.Mar_G = Mar_G
#         self.theta = theta
        
#         ############### FITTING ####################
#         self.copulas = []
        
#         for tr in tf.range(0,d-1,1,tf.int32): #d-1
#             print('-----------------------------------')
#             print('Row theta:',tr.numpy())
            
#             print('theta:',self.theta[:,tr,:])

#             # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
#             n_cop = d-1-tr
#             trans = Transform(n_cop)
#             print('n_cop in the row:',n_cop.numpy())
            
#             data_u = np.empty([theta.shape[0],2,n_cop],x.dtype)  
            
#             #### EDGES OF THE VINE
#             if self.vine_family == 'r-vine':
#                 if self.method == 'matrix':   
#                     ind_ee = edges_index(self.E,self.r_matrix,tr)
#                 elif self.method == 'optimal':   
#                     if tr == 0:
#                         self.r_matrix = np.zeros([self.n_cop,self.n_cop],np.int32)
#                         n = len(self.r_matrix) - 1
#                         ind_ee, weights = optimal_tree(self.theta[:,tr,:])
#                         edges = []
#                         cc = 0
#                         for edg in ind_ee:
#                             self.r_matrix[cc,cc] = edg[1] +1
#                             self.r_matrix[n,cc] = edg[0] +1
#                             edges.append({edg[0],edg[1]})
#                             cc += 1
#                         print(ind_ee)
#                     else:
#                         ind_ee, weights = optimal_tree(self.theta[:,tr,:-tr])
#                         n = len(self.r_matrix) - 1
#                         cc = 0
#                         for edg in ind_ee:
#                             parent = edges[edg[0]].intersection(edges[edg[1]])
#                             for par in parent:
#                                 self.r_matrix[n-tr,cc] = par +1
#                             cc += 1
#                         edges = []
#                         cc = 0
#                         for edg in ind_ee:
#                             edges.append({edg[0],edg[1]})
#                             cc += 1
#                         print('opt_tree',ind_ee)
#             elif (self.vine_family == 'c-vine') | (self.vine_family == 'd-vine'):
#                 ind_ee = edges_index(self.E,self.r_matrix,tr)
            
#             ro = 0
#             for ind in ind_ee:
#                 data_u[:,0,ro] = self.theta[:,tr,ind[0]]
#                 data_u[:,1,ro] = self.theta[:,tr,ind[1]]
#                 ro += 1
            
#             ## Transform data
#             self.data_u = data_u
#             self.data_s = trans.forward_u(self.data_u)
#             self.data_x = trans.forward_s(self.data_s)
            
#             ## Grid on P-Q space
#             self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))
            
#             #print(data_u)
            
#             #self.copulas.append([])
#             opt_bw = tf.TensorArray(x.dtype,size=n_cop)
            
#             if parallel == True:
#                 n_cop1 = tf.constant(n_cop,tf.int32)

#                 grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex}
#                 data_dict = {'data_s':self.data_s, 'data_x':self.data_x}
#                 par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,100], 'lr':[0.1, 0.01], 
#                                 'conv_tol': [0.000001,0.0000001], 'opt_method': opt_method}

#                 opt = optimization(grid_dict, data_dict, par_dict)
#                 opt_bw = opt            
#             elif parallel == False:
#                 if binning == False:
#                     opt_bw = tf.TensorArray(x.dtype,size=n_cop)
                
#                     for i in range(0,n_cop,1):
#                         print('col:',i)

#                         n_cop1 = tf.constant(1,tf.int32)

#                         grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
#                         data_dict = {'data_s':self.data_s[:,:,i], 'data_x':self.data_x[:,:,i]}
#                         par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,100], 'lr':[0.1, 0.01], 
#                                     'conv_tol': [0.000001,0.0000001], 'opt_method': opt_method}

#                         opt = optimization(grid_dict, data_dict, par_dict)
#                         opt_bw = opt_bw.write(i,opt)

#                     opt_bw = opt_bw.stack()  
#                 elif binning == True:
                    
#                     opt_bw = tf.TensorArray(x.dtype,size=n_cop)
# #                     n_bin = 5

#                     if tr == 0:
#                         for i in range(0,n_cop,1):
#                             print('col:',i)

#                             n_cop1 = tf.constant(1,tf.int32)

#                             grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
#                             data_dict = {'data_s':self.data_s[:,:,i], 'data_x':self.data_x[:,:,i]}
#                             par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,100], 'lr':[0.1, 0.01], 
#                                         'conv_tol': [0.000001,0.0000001], 'opt_method': opt_method}

#                             opt = optimization(grid_dict, data_dict, par_dict)
#                             opt_bw = opt_bw.write(i,opt)

#                         opt_bw = opt_bw.stack()  
#                     else:

#                         n = len(self.r_matrix)-1

#                         for i in range(0,n_cop,1):
#                             print('col:',i)

#                             opt_bin = tf.TensorArray(x.dtype,size=n_bin)
#                             parent = self.r_matrix[n-tr,n-2-i] - 1 # Because first node starts from 1 (and not from 0)
#                             print('parent', parent)
# #                             pp = tf.math.floormod(tf.shape(self.theta)[0],n_bin*5)                            
# #                             qc = pd.qcut(self.theta[:,0,parent], q=n_bin, precision=1)
# #                             qc = pd.qcut(self.theta[:tf.shape(self.theta)[0]-pp,0,parent], q=n_bin, precision=1)
#                             bins = create_bins(self.theta[:,0,parent],n_bin)
#                             val_to_bin = np.digitize(self.theta[:,0,parent], bins) -1

#                             for bb in range(0,n_bin,1):
#                                 print('bin:',bb)
# #                                 mask = qc.codes == bb
# #                                 mask = tf.where(tf.equal(qc.codes,bb))
#                                 mask = tf.where(tf.equal(val_to_bin,bb))

#                                 n_cop1 = tf.constant(1,tf.int32)
                                
#                                 data_s_bin = tf.gather_nd(self.data_s[:,:,i],mask)
#                                 data_x_bin = tf.gather_nd(self.data_x[:,:,i],mask)

#                                 grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
#                                 data_dict = {'data_s':data_s_bin, 'data_x':data_x_bin}
#                                 par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,100], 'lr':[0.1, 0.01], 
#                                             'conv_tol': [0.0001,0.0001], 'opt_method': opt_method}

#                                 opt = optimization(grid_dict, data_dict, par_dict)
# #                                 print(opt)
#                                 opt_bin = opt_bin.write(bb,opt)
#                             opt_bin = opt_bin.stack()
#                             opt_bin = tf.reshape(opt_bin,[tf.shape(opt)[0],n_bin])
                            
#                             print(opt_bin)
                
#                             opt_bw = opt_bw.write(i,opt_bin)

#                         opt_bw = opt_bw.stack()
#                         opt_bw = tf.reshape(opt_bw,[tf.shape(opt)[0],n_cop,n_bin])
# #                         opt_bin = tf.reshape(opt_bin,[tf.shape(opt)[0],n_cop,n_bin])
            
#             copula = copula_obj(opt_bw)
#             self.copulas.append(copula)
            
#             ## UPDATE THETA
#             if binning == False:
                
#                 grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x}
#                 data_dict = {'data_s':self.data_s, 'data_x':self.data_x, 'theta':self.theta}
#                 par_dict = {'copulas': self.copulas[tr], 'n_cop':tf.convert_to_tensor(n_cop), 'batch':tf.constant(2,tf.int32), 'tr':tr}

#                 self.copulas[tr].pd_grid_uv, self.copulas[tr].cdf, self.theta = evaluate_fit(data_dict, grid_dict, par_dict)
            
#             elif binning == True:
                
#                 if tr == 0:
#                     grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x}
#                     data_dict = {'data_s':self.data_s, 'data_x':self.data_x, 'theta':self.theta}
#                     par_dict = {'copulas': self.copulas[tr], 'n_cop':tf.convert_to_tensor(n_cop), 'batch':tf.constant(2,tf.int32), 'tr':tr}

#                     self.copulas[tr].pd_grid_uv, self.copulas[tr].cdf, self.theta = evaluate_fit(data_dict, grid_dict, par_dict)
                
#                 else:
                    
# #                     exc = tf.math.floormod(tf.shape(self.theta)[0],n_bin*5)
# #                     len_bin = (tf.shape(x)[0]-pp)/n_bin
                    
#                     len_bin = tf.shape(theta)[0]/n_bin
#                     data_s_bin = np.empty([len_bin,tf.shape(self.data_s)[1],n_cop,n_bin],x.dtype)
#                     data_x_bin = np.empty([len_bin,tf.shape(self.data_s)[1],n_cop,n_bin],x.dtype)
# #                     print('init',data_s_bin.shape())
#                     for i in range(0,n_cop,1):

#                         parent = self.r_matrix[n-tr,n-2-i] - 1 # Because first node starts from 1 (and not from 0)
# #                         qc = pd.qcut(self.theta[:,0,parent], q=n_bin, precision=1)
# #                         qc = pd.qcut(self.theta[:tf.shape(self.theta)[0],0,parent], q=n_bin, precision=1)  # Remove points to be divisible
#                         bins = create_bins(self.theta[:,0,parent],n_bin)
#                         val_to_bin = np.digitize(self.theta[:,0,parent], bins) -1
#                         for bb in range(0,n_bin,1):
# #                             mask = tf.where(tf.equal(qc.codes,bb))
#                             mask = tf.where(tf.equal(val_to_bin,bb))
        
#                             data_s_bin[:,:,i,bb] = tf.gather_nd(self.data_s[:,:,i],mask)
#                             data_x_bin[:,:,i,bb] = tf.gather_nd(self.data_x[:,:,i],mask)
                
#                     self.copulas[tr].pd_grid_uv = np.zeros([self.knots,self.knots,n_cop,n_bin],x.dtype)
#                     self.copulas[tr].cdf = np.zeros([self.knots,self.knots,n_cop,n_bin],x.dtype)
#                     for bb in range(0,n_bin,1):
#                         ## UPDATE THETA

#                         grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x}
#                         data_dict = {'data_s':data_s_bin[:,:,:,bb], 'data_x':data_x_bin[:,:,:,bb],'bin':bb}
#                         par_dict = {'copulas': self.copulas[tr], 'n_cop':tf.convert_to_tensor(n_cop), 'batch':tf.constant(2,tf.int32), 'tr':tr}

#                         self.copulas[tr].pd_grid_uv[:,:,:,bb], self.copulas[tr].cdf[:,:,:,bb] = evaluate_fit_bin(data_dict, grid_dict, par_dict)
                    
#                     interp_cdf_bin = np.zeros([tf.shape(theta)[0],n_cop],x.dtype)
#                     for i in range(0,n_cop,1):
#                         print('col:',i)
#                         parent = self.r_matrix[n-tr,n-2-i] - 1 # Because first node starts from 1 (and not from 0)
# #                         qc = pd.qcut(theta[:,0,parent], q=n_bin, precision=1)
# #                         qc = pd.qcut(theta[:tf.shape(theta)[0],0,parent], q=n_bin, precision=1)
#                         bins = create_bins(self.theta[:,0,parent],n_bin)
#                         val_to_bin = np.digitize(self.theta[:,0,parent], bins) -1
#                         for bb in range(0,n_bin,1):
#                             print('bin:',bb)
# #                             mask = tf.where(tf.equal(qc.codes,bb))
#                             mask = tf.where(tf.equal(val_to_bin,bb))
        
#                             ## Update theta  
#                             ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin[:,:,i,bb],self.grid_s.min,self.grid_s.max,self.copulas[tr].cdf[:,:,i,bb],axis=-2)
#                             ccdf_data = tf.squeeze(ccdf_data)
#                             mar_p1, mar_s1 = kernel_cdf(ccdf_data, self.grid_u.ex)
# #                             print('mar_p1',tf.shape(mar_p1))
# #                             print('mar_s1',tf.shape(mar_s1))
#                             interp_cdf = interp1d_np(ccdf_data, mar_s1, mar_p1)
# #                             print('interp_cdf',interp_cdf)
# #                             print('interp_cdf_bin',tf.shape(interp_cdf_bin[mask,i]))
#                         #     theta = update_tensor(theta,interp_cdf,tr+1,i)
#                             interp_cdf_bin[mask,i] = tf.reshape(interp_cdf,[tf.shape(interp_cdf)[0],1])
                    
#                     for i in range(0,n_cop,1):
#     #                     interp_cdf_bin1 = tf.squeeze(interp_cdf_bin[:,i])
#                         self.theta[:,tr+1,i] = interp_cdf_bin[:,i]
                
#         return

#     ################################ EVALUATION #############################################
#     def evaluation(self, points):
        
#         d = tf.shape(points)[1]
        
#         ## Create Fp
#         Fp = np.empty([tf.shape(points)[0],self.n_cop,self.n_cop],self.data_u.dtype)
#         for i in range(0,self.n_cop,1):
#             Fp[:,0,i] = interp1d_np(points[:,i], self.Mar_G[i][0], self.Mar_G[i][1]).numpy()
#         self.Fp = Fp
        
#         ### Create logf
#         en = tf.TensorArray(points.dtype,size=d)
#         logf = tf.zeros(tf.shape(points),points.dtype)
#         for i in tf.range(0,d,1,tf.int32):
#             den1,mden1 = kernel_pdf2(self.margin[i].ker)      
#             inter = interp_pdf(points[:,i], mden1, den1) #interp1d_np
#             # Product of pdf is the sum of logarithm - Product of pdf margingales evaluated on copula samples
#             #logf = logf + tf.math.log(inter)
#             logf_tmp = logf[:,0] + tf.math.log(inter)
#             logf = update_tensor2D(logf,0,logf_tmp)

#             m_diff = mden1[1:] - mden1[:-1]
#             m_diff = tf.concat([m_diff, tf.expand_dims(m_diff[-1], 0)], 0)

#             # log 2 of the pdf on the grid
#             log_pd = tf.py_function(np.log2, [den1], den1.dtype)
#             log_pd = replace_inf(log_pd, tf.constant(den1.dtype.min,den1.dtype))
#             en = en.write(i,- tf.math.reduce_sum(den1*log_pd*tf.transpose(m_diff),0))  #log_pd
#             #vec = tf.linspace(tf.math.reduce_min(vine1.margin[i].ker),tf.math.reduce_max(vine1.margin[i].ker),100)
#             #inter_en = interp1d_np(vec, mden1, den1)
#             #en = en.write(i,tf.py_function(stats.entropy, [inter_en,vec], vec.dtype))
#             del den1,mden1,logf_tmp,m_diff,log_pd
#         en = en.stack()
#         self.logf = logf.numpy()
        
#         for tr in tf.range(0,d-1,1,tf.int32): #d-1
#             print('Row theta:',tr.numpy())
            
#             # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
#             n_cop = d-1-tr
#             trans = Transform(n_cop)
#             print('n_cop in the row:',n_cop.numpy())
            
#             data_u = np.empty([self.theta.shape[0],2,n_cop],self.data_u.dtype)
#             points_u = np.empty([self.Fp.shape[0],2,n_cop],self.data_u.dtype)
            
#             #### EDGES OF THE VINE
#             if self.vine_family == 'r-vine':
#                 if self.method == 'matrix':   
#                     ind_ee = edges_index(self.E,self.r_matrix,tr)
#                 elif self.method == 'optimal':   
#                     if tr == 0:
#                         ind_ee, weights = optimal_tree(self.theta[:,tr,:])
#                     else:
#                         ind_ee, weights = optimal_tree(self.theta[:,tr,:-tr])
#             elif (self.vine_family == 'c-vine') | (self.vine_family == 'd-vine'):
#                 ind_ee = edges_index(self.E,self.r_matrix,tr)
            
#             ro = 0
#             for ind in ind_ee:
#                 data_u[:,0,ro] = self.theta[:,tr,ind[0]]
#                 data_u[:,1,ro] = self.theta[:,tr,ind[1]]
#                 points_u[:,0,ro] = self.Fp[:,tr,ind[0]]
#                 points_u[:,1,ro] = self.Fp[:,tr,ind[1]]
#                 ro += 1
            
#             ## Transform data
#             self.data_u = data_u
#             self.data_s = trans.forward_u(self.data_u)
#             self.data_x = trans.forward_s(self.data_s)
            
#             ## Transform points
#             self.points_u = points_u
#             self.points_s = trans.forward_u(self.points_u)
#             self.points_x = trans.forward_s(self.points_s)
            
#             ## Grid on P-Q space
#             self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))
            
#             #print(data_u)
#             grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x}
#             data_dict = {'data_s':self.data_s, 'data_x':self.data_x, 'theta':self.theta}
#             par_dict = {'copulas': self.copulas[tr], 'n_cop':tf.convert_to_tensor(n_cop), 'batch':tf.constant(2,tf.int32), 'tr':tr}
            
# #             pd_grid_uv, cdf1, self.theta = evaluate_fit(data_dict, grid_dict, par_dict)
    
#             pd_grid_uv = self.copulas[tr].pd_grid_uv
#             cdf1 = self.copulas[tr].cdf
    
#             batch_size = tf.constant(2,tf.int32)
            
#             if self.binning == False:
            
#                 for i in range(0,n_cop,1):

#                     ccdf_data = tfp.math.batch_interp_regular_nd_grid(self.data_s[:,:,i],self.grid_s.min,self.grid_s.max,cdf1[:,:,i],axis=-2)
#                     mar_p1, mar_s1 = kernel_cdf(ccdf_data, self.grid_u.ex)

#                     pd_points, ccdf_points = evaluate_points(self.points_s[:,:,i], batch_size, self.grid_s, cdf1[:,:,i], pd_grid_uv[:,:,i])    

#                     # Update logf
#                     logftr = tf.math.log(pd_points) 
#                     logf_tmp = self.logf[:,tr+1] + tf.squeeze(logftr)
#                     self.logf = update_tensor2D(self.logf,tr+1,logf_tmp)

#                     # Update Fp
#                     interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
#                     self.Fp[:,tr+1,i] = interp_cdf_poi
            
#             elif self.binning == True:
#                 n_bin = self.n_bin
#                 n = len(self.r_matrix)-1
#                 if tr == 0:
#                     for i in range(0,n_cop,1):

#                         ccdf_data = tfp.math.batch_interp_regular_nd_grid(self.data_s[:,:,i],self.grid_s.min,self.grid_s.max,cdf1[:,:,i],axis=-2)
#                         mar_p1, mar_s1 = kernel_cdf(ccdf_data, self.grid_u.ex)

#                         pd_points, ccdf_points = evaluate_points(self.points_s[:,:,i], batch_size, self.grid_s, cdf1[:,:,i], pd_grid_uv[:,:,i])    

#                         # Update logf
#                         logftr = tf.math.log(pd_points) 
#                         logf_tmp = self.logf[:,tr+1] + tf.squeeze(logftr)
#                         self.logf = update_tensor2D(self.logf,tr+1,logf_tmp)

#                         # Update Fp
#                         interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
#                         self.Fp[:,tr+1,i] = interp_cdf_poi
                    
#                 else:
                    
# #                     n_bin = 5
#                     len_bin = tf.shape(self.theta)[0]/n_bin
#                     data_s_bin = np.empty([len_bin,tf.shape(self.data_s)[1],n_cop,n_bin],self.data_u.dtype)
#                     len_bin1 = tf.shape(self.Fp)[0]/n_bin                 
# #                     points_s_bin = np.empty([len_bin1,tf.shape(self.points_s)[1],n_cop,n_bin],x.dtype)
#                     points_s_bin = []
                    
#                     for i in range(0,n_cop,1):

#                         parent = self.r_matrix[n-tr,n-2-i] - 1 # Because first node starts from 1 (and not from 0)
#                         print(parent)
# #                         qc = pd.qcut(self.theta[:,0,parent], q=n_bin, precision=1)
#                         bins = create_bins(self.theta[:,0,parent],n_bin)

#                         val_to_bin = np.digitize(self.theta[:,0,parent], bins) -1
#                         val_to_bin1 = np.digitize(self.Fp[:,0,parent], bins) -1

                        
# #                         ### POINTS BINNING PROBLEM
# #                         ind_sort = np.argsort(self.Fp[:,0,parent])

# #                         ind_bin = np.arange(0,n_bin,1)
# #                         ind_bin = np.tile(ind_bin,(int(len_bin1),1)).T
# #                         ind_bin = np.reshape(ind_bin,(len_bin1*n_bin))
                        
# #                         ind_fin = np.zeros([tf.shape(self.Fp[:,0,parent])[0]],np.int32)
# #                         ind_fin[ind_sort] = ind_bin
                        
# #                         qc1 = pd.qcut(self.Fp[:,0,parent], q=n_bin, precision=1)
#                         points_s_bin1 = []
#                         for bb in range(0,n_bin,1):
# #                             mask = tf.where(tf.equal(qc.codes,bb))
# #                             mask1 = tf.where(tf.equal(qc1.codes,bb))
# #                             mask1 = tf.where(tf.equal(ind_fin,bb))
                            
#                             mask = tf.where(tf.equal(val_to_bin,bb))
#                             mask1 = tf.where(tf.equal(val_to_bin1,bb))

#                             data_s_bin[:,:,i,bb] = tf.gather_nd(self.data_s[:,:,i],mask)
# #                             points_s_bin[:,:,i,bb] = tf.gather_nd(self.points_s[:,:,i],mask1)
#                             points_s_bin1.append(tf.gather_nd(self.points_s[:,:,i],mask1))

#                         points_s_bin.append(points_s_bin1)

#                     log_f_bin = np.empty([tf.shape(self.logf)[0],n_cop],self.data_u.dtype)
#                     Fp_bin = np.empty([tf.shape(self.Fp)[0],n_cop],self.data_u.dtype)
#                     for i in range(0,n_cop,1):
#                         print('col:',i)
#                         parent = self.r_matrix[n-tr,n-2-i] - 1 # Because first node starts from 1 (and not from 0)
                        
# #                         ### POINTS BINNING PROBLEM
# #                         ind_sort = np.argsort(self.Fp[:,0,parent])

# #                         ind_bin = np.arange(0,n_bin,1)
# #                         ind_bin = np.tile(ind_bin,(int(len_bin1),1)).T
# #                         ind_bin = np.reshape(ind_bin,(len_bin1*n_bin))
                        
# #                         ind_fin = np.zeros([tf.shape(self.Fp[:,0,parent])[0]],np.int32)
# #                         ind_fin[ind_sort] = ind_bin
#                         bins = create_bins(self.theta[:,0,parent],n_bin)
#                         val_to_bin1 = np.digitize(self.Fp[:,0,parent], bins) -1
                        
# #                         qc1 = pd.qcut(self.Fp[:,0,parent], q=n_bin, precision=1)
#                         for bb in range(0,n_bin,1):
#                             print('bin:',bb)
# #                             mask1 = tf.where(tf.equal(qc1.codes,bb))
# #                             mask1 = tf.where(tf.equal(ind_fin,bb))
#                             mask1 = tf.where(tf.equal(val_to_bin1,bb))

#                             ## Update theta  
#                             ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin[:,:,i,bb],self.grid_s.min,self.grid_s.max,self.copulas[tr].cdf[:,:,i,bb],axis=-2)
#                             ccdf_data = tf.squeeze(ccdf_data)
# #                             print('ccdf_data',ccdf_data)
#                             mar_p1, mar_s1 = kernel_cdf(ccdf_data, self.grid_u.ex)
#     #                             print('mar_p1',tf.shape(mar_p1))
#     #                             print('mar_s1',tf.shape(mar_s1))
                            
# #                             pd_points, ccdf_points = evaluate_points(points_s_bin[:,:,i,bb], batch_size, self.grid_s, cdf1[:,:,i,bb], pd_grid_uv[:,:,i,bb]) 
#                             pd_points, ccdf_points = evaluate_points(points_s_bin[i][bb], batch_size, self.grid_s, cdf1[:,:,i,bb], pd_grid_uv[:,:,i,bb]) 
                            
# #                             print('pd_points',tf.shape(pd_points))
#                             # Update logf
#                             logftr = tf.math.log(pd_points) 
#                             logf_tmp = tf.gather_nd(self.logf[:,tr+1],mask1)
# #                             print('logf_tmp',logf_tmp)
#                             logf_tmp = logf_tmp + tf.squeeze(logftr)

#                             log_f_bin[mask1,i] = tf.reshape(logf_tmp,[tf.shape(logf_tmp)[0],1])
            
#                             # Update Fp
#                             interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
#                             Fp_bin[mask1,i] = tf.reshape(interp_cdf_poi,[tf.shape(interp_cdf_poi)[0],1])
                        
#                         log_f_bin1 = tf.squeeze(log_f_bin[:,i])
# #                         print('add log f 2d',log_f_bin)
#                         Fp_bin1 = tf.squeeze(Fp_bin[:,i])
#                         self.logf = update_tensor2D(self.logf,tr+1,log_f_bin1)
#                         self.Fp[:,tr+1,i] = Fp_bin1
            
#         logp = self.logf[:,0]
#         logp_copula = tf.zeros(tf.shape(self.logf[:,0]),points.dtype)
#         for i in tf.range(1,d,1,tf.int32):
#             #print('loghi',logf[:,i])
#             logp = logp + self.logf[:,i]
#             logp_copula = logp_copula + self.logf[:,i]
#         #print('logp',logp)
#         p = tf.exp(logp)
#         p_copula = tf.exp(logp_copula)
#         return p, p_copula

# File: src/DVC_tensorflow/classes/objects.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from utils.tensor_op import *
from utils.prob_op import *
from utils.interpolation import *
from pre_proc.transformation import *
from grid.grid_op import *
from grid.grid_class import *
from evalu.cop_eval import *
from evalu.vine_eval import *
from vine_tree.tree_op import *
from optim.vine_fit import *
from param.cond_copula import *

## VINE OBJ BIN
from utils.dataset_op import create_bins

class copula_obj(object):
    """Copula object.
    """
    def __init__(self, opt_bw):
        """Create a copula object.
        Args:
            opt_bw: Optimal fitted bandwidth.
        """
        self.opt_bw = opt_bw
        self.pd_grid_uv = None
        self.cdf = None

class cop_par_obj(object):
    """Marginal object.
    """
    def __init__(self, family, theta):
        """Create a marginal object.
        Args:
            ker: Kernel of the marginal.
            family: Type of the marginal.
            min: Min of the range of the marginal.
            max: Max of the range of the marginal.
        """
        self.family = family
        self.theta = theta

class margin_obj(object):
    """Marginal object.
    """
    def __init__(self, dist, theta, is_cont):
        """Create a marginal object.
        Args:
            ker: Kernel of the marginal.
            family: Type of the marginal.
            min: Min of the range of the marginal.
            max: Max of the range of the marginal.
        """
        self.dist = dist
        self.theta = theta
        self.is_cont = is_cont
        self.ker = None

class vine_obj_bin(object):
    """Vine object.
    """
    def __init__(self, vine_family, families, vine_depth, margin, knots, method, *args):
        """Create a marginal object.
        Args:
            families: Copula family.
            theta: Correlation factor.
            margin: Margin of the copula.
        """
        self.method = method
        self.vine_family = vine_family
        self.families = families
        self.theta1 = []
        self.theta2 = []
        self.rang = None
        self.n_cop = vine_depth
        self.margin = margin
        self.knots = knots
        
        self.ind_vine = []
        for i in range(0,self.n_cop-1,1):
            self.ind_vine.append([])
        
        
        if self.vine_family == 'r-vine':
            #self.method_rmat = args[0]
            if self.method == 'matrix':
                self.r_matrix = args[0] #-Houman used to be args[1]
                self.E, self.ind_vine, self.nodes, self.matrix_edges = prepare_regular(self.r_matrix)
            elif self.method == 'random':
                self.r_matrix, _, _, _ = random_r_matrix_gen(self.n_cop)
                self.E, self.ind_vine, self.nodes, self.matrix_edges = prepare_regular(self.r_matrix)
                #print(self.r_matrix)
        
        if (self.vine_family == 'c-vine') | (self.vine_family == 'd-vine'):
            self.r_matrix, self.ind_vine, self.nodes, self.matrix_edges = prepare_vine(self.vine_family, self.n_cop)
                    
        self.Mar_G = None
        self.theta = None
        self.Fp = None
        self.logf = None
        self.copulas = None
        
        self.data_u = None
        self.data_s = None
        self.data_x = None
        
        self.points_u = None
        self.points_s = None
        self.points_x = None
        
        self.grid_u = None
        self.grid_s = None
        self.grid_x = None
        
        self.binning = False
        self.n_bin = 1
    
    def select_batch_size_cdf(self, x):
        batch_size_cdf = tf.constant(1,tf.int32)
        if np.shape(x)[0] > 5000:
            batch_size_cdf = tf.constant(10,tf.int32)
        elif np.shape(x)[0] > 10000:
            batch_size_cdf = tf.constant(100,tf.int32)
        elif np.shape(x)[0] > 50000:
            batch_size_cdf = tf.constant(200,tf.int32)
        elif np.shape(x)[0] > 100000:
            batch_size_cdf = tf.constant(500,tf.int32)
        elif np.shape(x)[0] > 200000:
            batch_size_cdf = tf.constant(1000,tf.int32)
        elif np.shape(x)[0] > 500000:
            batch_size_cdf = tf.constant(2000,tf.int32)
        return batch_size_cdf

    def select_batch_size(self, data):
        batch_size = tf.constant(1,tf.int32)
        if np.shape(data)[0] > 2000:
            batch_size = tf.constant(5,tf.int32)
        elif np.shape(data)[0] > 10000:
            batch_size = tf.constant(10,tf.int32)
        elif np.shape(data)[0] > 50000:
            batch_size = tf.constant(20,tf.int32)
        elif np.shape(data)[0] > 100000:
            batch_size = tf.constant(50,tf.int32)
        elif np.shape(data)[0] > 200000:
            batch_size = tf.constant(100,tf.int32)
        elif np.shape(data)[0] > 500000:
            batch_size = tf.constant(200,tf.int32)
        return batch_size

    def fit(self, x, gen_dict, npc_dict, par_dict, bin_dict): #*args
        
        
        np_type = x.dtype
        x = tf.convert_to_tensor(x)

        ## Initialization
        self.binning = gen_dict['binning']
        self.parallel = gen_dict['parallel']
        self.param = gen_dict['param']
        self.fitted = gen_dict['fitted']
        
        self.vine_depth = gen_dict['vine_depth'] -1

        d = x.shape[1]
        self.n_cop = d
        
        if self.param == False:
            self.opt_method = npc_dict['opt_method']
            batch_paral = npc_dict['batch_paral']
        else:
            param_families = par_dict['param_families']
        if self.binning == True:
            self.n_bin = bin_dict['n_bin']
        
        ## Select batch size for CDF
        batch_size_cdf = self.select_batch_size_cdf(x)
        
        ## Make grid

        u_1, ex_u = mk_grid(tf.convert_to_tensor(self.knots),np_type)
        trans = Transform(self.n_cop)
        
        ## Grid objects
        self.grid_u = grid_obj(ex_u)
        self.grid_s = grid_obj(trans.forward_u(ex_u))
        
        ## Bivariate normal
        x1_s, x2_s = self.grid_s.axis()
        NORM = biv_norm(x1_s, x2_s)
        self.grid_u.axis()
        self.grid_s.min_grid()
        self.grid_s.max_grid()
        
        ## Create Mar_G, theta and Fp
        self.Mar_G = []
        self.theta_flip = np.zeros([tf.shape(x)[0],self.n_cop,self.n_cop],np_type)
        self.theta = np.zeros([tf.shape(x)[0],self.n_cop,self.n_cop],np_type)
        for i in range(0,self.n_cop,1):
            ccc = tf.convert_to_tensor(self.margin[i].ker)
            interp_cdf, mar_s1, mar_p1 = kernel_cdf(ccc,ccc,ex_u)

            self.Mar_G.append([mar_s1, mar_p1])
            self.theta[:,0,i] = interp_cdf.numpy() #interp1d_np(ccc, mar_s1, mar_p1).numpy()
            del ccc, mar_p1, mar_s1
        
        ######################################### FITTING #######################################################
        
        if self.fitted == False:
            self.copulas = []
        self.correlations = []
        self.correlations_bins = []
        self.flip_flag = []
        self.ind_edge_rel = []
        
        for tr in tf.range(0,d-1,1,tf.int32): #d-1
          #  print('-----------------------------------')
          #  print('Row theta:',tr.numpy())
            
            if self.fitted == True:
                self.vine_family = 'r-vine'
                self.method = 'matrix'
            
           # print('theta:',self.theta[:,tr,:])

            ## Number of copulas in the level
            ## Create object for projections in the other spaces
            
            n_cop = d-1-tr
            trans = Transform(n_cop)
     #       print('n_cop in the row:',n_cop.numpy())
            
            ###### COMPUTE THE EDGES OF THE VINE LEVEL
            
            if self.vine_family == 'r-vine':
                
                if (self.method == 'matrix') | (self.method == 'random'):   
                    edges_now = self.ind_vine[tr]
                    
                elif (self.method == 'optimal'): # | (self.method == 'random'):   
                    
                    random = False
                    
                    # if (self.method == 'random'):
                    #     random = True
                        
                    if tr == 0:
                        self.r_matrix = np.zeros([self.n_cop,self.n_cop],np.int32)
                        n = len(self.r_matrix) - 1
                        ind_ee, weights = optimal_tree(self.theta[:,tr,:],self.theta_flip[:,tr,:],self.ind_vine,tr,random)
                        edges_now = ind_ee
                        self.ind_vine[tr] = ind_ee
                #        print('opt_tree',ind_ee)

                        edges = []
                        for j in range(0,len(ind_ee),1):
                            edg = ind_ee[len(ind_ee)-1-j]
                            self.r_matrix[n,j] = edg[0] +1
                            self.r_matrix[j,j] = edg[1] +1
                            edges.append({edg[0],edg[1]})

                        edges = np.flip(edges)

                        self.nodes = np.zeros(self.n_cop,np.int32)
                        V = set(range(1,self.n_cop+1))
                        for i in range(0,self.n_cop,1):
                            self.nodes[i]=self.r_matrix[i,i]
                            u_nod = {self.nodes[i]}
                            if u_nod.issubset(V):
                                V.remove(self.nodes[i])
                        self.nodes = np.flip(self.nodes)

                        for elem in V:
                            ind = np.where(self.nodes == 0)
                            self.nodes[self.nodes == 0] = elem
                            self.r_matrix[n-ind[0],n-ind[0]] = elem
                        #print(self.r_matrix)
                    else:

                        ind_ee, weights = optimal_tree(self.theta[:,tr,:],self.theta_flip[:,tr,:],self.ind_vine,tr,random)
              #          print('opt_tree',ind_ee)
                        edges_now = ind_ee
                        self.ind_vine[tr] = ind_ee
                    #print(self.ind_vine[tr])

            elif (self.vine_family == 'c-vine') | (self.vine_family == 'd-vine'):
                edges_now = self.ind_vine[tr]
            
            
            ######### FROM THETA MATRIX TAKE THE DATA CDF FOR THE COPULA FITTING
            # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
            
            self.data_u = np.zeros([np.shape(self.theta)[0],2,n_cop],np_type)  

            for j in range(0,len(edges_now),1):
                edge = edges_now[j]
                
                ## When tr = 0 there is no parent variable.
                ## After check if has to get the CDF from theta flip
                if tr == 0:
                    self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
                else:
                    parent1, inx1, inx2 = parent_var(tr,self.ind_vine,edge)

                    if self.ind_vine[tr-1][edge[0]][0] != parent1: 
                        self.data_u[:,:,j] = np.concatenate((self.theta_flip[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
                    else:
                        self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
            
            ##### Transform data
            self.data_s = trans.forward_u(self.data_u)
#             self.data_s = check_bound3(self.data_s,tf.constant(3.2-1e-6,x.dtype),tf.constant(-3.2+1e-6,x.dtype))
            self.data_x = trans.forward_s(self.data_s)
            
            ##### Grid on P-Q space
            self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))
            
            ############################### FIT BANDWIDTH  #####################################
            
            if self.fitted == False:
            
                #self.copulas.append([])
                opt_bw = tf.TensorArray(x.dtype,size=n_cop)

                if tr > self.vine_depth:
                    if self.parallel:

                        par_copulas = []
                        tau_values = []
#                         if n_cop == 1:
#                             n_cop = 1  ## THIS BECAUSE THERE IS A PROBLEM IN SHAPE 'a' WITH FIT_STUDENT EVEN IF I FORCE IT TO BE THE SAME

                        ### NOT BINNING, PARALLEL, PARAMETRIC
                        start_time = perf_counter()
                        families = ["ind"] # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
                        aic, theta_par, logp = parametric_fit(self.data_u, families, n_cop)
                        time_fit_gauss = perf_counter()  - start_time

                        #print('aic',aic)
                        #print('theta_par',theta_par)

                        for i in range(0,n_cop,1):
                            tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
                            tau_values.append(tau)
                            
                            ind_fam = np.argmin(aic[i])
                            family = families[ind_fam]
                            theta_est = theta_par[i][ind_fam]
                            #print('fam_fit',family)
                            #print('theta_fit',theta_est)
                            cop_p = cop_par_obj(family,theta_est)
                            par_copulas.append(cop_p)
                        self.copulas.append(par_copulas)
                        self.correlations.append(tau_values)
                    else:

                        ### BINNING, NOT PARALLEL, PARAMETRIC

                        par_copulas = []
                        tau_values = []
                        tau_val_bin = []
                        n_cop1 = 1
                        for j in range(0,len(edges_now),1):
                            
                            tau, p_value = kendalltau(self.data_u[:,0,j],self.data_u[:,1,j])
                            tau_values.append(tau)
                            #print('Tau value before binning: ',tau)
                            
                            ind_now = edges_now[j]
                            parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                            ind1 = parent11

                            #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                            if tr == 1:
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                            else:
                                ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                                parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                                if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                    bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                else:
                                    bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)
                            
                            
                            bin_copulas = []
                            tau_binned = []
                            for bb in range(0,self.n_bin,1):
                                #print('bin:',bb)
                                mask = np.where(val_to_bin == bb)
                                u_bin = self.data_u[mask[0],:,j]
                                
                                ### CDF FORCE UNIFORM
                                vv_bin_new = u_bin
                                for zz in range(0,2,1):
                                    vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(u_bin[:,zz],u_bin[:,zz],self.grid_u.ex)
                                u_bin = vv_bin_new[...,tf.newaxis]
                                ###
                                
                                tau, p_value = kendalltau(u_bin[:,0,0],u_bin[:,1,0])
                                #print('Tau value bin -',bb, '- is: ', tau)
                                tau_binned.append(tau)
                                corr = stats.pearsonr(u_bin[:,0,0],u_bin[:,1,0])
                                #print('Corr value  UV space: ',corr[0])
#                                     tau_binned.append(corr[0])

                                start_time = perf_counter()
                                families = ["ind"] # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
                                aic, theta_par, logp = parametric_fit(u_bin, families, n_cop1)
                                time_fit_gauss = perf_counter()  - start_time

                                #print('aic',aic)
                                #print('theta_par',theta_par)

                                ind_fam = np.argmin(aic)

                                cop_p = cop_par_obj(families[ind_fam],theta_par[0][ind_fam])
                                bin_copulas.append(cop_p)

                                #print('fam_fit',families[ind_fam])
                                #print('theta_fit',theta_par[0][ind_fam])
                                #print('--------------------')

                            par_copulas.append(bin_copulas)
                            tau_val_bin.append(tau_binned)

                        self.copulas.append(par_copulas)
                        self.correlations.append(tau_values)
                        self.correlations_bins.append(tau_val_bin)
                else:
                    if (tr == 0) | (self.binning == False):

                        if self.parallel == False:

                            if self.param == True:

                                ### NOT BINNING, NOT PARALLEL, PARAMETRIC
                                par_copulas = []
                                tau_values = []
                                n_cop1 = tf.constant(1,tf.int32)
                                for j in range(0,len(edges_now),1):
                                    
                                    tau, p_value = kendalltau(self.data_u[:,0,j],self.data_u[:,1,j])
                                    tau_values.append(tau)

                                    start_time = perf_counter()
                                    families = param_families # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
                                    aic, theta_par, logp = parametric_fit(self.data_u[:,:,j][...,tf.newaxis], families, n_cop1)
                                    time_fit_gauss = perf_counter()  - start_time

                                    #print('aic',aic)
                                    #print('theta_par',theta_par)

                                    ind_fam = np.argmin(aic)
                                    ## Gaussian
                                    family = families[ind_fam]
                                    theta_est = theta_par[0][ind_fam]

                                    #print('fam_fit',family)
                                    #print('theta_fit',theta_est)

                                    cop_p = cop_par_obj(family,theta_est)
                                    par_copulas.append(cop_p)

                                self.copulas.append(par_copulas)
                                self.correlations.append(tau_values)
                            else: #param flag

                                ### NOT BINNING, NOT PARALLEL, NOT PARAMETRIC
                                opt_bw = tf.TensorArray(x.dtype,size=n_cop)
                                tau_values = []
                                
                                ## Batches
                                batch_size = self.select_batch_size(self.data_s)

                                for i in range(0,n_cop,1):
                          #          print('col:',i)
                                    
                                    tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
                                    tau_values.append(tau)

                                    n_cop1 = tf.constant(1,tf.int32)

                                    grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
                                    data_dict = {'data_s':self.data_s[:,:,i], 'data_x':self.data_x[:,:,i]}
                                    par_dict = {'n_cop':n_cop1, 'batch':batch_size, 'max_iter': [70,100], 'lr':[0.1, 0.03], #lr = 0.1, 0.01
                                                'conv_tol': [1e-5,5e-5], 'opt_method': self.opt_method}  #1e-5

                                    opt = optimization(grid_dict, data_dict, par_dict)
                                    opt_bw = opt_bw.write(i,opt)

                                opt_bw = opt_bw.stack()

                                bw = bandwidth_mul(self.data_x,2,n_cop)
                                bw1 = np.transpose(np.squeeze(opt_bw))*bw
                                
                                ### Check constraints on the bandwidth
                                bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(1e-2,x.dtype))  ##It was 5e-3 but too low

                                copula = copula_obj(bw1.numpy())
                                self.copulas.append(copula)
                                self.correlations.append(tau_values)

              #                  print('opt_bw',bw1)

                        else: #Parallel

                            if self.param == True:

                                par_copulas = []
                                tau_values = []
        #                         if n_cop == 1:
        #                             n_cop = 1  ## THIS BECAUSE THERE IS A PROBLEM IN SHAPE 'a' WITH FIT_STUDENT EVEN IF I FORCE IT TO BE THE SAME

                                ### NOT BINNING, PARALLEL, PARAMETRIC
                                start_time = perf_counter()
                                families = param_families # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
                                aic, theta_par, logp = parametric_fit(self.data_u, families, n_cop)
                                time_fit_gauss = perf_counter()  - start_time

                                #print('aic',aic)
                                #print('theta_par',theta_par)
        
                                for i in range(0,n_cop,1):
                                    tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
                                    tau_values.append(tau)
                                    
                                    ind_fam = np.argmin(aic[i])
                                    family = families[ind_fam]
                                    theta_est = theta_par[i][ind_fam]
                                    #print('fam_fit',family)
                                    #print('theta_fit',theta_est)
                                    cop_p = cop_par_obj(family,theta_est)
                                    par_copulas.append(cop_p)
                                self.copulas.append(par_copulas)
                                self.correlations.append(tau_values)

                            else: #param

                                ### NOT BINNING, PARALLEL, NOT PARAMETRIC
                                n_cop1 = tf.constant(n_cop,tf.int32)

                                ## Batches
                                batch_size = self.select_batch_size(self.data_s)
                                
                                if self.opt_method == 'LL1':
                                    opt_bw = np.zeros((1,n_cop1),np_type)
                                else:
                                    opt_bw = np.zeros((2,n_cop1),np_type)
                                    
                                batch_parallel = batch_paral
                                batch_len1 = n_cop1/batch_parallel
                                batch_len = tf.cast(batch_len1,tf.int32)
                                
                                if batch_len <= 1:
                                    batch_len = n_cop1
                                    batch_parallel = 1
                                else:
                                    while batch_parallel*batch_len < n_cop1:
                                            batch_parallel += 1
                                
                                for j in range(0,batch_parallel,1):

                                    grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,batch_len*j:batch_len*(j+1)]}
                                    data_dict = {'data_s':self.data_s[:,:,batch_len*j:batch_len*(j+1)], 'data_x':self.data_x[:,:,batch_len*j:batch_len*(j+1)]}
                                    par_dict = {'n_cop':tf.shape(self.data_s[:,:,batch_len*j:batch_len*(j+1)])[2], 'batch':batch_size, 'max_iter': [70,100], 'lr':[0.1, 0.03], 
                                                    'conv_tol': [1e-5,5e-5], 'opt_method': self.opt_method}  ## 1e-5,5e-5

                                    opt = optimization(grid_dict, data_dict, par_dict)
                        #            print('opt',opt)
                                    
                                    opt_bw[:,batch_len*j:batch_len*(j+1)] = opt.numpy() #[...,tf.newaxis]
                                    
                                opt_bw = tf.convert_to_tensor(opt_bw)
                                
                                bw = bandwidth_mul(self.data_x,2,n_cop)
                                bw1 = np.transpose(np.squeeze(opt_bw))*bw
                                bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(1e-2,x.dtype))  ##It was 5e-3 but too low

                                
                                copula = copula_obj(bw1.numpy())
                                self.copulas.append(copula)
                                
                                tau_values = []
                                for i in range(0,n_cop,1):
                                    tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
                                    tau_values.append(tau)
                                self.correlations.append(tau_values)

                         #       print('opt_bw',bw1)

                    else: #Binning

                        if self.parallel == False:

                            if self.param == True:

                                ### BINNING, NOT PARALLEL, PARAMETRIC

                                par_copulas = []
                                tau_values = []
                                tau_val_bin = []
                                n_cop1 = 1
                                for j in range(0,len(edges_now),1):
                                    
                                    tau, p_value = kendalltau(self.data_u[:,0,j],self.data_u[:,1,j])
                                    tau_values.append(tau)
                                    #print('Tau value before binning: ',tau)
                                    
                                    ind_now = edges_now[j]
                                    parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                                    ind1 = parent11

                                    #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                                    if tr == 1:
                                        bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                        val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                        val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                    else:
                                        ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                                        parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                                        if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                            bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                            val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                            val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                        else:
                                            bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                            val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                            val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)
                                    
                                    
                                    bin_copulas = []
                                    tau_binned = []
                                    for bb in range(0,self.n_bin,1):
                                        #print('bin:',bb)
                                        mask = np.where(val_to_bin == bb)
                                        u_bin = self.data_u[mask[0],:,j]
                                        
                                        ### CDF FORCE UNIFORM
                                        vv_bin_new = u_bin
                                        for zz in range(0,2,1):
                                            vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(u_bin[:,zz],u_bin[:,zz],self.grid_u.ex)
                                        u_bin = vv_bin_new[...,tf.newaxis]
                                        ###
                                        
                                        tau, p_value = kendalltau(u_bin[:,0,0],u_bin[:,1,0])
                                        #print('Tau value bin -',bb, '- is: ', tau)
                                        tau_binned.append(tau)
                                        corr = stats.pearsonr(u_bin[:,0,0],u_bin[:,1,0])
                                        #print('Corr value  UV space: ',corr[0])
    #                                     tau_binned.append(corr[0])

                                        start_time = perf_counter()
                                        families = param_families # ["ind","gaussian","student","clayton","claytonrot90"] #families to fit
                                        aic, theta_par, logp = parametric_fit(u_bin, families, n_cop1)
                                        time_fit_gauss = perf_counter()  - start_time

                                        #print('aic',aic)
                                        #print('theta_par',theta_par)

                                        ind_fam = np.argmin(aic)

                                        cop_p = cop_par_obj(families[ind_fam],theta_par[0][ind_fam])
                                        bin_copulas.append(cop_p)

                                        #print('fam_fit',families[ind_fam])
                                        #print('theta_fit',theta_par[0][ind_fam])
                                        #print('--------------------')

                                    par_copulas.append(bin_copulas)
                                    tau_val_bin.append(tau_binned)

                                self.copulas.append(par_copulas)
                                self.correlations.append(tau_values)
                                self.correlations_bins.append(tau_val_bin)

                            else: #param

                                ### BINNING, NOT PARALLEL, NOT PARAMETRIC
                                n = len(self.r_matrix)-1
                                tau_values = []
                                tau_val_bin = []
                                
                                for i in range(0,n_cop,1):
                                    #print('col:',i)
                                    
                                    tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
                                    tau_values.append(tau)
                                    #print('Tau value before binning: ',tau)
                                    
                                    tau_binned = []
                                    opt_bin = tf.TensorArray(x.dtype,size=self.n_bin)
    #                                 parent = self.r_matrix[n-tr,n-2-i] - 1 # Because first node starts from 1 (and not from 0)
    #                                 bins = create_bins(self.theta[:,0,parent],self.n_bin)
    #                                 val_to_bin = np.digitize(self.theta[:,0,parent], bins) -1
                                    ind_now = edges_now[i]  #j
                                    parent, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
    #                                 print('ind_now',ind_now)
    #                                 print('par',parent11)
    #                                 print('prev',ind_vine[tr-1][ind_now[0]],ind_vine[tr-1][ind_now[1]])
                                    ind1 = parent
                                    
                                    #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                                    if tr == 1:
                                        bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                        val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                        val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                    else:
                                        ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                                        parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                                        if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                            bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                            val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                            val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                        else:
                                            bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                            val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                            val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)

                                    for bb in range(0,self.n_bin,1):
                                        #print('bin:',bb)
                                        mask = tf.where(tf.equal(val_to_bin,bb))
                                        n_cop1 = tf.constant(1,tf.int32)
                                        
                                        data_u_bin = tf.gather_nd(self.data_u[:,:,i],mask)
                                        tau, p_value = kendalltau(data_u_bin[:,0],data_u_bin[:,1])
                                        tau_binned.append(tau)
                                        #print('Tau value bin -',bb, '- is: ', tau)
                                        
                                        ### CDF FORCE UNIFORM
                                        data_u_bin_new = np.zeros(np.shape(data_u_bin),np_type)
                                        for zz in range(0,2,1):
                                            data_u_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],data_u_bin[:,zz],self.grid_u.ex)
                                        data_u_bin = data_u_bin_new[...,np.newaxis]
                                        ###

                                        trans = Transform(1)
                                        data_s_bin = trans.forward_u(data_u_bin)#[:,:,0]
                                        data_x_bin = trans.forward_s(data_s_bin)

    #                                     data_s_bin = tf.gather_nd(self.data_s[:,:,i],mask)
    #                                     data_x_bin = tf.gather_nd(self.data_x[:,:,i],mask)

                                        grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
                                        data_dict = {'data_s':data_s_bin, 'data_x':data_x_bin}
                                        par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,70], 'lr':[0.1, 0.03],
                                                    'conv_tol': [1e-4,1e-4], 'opt_method': self.opt_method}

                                        opt = optimization(grid_dict, data_dict, par_dict)
                                        
                                        bw = bandwidth_mul(data_x_bin,2,n_cop1)
                                        bw1 = opt*bw
                                        bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(1e-2,x.dtype))  ##It was 5e-3 but too low
                                        
                                        opt_bin = opt_bin.write(bb,bw1)
    #                                     opt_bin = opt_bin.write(bb,opt)
                                    opt_bin = opt_bin.stack()
    #                                 opt_bin = tf.reshape(opt_bin,[tf.shape(opt)[0],self.n_bin])
                                    opt_bin = tf.reshape(opt_bin,[2,self.n_bin])

                             #       print(opt_bin)
                                    tau_val_bin.append(tau_binned)

                                    opt_bw = opt_bw.write(i,opt_bin)
                                opt_bw = opt_bw.stack()
    #                             opt_bw = tf.reshape(opt_bw,[tf.shape(opt)[0],n_cop,self.n_bin])
                                opt_bw = tf.reshape(opt_bw,[2,n_cop,self.n_bin])

    #                             bw = bandwidth_mul(self.data_x,2,n_cop)
                                
    #                             bw1 = np.zeros((2,n_cop,self.n_bin),np_type)
    #                             for i in range(0,n_cop,1):
    #                                 bw1[:,i,:] = opt_bw[:,i,:]*bw[:,i][...,np.newaxis]
                                
    #                             ### If bw < 5e-3 gives nan
    #                             bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(5e-3,x.dtype))
                                bw1 = opt_bw
        
                                copula = copula_obj(bw1.numpy())
                                self.copulas.append(copula)
                                self.correlations.append(tau_values)
                                self.correlations_bins.append(tau_val_bin)

                     #           print('opt_bw',bw1)

                        else: #parallel

                            if self.param == True:
                                print('Miss to implement')
                            else:
                                tau_values = []
                                tau_val_bin = []
                                
                                len_bin = tf.shape(self.theta)[0]/self.n_bin
                                len_bin = tf.cast(len_bin,tf.int32)
    #                             print('len bin',len_bin)
                                data_s_bin = np.zeros([len_bin,tf.shape(self.data_s)[1],n_cop,self.n_bin],np_type)
                                data_x_bin = np.zeros([len_bin,tf.shape(self.data_s)[1],n_cop,self.n_bin],np_type)

                                for i in range(0,n_cop,1):
                                    
                                    tau, p_value = kendalltau(self.data_u[:,0,i],self.data_u[:,1,i])
                                    tau_values.append(tau)
                                    #print('Tau value before binning: ',tau)
                                    
                                    tau_binned = []
                                    ind_now = edges_now[i] #j
                                    parent, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                                    ind1 = parent
                                    
                                    #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                                    if tr == 1:
                                        bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                        val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                        val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                    else:
                                        ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                                        parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                                        if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                            bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                            val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                            val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                        else:
                                            bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                            val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                            val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)
                            
                                    for bb in range(0,self.n_bin,1):
                                        mask = tf.where(tf.equal(val_to_bin,bb))
                                        #print('mask bin',np.shape(mask))
                                        
                                        ##### FIXED FOR THE UNIFORMITY
                                        data_u_bin = tf.gather_nd(self.data_u[:,:,i],mask)

                                        ### CDF FORCE UNIFORM
                                        data_u_bin_new = np.zeros(np.shape(data_u_bin),np_type)
                                        for zz in range(0,2,1):
                                            data_u_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],data_u_bin[:,zz],self.grid_u.ex)
                                        data_u_bin = data_u_bin_new[...,np.newaxis]
                                        ###
                                        #print(zz)
                                        #print(data_u_bin.shape)
                                        trans = Transform(1)
                                        data_s_bin[:,:,i,bb] = trans.forward_u(data_u_bin)[:,:,0]
                                        data_x_bin[:,:,i,bb] = trans.forward_s(data_s_bin[:,:,i,bb][...,tf.newaxis])[:,:,0]
                                        
    #                                     data_s_bin[:,:,i,bb] = tf.gather_nd(self.data_s[:,:,i],mask)
    #                                     data_x_bin[:,:,i,bb] = tf.gather_nd(self.data_x[:,:,i],mask)
                                        
                                        u_bin = tf.gather_nd(self.data_u[:,:,i],mask)
                                        tau, p_value = kendalltau(u_bin[:,0],u_bin[:,1])
                                        tau_binned.append(tau)
                                        #print('Tau value bin -',bb, '- is: ', tau)
                                        corr = stats.pearsonr(u_bin[:,0],u_bin[:,1])
                                        #print('Corr value  UV space: ',corr[0])
                                    tau_val_bin.append(tau_binned)


                                opt_bw = np.zeros((2,n_cop,self.n_bin),np_type)
                                batch_parallel = batch_paral
                                batch_len1 = n_cop/batch_parallel
                                batch_len = tf.cast(batch_len1,tf.int32)

                                #                             if tf.cast(batch_len1,x.dtype) > tf.cast(batch_len,x.dtype):
                                if batch_len <= 1:
                                    batch_len = n_cop
                                    batch_parallel = 1
                                else:
                                    while batch_parallel*batch_len < n_cop:
                                            batch_parallel += 1

                                for j in range(0,batch_parallel,1):

                                    for bb in range(0,self.n_bin,1):
                                        ## UPDATE THETA
                                        n_batch = tf.shape(data_s_bin[:,:,batch_len*j:batch_len*(j+1),bb])[2]

                                        grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,batch_len*j:batch_len*(j+1)]}
                                        data_dict = {'data_s':data_s_bin[:,:,batch_len*j:batch_len*(j+1),bb], 'data_x':data_x_bin[:,:,batch_len*j:batch_len*(j+1),bb]}
                                        par_dict = {'n_cop':n_batch, 'batch':batch_size, 
                                                    'max_iter': [70,100], 'lr':[0.1, 0.03], 'conv_tol': [1e-4,1e-4], 'opt_method': self.opt_method}

                                        opt = optimization(grid_dict, data_dict, par_dict)

                                        bw = bandwidth_mul(data_x_bin[:,:,batch_len*j:batch_len*(j+1),bb],2,n_batch)

                                        bw1 = opt*bw
                                        bw1 = check_bound3(bw1,tf.constant(2,x.dtype),tf.constant(1e-2,x.dtype))

                                        opt_bw[:,batch_len*j:batch_len*(j+1),bb] = bw1.numpy()
                                
                    #            print('opt_bw',opt_bw)
                                copula = copula_obj(opt_bw)
                                self.copulas.append(copula)
                                self.correlations.append(tau_values)
                                self.correlations_bins.append(tau_val_bin)
                   
            ##############################  UPDATE THETA #####################################
            
            n = np.shape(self.r_matrix)[0] -1
            
            #### if optimal or random, flip_flag = [True,False,True,False,...] in order to evaluate all possible orders
            #### Otherwise just stores when to flip based on the parent variable
            ## Flip_flap stores boolean if flipped or not
            ## ind_edge_rel1 refers to the index of the copula
            
            flip_flag1 = []
            ind_edge_rel1 = []
            parent_all = []
            if (self.vine_family == 'r-vine'):
                if (self.method == 'optimal'): # | (self.method == 'random'):
                    for j in range(0,len(edges_now),1):
                        edge = edges_now[j]
                        flip_flag1.append(True)
                        flip_flag1.append(False)
                        ind_edge_rel1.append(j)
                        ind_edge_rel1.append(j)
                        parent_all.append([edge[0],edge[1]])
                else:
                    flip_flag1, ind_edge_rel1, parent_all = flip_check_all(self.ind_vine, tr, self.binning, self.n_bin)
            else:
                flip_flag1, ind_edge_rel1, parent_all = flip_check_all(self.ind_vine, tr, self.binning, self.n_bin)
            
            ## vv_s is another variable which stores the data_s taking into account also the flipping
            
            vv_u = np.zeros((np.shape(self.data_u)[0],np.shape(self.data_u)[1],len(ind_edge_rel1)),self.data_u.dtype)
            vv_s = np.zeros((np.shape(self.data_u)[0],np.shape(self.data_u)[1],len(ind_edge_rel1)),self.data_u.dtype)

            for j in range(0,len(ind_edge_rel1),1):
                ind_edge = ind_edge_rel1[j]
                edge = edges_now[ind_edge]

                if tr > self.vine_depth:
                    if (tr==0) | (self.binning == False):

                        cop_p = self.copulas[tr][ind_edge]

                        if flip_flag1[j] == True:
                            vv = self.data_u[:,:,ind_edge]
                            self.theta_flip[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                        else:
                            vv = np.flip(self.data_u[:,:,ind_edge],1)
                            self.theta[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))

                    else: #binning

                        parent11, inx1, inx2 = parent_var(tr,self.ind_vine,edge)
                        ind1 = parent11
        
                        
                        #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                        if tr == 1:
                            bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                            val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                            val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                        else:
                            ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                            if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                            else:
                                bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)

                        flip_flag_bin = []
#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
                        
                        for bb in range(0,self.n_bin,1):

                            cop_p = self.copulas[tr][ind_edge][bb]

                            mask = np.where(val_to_bin == bb)

                            if flip_flag1[j] == True:
                                vv = self.data_u[:,:,ind_edge]
                                vv_bin = vv[mask[0],:]
                                
                                ### CDF FORCE UNIFORM
                                vv_bin_new = vv_bin
                                for zz in range(0,2,1):
                                    vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],self.grid_u.ex)
                                vv_bin = vv_bin_new
                                ###
#                                 print('flip')
#                                 print('bin-',bb,',bef: ',vv_bin)
                                self.theta_flip[mask[0],tr+1,ind_edge] = copulaccdf(cop_p,vv_bin) 
#                                 print('bin-',bb,': ',self.theta_flip[mask[0],tr+1,ind_edge])
                            else:
                                vv = np.flip(self.data_u[:,:,ind_edge],1)
                                vv_bin = vv[mask[0],:]
                                
                                ### CDF FORCE UNIFORM
                                vv_bin_new = vv_bin
                                for zz in range(0,2,1):
                                    vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],self.grid_u.ex)
                                vv_bin = vv_bin_new
                                ###
#                                 print('no')
#                                 print('bin-',bb,',bef: ',vv_bin)
                                self.theta[mask[0],tr+1,ind_edge] = copulaccdf(cop_p,vv_bin)
                    # cop_p = self.copulas[tr][ind_edge]

                    # if flip_flag1[j] == True:
                    #     vv = self.data_u[:,:,ind_edge]
                    #     self.theta_flip[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                    # else:
                    #     vv = np.flip(self.data_u[:,:,ind_edge],1)
                    #     self.theta[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                else:
                    if self.param == True:
                        if (tr==0) | (self.binning == False):

                            cop_p = self.copulas[tr][ind_edge]

                            if flip_flag1[j] == True:
                                vv = self.data_u[:,:,ind_edge]
                                self.theta_flip[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                            else:
                                vv = np.flip(self.data_u[:,:,ind_edge],1)
                                self.theta[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))

                        else: #binning

                            parent11, inx1, inx2 = parent_var(tr,self.ind_vine,edge)
                            ind1 = parent11
            
                            
                            #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                            if tr == 1:
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                            else:
                                ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                                parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                                if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                    bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                else:
                                    bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)

                            flip_flag_bin = []
    #                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
                            
                            for bb in range(0,self.n_bin,1):

                                cop_p = self.copulas[tr][ind_edge][bb]

                                mask = np.where(val_to_bin == bb)

                                if flip_flag1[j] == True:
                                    vv = self.data_u[:,:,ind_edge]
                                    vv_bin = vv[mask[0],:]
                                    
                                    ### CDF FORCE UNIFORM
                                    vv_bin_new = vv_bin
                                    for zz in range(0,2,1):
                                        vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],self.grid_u.ex)
                                    vv_bin = vv_bin_new
                                    ###
    #                                 print('flip')
    #                                 print('bin-',bb,',bef: ',vv_bin)
                                    self.theta_flip[mask[0],tr+1,ind_edge] = copulaccdf(cop_p,vv_bin) 
    #                                 print('bin-',bb,': ',self.theta_flip[mask[0],tr+1,ind_edge])
                                else:
                                    vv = np.flip(self.data_u[:,:,ind_edge],1)
                                    vv_bin = vv[mask[0],:]
                                    
                                    ### CDF FORCE UNIFORM
                                    vv_bin_new = vv_bin
                                    for zz in range(0,2,1):
                                        vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],self.grid_u.ex)
                                    vv_bin = vv_bin_new
                                    ###
    #                                 print('no')
    #                                 print('bin-',bb,',bef: ',vv_bin)
                                    self.theta[mask[0],tr+1,ind_edge] = copulaccdf(cop_p,vv_bin)
    #                                 print('bin-',bb,': ',self.theta[mask[0],tr+1,ind_edge])
                    else: #param
                        if flip_flag1[j] == True:
                            vv_u[:,:,j] = np.flip(self.data_u[:,:,ind_edge],1)
                            vv_s[:,:,j] = np.flip(self.data_s[:,:,ind_edge],1) #self.data_s[:,:,j] #Flip cambia per npc
                        else:
                            vv_u[:,:,j] = self.data_u[:,:,ind_edge]
                            vv_s[:,:,j] = self.data_s[:,:,ind_edge] #np.flip(self.data_s[:,:,j],1)

            self.flip_flag.append(flip_flag1)
            self.ind_edge_rel.append(ind_edge_rel1)

            
            if self.param == False:
                
                n_eval = len(self.ind_edge_rel[tr])
                self.data_u = vv_u[:,:,:n_eval]
                self.data_s = vv_s[:,:,:n_eval]
                trans = Transform(n_eval)
                self.data_x = trans.forward_s(self.data_s)
                grid_x = trans.forward_s(self.grid_s.ex)
                
                del vv_s
                
                if tr <= self.vine_depth:
                    if (tr == 0) | (self.binning == False):
                        
                    ## Batches
                        batch_size = self.select_batch_size(self.data_s)

                        grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':grid_x}
                        data_dict = {'data_s':self.data_s, 'data_x':self.data_x, 'theta':self.theta, 'theta_flip':self.theta_flip}
                        par_dict = {'copulas': self.copulas[tr], 'n_eval':tf.convert_to_tensor(n_eval), 'batch':batch_size, 'batch_cdf':batch_size_cdf, 'tr':tr,
                                'ind_edge_rel': self.ind_edge_rel[tr], 'flip_flag': self.flip_flag[tr]}

                        self.copulas[tr].pd_grid_uv, self.copulas[tr].cdf, self.theta, self.theta_flip = evaluate_fit(data_dict, grid_dict, par_dict)
                        
                    else: #binning
                        
                        len_bin = tf.shape(self.theta)[0]/self.n_bin
                        len_bin = tf.cast(len_bin,tf.int32)
                        data_s_bin = np.empty([len_bin,tf.shape(self.data_s)[1],n_eval,self.n_bin],np_type)
                        data_x_bin = np.empty([len_bin,tf.shape(self.data_s)[1],n_eval,self.n_bin],np_type)

                        for i in range(0,n_eval,1):
                            ind_edge = self.ind_edge_rel[tr][i]
    #                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
    #                         ind1 = np.where(self.nodes == parent11)
    #                         ind1 = ind1[0][0]

    #                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
    #                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
                            ind_now = edges_now[ind_edge]
                            parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                            ind1 = parent11
    #                                 print('ind_now',ind_now)
    #                                 print('par',parent11)
    #                                 print('prev',ind_vine[tr-1][ind_now[0]],ind_vine[tr-1][ind_now[1]])

                            #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                            if tr == 1:
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                            else:
                                ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                                parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                                if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                    bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                else:
                                    bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)
                            
                            for bb in range(0,self.n_bin,1):
                                mask = tf.where(tf.equal(val_to_bin,bb))
                                
                                ##### FIXED FOR THE UNIFORMITY
                                data_u_bin = tf.gather_nd(self.data_u[:,:,i],mask)
                                
                                ### CDF FORCE UNIFORM
                                data_u_bin_new = np.zeros(np.shape(data_u_bin),np_type)
                                for zz in range(0,2,1):
                                    data_u_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],data_u_bin[:,zz],self.grid_u.ex)
                                data_u_bin = data_u_bin_new[...,np.newaxis]
                                ###
                                
                                trans = Transform(1)
                                data_s_bin[:,:,i,bb] = trans.forward_u(data_u_bin)[:,:,0]
                                data_x_bin[:,:,i,bb] = trans.forward_s(data_s_bin[:,:,i,bb][...,tf.newaxis])[:,:,0]
                                
    #                             data_s_bin[:,:,i,bb] = tf.gather_nd(self.data_s[:,:,i],mask)
    #                             data_x_bin[:,:,i,bb] = tf.gather_nd(self.data_x[:,:,i],mask)
                    
                        self.copulas[tr].pd_grid_uv = np.zeros([self.knots,self.knots,n_eval,self.n_bin],np_type)
                        self.copulas[tr].cdf = np.zeros([self.knots,self.knots,n_eval,self.n_bin],np_type)
                        
                        for bb in range(0,self.n_bin,1):
                            ## UPDATE THETA

                            grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':grid_x}
                            data_dict = {'data_s':data_s_bin[:,:,:,bb], 'data_x':data_x_bin[:,:,:,bb]} #],'bin':bb
                            par_dict = {'bw': tf.convert_to_tensor(self.copulas[tr].opt_bw[:,:,bb]), 'n_cop':tf.convert_to_tensor(n_eval), 'batch':tf.constant(2,tf.int32), 'tr':tr, 'ind_edge_rel':self.ind_edge_rel[tr]}

                            self.copulas[tr].pd_grid_uv[:,:,:,bb], self.copulas[tr].cdf[:,:,:,bb] = evaluate_fit_bin(data_dict, grid_dict, par_dict)
                        
                        interp_cdf_bin = np.zeros([tf.shape(self.theta)[0],n_eval],np_type)
                        for i in range(0,n_eval,1):
                   #         print('col:',i)
                            ind_edge = self.ind_edge_rel[tr][i]
    #                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
    #                         ind1 = np.where(self.nodes == parent11)
    #                         ind1 = ind1[0][0]

    #                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)

    #                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
                            ind_now = edges_now[ind_edge]
                            parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                            ind1 = parent11

                            #### TAKE INDEX OF THE PARENT IF IT WAS FLIPPED OR NOT
                            if tr == 1:
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                            else:
                                ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                                parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  

                                if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                    bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                else:
                                    bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)

                            for bb in range(0,self.n_bin,1):
    #                             print('bin:',bb)
                                
                                mask = tf.where(tf.equal(val_to_bin,bb))

                                ## Update theta  
                                ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin[:,:,i,bb],self.grid_s.min,self.grid_s.max,self.copulas[tr].cdf[:,:,i,bb],axis=-2)
                                ccdf_data = tf.squeeze(ccdf_data)
                                    
    #                             u_bin = tf.gather_nd(self.data_u[:,:,i],mask)
                            
    #                             print('bin-',bb,',bef: ',u_bin)
    #                             print('bin-',bb,',bef: ',ccdf_data)
                                interp_cdf, mar_s, mar_p = kernel_cdf(ccdf_data,ccdf_data,self.grid_u.ex)
    #                             print('bin-',bb,': ',interp_cdf)
                                
                                if self.flip_flag[tr][i] == True:
                                    self.theta_flip[mask,tr+1,ind_edge] = tf.reshape(interp_cdf,[tf.shape(interp_cdf)[0],1])
                                else:
                                    self.theta[mask,tr+1,ind_edge] = tf.reshape(interp_cdf,[tf.shape(interp_cdf)[0],1])
        
        ### After finding the optimal or the random vine, it stores the connection in the r_matrix
        # if self.vine_depth == self.n_cop-1:  #vine_depth
        if self.vine_family == 'r-vine':
            if (self.method == 'optimal'): # | (self.method == 'random'):
                #print('SAVED OPTIMAL OR RANDOM R-VINE')
                self.r_matrix, self.E, self.nodes = prepare_optimal(self.n_cop,self.ind_vine)
        
        return
    
    
        ################################ EVALUATION ##############################################################################################
    def evaluation(self, points):
    
        if isinstance(points, tuple):
            tensor_dtype = points[0].dtype
            #points = tf.convert_to_tensor(points, dtype=tensor_dtype)
        #elif isinstance(points, tf.Tensor):  # Assuming you're using TensorFlow
        else:
            tensor_dtype = points.dtype

        d = self.n_cop
        ## Create Fp
        self.Fp = np.zeros([tf.shape(points)[0],self.n_cop,self.n_cop],self.data_u.dtype)
        self.Fp_flip = np.zeros([tf.shape(points)[0],self.n_cop,self.n_cop],self.data_u.dtype)
        ### Create logf
        logf = tf.zeros([tf.shape(points)[0],tf.shape(points)[1],self.n_cop],tensor_dtype) #self.n_cop = vine_depth
        self.logf_flip = tf.zeros(tf.shape(points),tensor_dtype)
        
        #print(self.Fp.shape)
        
        for i in range(0, d, 1):


            interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(self.margin[i].ker,points[:,i],self.grid_u.ex)
            if len(interp_cdf_poi.shape) == 1:
                self.Fp[:, 0, i] = interp_cdf_poi  # Assuming you want to assign the first column of interp_cdf_poi    
            else :
                self.Fp[:, 0, i] = interp_cdf_poi[:, 0]  # Assuming you want to assign the first column of interp_cdf_poi
            den1, mden1 = kernel_pdf2(self.margin[i].ker)
            inter = interp_pdf(points[:, i], mden1, den1)  # interp1d_np
            '''
            for i in range(0,d,1):
            interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(self.margin[i].ker,points[:,i],self.grid_u.ex)
#            self.Fp[:,0,i] = interp_cdf_poi.numpy()            
#            den1,mden1 = kernel_pdf2(self.margin[i].ker)      
#            inter = interp_pdf(points[:,i], mden1, den1) #interp1d_np
            self.Fp[:, 0, i] = interp_cdf_poi[:, 0]  # Assuming you want to assign the first column of interp_cdf_poi
            den1, mden1 = kernel_pdf2(self.margin[i].ker)
            inter = interp_pdf(points[:, i], mden1, den1)  # interp1d_np
            '''
            # Product of pdf is the sum of logarithm - Product of pdf margingales evaluated on copula samples
            logf = update_tensor(logf,tf.math.log(inter),i,0)
           
            del den1,mden1, inter, interp_cdf_poi #,logf_tmp
            
        self.logf = logf.numpy()
        logf_marginal = self.logf
        
        for tr in range(0,d-1,1): #d-1
            #    print('Row theta:',tr)
            
            if self.vine_family == 'r-vine':
                    if (self.method == 'optimal'): # | (self.method == 'random'):
                        self.flip_flag[tr], self.ind_edge_rel[tr], parent_all = flip_check_all(self.ind_vine, tr, self.binning, self.n_bin)
            
            # Number of copuals to evaluate and create Transform object
            
            n_eval = len(self.ind_edge_rel[tr])
            trans = Transform(n_eval)
            #  print('n to eval in the row:',n_eval)
            
            ## Edges of the vine
            edges_now = self.ind_vine[tr]
            
            ######### FROM THETA MATRIX TAKE THE DATA CDF FOR THE COPULA FITTING
            # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
            
            self.data_u = np.zeros([np.shape(self.theta)[0],2,n_eval],self.data_u.dtype)
            self.points_u = np.zeros([np.shape(self.Fp)[0],2,n_eval],self.data_u.dtype)
            for j in range(0,n_eval,1):
                edge = edges_now[self.ind_edge_rel[tr][j]]
                if tr == 0:
                    self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
                    self.points_u[:,:,j] = np.concatenate((self.Fp[:,tr,edge[0]][...,np.newaxis],self.Fp[:,tr,edge[1]][...,np.newaxis]),1)
                else:
                    parent1, inx1, inx2 = parent_var(tr,self.ind_vine,edge)
                    
                    if self.ind_vine[tr-1][edge[0]][0] != parent1: 
                        self.data_u[:,:,j] = np.concatenate((self.theta_flip[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
                        self.points_u[:,:,j] = np.concatenate((self.Fp_flip[:,tr,edge[0]][...,np.newaxis],self.Fp[:,tr,edge[1]][...,np.newaxis]),1)
                    else:
                        self.data_u[:,:,j] = np.concatenate((self.theta[:,tr,edge[0]][...,np.newaxis],self.theta[:,tr,edge[1]][...,np.newaxis]),1)
                        self.points_u[:,:,j] = np.concatenate((self.Fp[:,tr,edge[0]][...,np.newaxis],self.Fp[:,tr,edge[1]][...,np.newaxis]),1)
                
                if self.param == False:
                    if self.flip_flag[tr][j] == True:
                        self.data_u[:,:,j] = np.flip(self.data_u[:,:,j],1)
                        self.points_u[:,:,j] = np.flip(self.points_u[:,:,j],1)
            
            ### Transform data
            self.data_s = trans.forward_u(self.data_u)
#             self.data_s = check_bound3(self.data_s,tf.constant(3.2-1e-6,points.dtype),tf.constant(-3.2+1e-6,points.dtype))
            self.data_x = trans.forward_s(self.data_s)
            
            ### Transform points
            self.points_s = trans.forward_u(self.points_u)
#             self.points_s = check_bound3(self.points_s,tf.constant(3.2-1e-6,points.dtype),tf.constant(-3.2+1e-6,points.dtype))
            self.points_x = trans.forward_s(self.points_s)
            
            ### Grid on P-Q space
            self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))
            
            n = np.shape(self.r_matrix)[0] -1
            
            if self.param == False:

                if tr > self.vine_depth:
                    if (tr==0) | (self.binning == False):
                    
                        for j in range(0,len(self.ind_edge_rel[tr]),1):

                            ind_edge = self.ind_edge_rel[tr][j]

                            cop_p = self.copulas[tr][ind_edge]

                            if self.flip_flag[tr][j] == True:
                                vv = self.points_u[:,:,j]
                                self.Fp_flip[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                            else:
                                vv = np.flip(self.points_u[:,:,j],1)
                                if (self.vine_family == 'c-vine') & (cop_p.family == 'ind') & (j == 0):
                                        vv = self.points_u[:,:,j]
                                self.Fp[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                                
                            
    #                         vv = np.flip(vv,1)   Does not change the order in the copulapdf it seems at least for gaussian
                            vv = vv[...,np.newaxis]
                            pd_points = np.squeeze(copulapdf(cop_p,vv))

                            # Update logf
                            logftr = tf.math.log(pd_points)
                
                            self.logf[:,ind_edge,tr+1] = np.squeeze(logftr) #.numpy()

                    else:
                        
                        log_f_bin = np.zeros([tf.shape(self.logf)[0],n_eval],self.data_u.dtype)
                        for j in range(0,len(self.ind_edge_rel[tr]),1):
                            ind_edge = self.ind_edge_rel[tr][j]

    #                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
    #                         ind1 = np.where(self.nodes == parent11)
    #                         ind1 = ind1[0][0]

    #                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
    #                         val_to_bin = np.digitize(self.Fp[:,0,ind1], bins) -1
                            ind_now = edges_now[ind_edge]
                            parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                            ind1 = parent11
    #                                 print('ind_now',ind_now)
    #                                 print('par',parent11)
    #                                 print('prev',ind_vine[tr-1][ind_now[0]],ind_vine[tr-1][ind_now[1]])
                            
                            if tr == 1:
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                            else:
                                ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                                parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  
                                if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                    bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                    val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                                else:
                                    bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)
                                    val_to_bin1 = np.digitize(self.Fp_flip[:,tr-1,ind1], bins) -1
                            
                            for bb in range(0,self.n_bin,1):

                                cop_p = self.copulas[tr][ind_edge][bb]

                                mask = np.where(val_to_bin == bb)
                                mask1 = np.where(val_to_bin1 == bb)
                                data_u_bin = self.data_u[mask[0],:,j]

                                if self.flip_flag[tr][j] == True:

                                    vv = self.points_u[:,:,j]
                                    vv_bin = vv[mask[0],:]
                                    
                                    ### CDF FORCE UNIFORM
                                    vv_bin_new = vv_bin
                                    for zz in range(0,2,1):
                                        vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],vv_bin[:,zz],self.grid_u.ex)
                                    vv_bin = vv_bin_new
                                    ###
                                    self.Fp_flip[mask[0],tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv_bin))
                                else:
                                    vv = np.flip(self.points_u[:,:,j],1)
                                    if (self.vine_family == 'c-vine') & (cop_p.family == 'ind') & (j == 0):
                                        vv = self.points_u[:,:,j]
            
                                    vv_bin = vv[mask[0],:]
                                    
                                    ### CDF FORCE UNIFORM
                                    vv_bin_new = vv_bin
                                    for zz in range(0,2,1):
                                        vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],vv_bin[:,zz],self.grid_u.ex)
                                    vv_bin = vv_bin_new
                                    ###
                                    
                                    self.Fp[mask[0],tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv_bin))

                                vv_bin = vv_bin[...,np.newaxis]
                                pd_points = np.squeeze(copulapdf(cop_p,vv_bin))

                                ## Update logf
                                logftr = tf.math.log(pd_points) 
                                
                                self.logf[mask[0],ind_edge,tr+1] = logftr
#                     for j in range(0,len(self.ind_edge_rel[tr]),1):

#                         ind_edge = self.ind_edge_rel[tr][j]

#                         cop_p = self.copulas[tr][ind_edge]

#                         if self.flip_flag[tr][j] == True:
#                             vv = self.points_u[:,:,j]
#                             self.Fp_flip[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
#                         else:
#                             vv = np.flip(self.points_u[:,:,j],1)
#                             if (self.vine_family == 'c-vine') & (cop_p.family == 'ind') & (j == 0):
#                                     vv = self.points_u[:,:,j]
#                             self.Fp[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                            
                        
# #                         vv = np.flip(vv,1)   Does not change the order in the copulapdf it seems at least for gaussian
#                         vv = vv[...,np.newaxis]
#                         pd_points = np.squeeze(copulapdf(cop_p,vv))

#                         # Update logf
#                         logftr = tf.math.log(pd_points)
            
#                         self.logf[:,ind_edge,tr+1] = np.squeeze(logftr) #.numpy()
                else:
                    pd_grid_uv = self.copulas[tr].pd_grid_uv
                    cdf1 = self.copulas[tr].cdf   
                    if (tr==0) | (self.binning == False):

                        for j in range(0,n_eval,1):
                            
                            ind_edge = self.ind_edge_rel[tr][j]
                            
                            ind_pd = j
                            if self.vine_family == 'r-vine':
                                if (self.method == 'optimal'): # | (self.method == 'random'):
                                    ind_pd = self.ind_edge_rel[tr][j]*2
                                    if self.flip_flag[tr][j] == True:
                                        ind_pd = ind_pd
                                    else:
                                        ind_pd = ind_pd + 1

                            ## Batches
                            batch_size = self.select_batch_size(self.data_s)

                            ccdf_data = tfp.math.batch_interp_regular_nd_grid(self.data_s[:,:,j],self.grid_s.min,self.grid_s.max,cdf1[:,:,ind_pd],axis=-2)
                            
                            pd_points, ccdf_points = evaluate_points(self.points_s[:,:,j], batch_size, self.grid_s, cdf1[:,:,ind_pd], pd_grid_uv[:,:,ind_pd])   

                            # Update logf
                            logftr = tf.math.log(pd_points) 
                            
                            self.logf[:,ind_edge,tr+1] = tf.squeeze(logftr).numpy() #update_tensor(logf,logftr,j,tr+1)

                            # Update Fp
                            
                            interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data,ccdf_points,self.grid_u.ex)

                            if self.flip_flag[tr][j] == False:
                                self.Fp[:,tr+1,ind_edge] = interp_cdf_poi
                            else:
                                self.Fp_flip[:,tr+1,ind_edge] = interp_cdf_poi

                    else: #binning

                        batch_size = tf.constant(1,tf.int32)

                        len_bin = tf.shape(self.theta)[0]/self.n_bin
                        len_bin = tf.cast(len_bin,tf.int32)
                        data_s_bin = np.zeros([len_bin,tf.shape(self.data_s)[1],n_eval,self.n_bin],self.data_u.dtype)
                        len_bin1 = tf.shape(self.Fp)[0]/self.n_bin   
                        len_bin1 = tf.cast(len_bin1,tf.int32)
                        points_s_bin = []

                        for i in range(0,n_eval,1):
                            ind_edge = self.ind_edge_rel[tr][i]

    #                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
    #                         ind1 = np.where(self.nodes == parent11)
    #                         ind1 = ind1[0][0]

    #                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)

    #                         val_to_bin = np.digitize(self.theta[:,0,ind1], bins) -1
    #                         val_to_bin1 = np.digitize(self.Fp[:,0,ind1], bins) -1
                            ind_now = edges_now[ind_edge]
                            parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                            ind1 = parent11
    #                                 print('ind_now',ind_now)
    #                                 print('par',parent11)
    #                                 print('prev',ind_vine[tr-1][ind_now[0]],ind_vine[tr-1][ind_now[1]])
                            
                            if tr == 1:
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                            else:
                                ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                                parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  
                                if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                    bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                    val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                                else:
                                    bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                    val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                    val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)
                                    val_to_bin1 = np.digitize(self.Fp_flip[:,tr-1,ind1], bins) -1

                            points_s_bin1 = []
                            for bb in range(0,self.n_bin,1):

                                mask = tf.where(tf.equal(val_to_bin,bb))
                                mask1 = tf.where(tf.equal(val_to_bin1,bb))
                                
                                ##### FIXED FOR THE UNIFORMITY
                                data_u_bin = tf.gather_nd(self.data_u[:,:,i],mask)
                                points_u_bin = tf.gather_nd(self.points_u[:,:,i],mask1)
                                
                                ### CDF FORCE UNIFORM
                                data_u_bin_new = np.zeros(np.shape(data_u_bin),self.data_u.dtype)
                                points_u_bin_new = np.zeros(np.shape(points_u_bin),self.data_u.dtype)
                                for zz in range(0,2,1):
                                    data_u_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],data_u_bin[:,zz],self.grid_u.ex)
                                    points_u_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],points_u_bin[:,zz],self.grid_u.ex)
                                data_u_bin = data_u_bin_new[...,np.newaxis]
                                points_u_bin = points_u_bin_new #[...,np.newaxis]
                                ###
                                
                                trans = Transform(1)
                                data_s_bin[:,:,i,bb] = trans.forward_u(data_u_bin)[:,:,0]
                                points_s_bin1.append(trans.forward_u(points_u_bin))#[:,:,0]

    #                             data_s_bin[:,:,i,bb] = tf.gather_nd(self.data_s[:,:,i],mask)
    #                             points_s_bin1.append(tf.gather_nd(self.points_s[:,:,i],mask1))

                            points_s_bin.append(points_s_bin1)

                        log_f_bin = np.zeros([tf.shape(self.logf)[0],n_eval],self.data_u.dtype)
                        Fp_bin = np.zeros([tf.shape(self.Fp)[0],n_eval],self.data_u.dtype)
                        for i in range(0,n_eval,1):

                            ind_edge = self.ind_edge_rel[tr][i]
                            
                            ind_now = edges_now[ind_edge]
                            parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                            ind1 = parent11
    #                                 print('ind_now',ind_now)
    #                                 print('par',parent11)
    #                                 print('prev',ind_vine[tr-1][ind_now[0]],ind_vine[tr-1][ind_now[1]])
                            
                            if tr == 1:
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                            else:
                                ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                                parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  
                                if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                    bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                    val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                                else:
                                    bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                    val_to_bin1 = np.digitize(self.Fp_flip[:,tr-1,ind1], bins) -1

                            for bb in range(0,self.n_bin,1):
                                mask1 = tf.where(tf.equal(val_to_bin1,bb))

                                ## Update theta  
                                ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin[:,:,i,bb],self.grid_s.min,self.grid_s.max,self.copulas[tr].cdf[:,:,i,bb],axis=-2)
                                ccdf_data = tf.squeeze(ccdf_data)
                    
                                pd_points, ccdf_points = evaluate_points(points_s_bin[i][bb], batch_size, self.grid_s, cdf1[:,:,i,bb], pd_grid_uv[:,:,i,bb]) 

                                ## Update logf
                                logftr = tf.math.log(pd_points) 
                    
                                self.logf[tf.squeeze(mask1),ind_edge,tr+1] = tf.squeeze(logftr).numpy()

                                ## Update Fp
                                interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data,ccdf_points,self.grid_u.ex)
                                Fp_bin[mask1,i] = tf.reshape(interp_cdf_poi,[tf.shape(interp_cdf_poi)[0],1])

                            Fp_bin1 = tf.squeeze(Fp_bin[:,i])

                            if self.flip_flag[tr][i] == True:
                                self.Fp_flip[:,tr+1,ind_edge] = Fp_bin1
                            else:
                                self.Fp[:,tr+1,ind_edge] = Fp_bin1

            else:

                if (tr==0) | (self.binning == False):
                    
                    for j in range(0,len(self.ind_edge_rel[tr]),1):

                        ind_edge = self.ind_edge_rel[tr][j]

                        cop_p = self.copulas[tr][ind_edge]

                        if self.flip_flag[tr][j] == True:
                            vv = self.points_u[:,:,j]
                            self.Fp_flip[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                        else:
                            vv = np.flip(self.points_u[:,:,j],1)
                            if (self.vine_family == 'c-vine') & (cop_p.family == 'ind') & (j == 0):
                                    vv = self.points_u[:,:,j]
                            self.Fp[:,tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv))
                            
                        
#                         vv = np.flip(vv,1)   Does not change the order in the copulapdf it seems at least for gaussian
                        vv = vv[...,np.newaxis]
                        pd_points = np.squeeze(copulapdf(cop_p,vv))

                        # Update logf
                        logftr = tf.math.log(pd_points)
            
                        self.logf[:,ind_edge,tr+1] = np.squeeze(logftr) #.numpy()

                else:
                    
                    log_f_bin = np.zeros([tf.shape(self.logf)[0],n_eval],self.data_u.dtype)
                    for j in range(0,len(self.ind_edge_rel[tr]),1):
                        ind_edge = self.ind_edge_rel[tr][j]

#                         parent11 = self.r_matrix[n-tr+1,n-1-ind_edge-tr]
#                         ind1 = np.where(self.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(self.theta[:,0,ind1],self.n_bin)
#                         val_to_bin = np.digitize(self.Fp[:,0,ind1], bins) -1
                        ind_now = edges_now[ind_edge]
                        parent11, inx1, inx2 = parent_var(tr,self.ind_vine,ind_now)
                        ind1 = parent11
#                                 print('ind_now',ind_now)
#                                 print('par',parent11)
#                                 print('prev',ind_vine[tr-1][ind_now[0]],ind_vine[tr-1][ind_now[1]])
                        
                        if tr == 1:
                            bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                            val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                            val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                            val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                        else:
                            ind_par_now = self.ind_vine[tr-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(tr-1,self.ind_vine,ind_par_now)  
                            if (self.ind_vine[tr-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(self.theta[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta[:,tr-1,ind1],bins)
                                val_to_bin1 = np.digitize(self.Fp[:,tr-1,ind1], bins) -1
                            else:
                                bins = create_bins(self.theta_flip[:,tr-1,ind1],self.n_bin)
                                val_to_bin = np.digitize(self.theta_flip[:,tr-1,ind1], bins) -1
                                val_to_bin = check_bins(self.theta_flip[:,tr-1,ind1],bins)
                                val_to_bin1 = np.digitize(self.Fp_flip[:,tr-1,ind1], bins) -1
                        
                        for bb in range(0,self.n_bin,1):

                            cop_p = self.copulas[tr][ind_edge][bb]

                            mask = np.where(val_to_bin == bb)
                            mask1 = np.where(val_to_bin1 == bb)
                            data_u_bin = self.data_u[mask[0],:,j]

                            if self.flip_flag[tr][j] == True:

                                vv = self.points_u[:,:,j]
                                vv_bin = vv[mask[0],:]
                                
                                ### CDF FORCE UNIFORM
                                vv_bin_new = vv_bin
                                for zz in range(0,2,1):
                                    vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],vv_bin[:,zz],self.grid_u.ex)
                                vv_bin = vv_bin_new
                                ###
                                self.Fp_flip[mask[0],tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv_bin))
                            else:
                                vv = np.flip(self.points_u[:,:,j],1)
                                if (self.vine_family == 'c-vine') & (cop_p.family == 'ind') & (j == 0):
                                    vv = self.points_u[:,:,j]
        
                                vv_bin = vv[mask[0],:]
                                
                                ### CDF FORCE UNIFORM
                                vv_bin_new = vv_bin
                                for zz in range(0,2,1):
                                    vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],vv_bin[:,zz],self.grid_u.ex)
                                vv_bin = vv_bin_new
                                ###
                                
                                self.Fp[mask[0],tr+1,ind_edge] = np.squeeze(copulaccdf(cop_p,vv_bin))

                            vv_bin = vv_bin[...,np.newaxis]
                            pd_points = np.squeeze(copulapdf(cop_p,vv_bin))

                            ## Update logf
                            logftr = tf.math.log(pd_points) 
                            
                            self.logf[mask[0],ind_edge,tr+1] = logftr
                            
            
        logp = tf.math.reduce_sum(self.logf[:,:,0],1)
        logp_copula = tf.zeros(tf.shape(self.logf[:,0,0]),points.dtype)
        
        for i in tf.range(1,self.n_cop,1,tf.int32):  #self.n_cop
#             unn, ind_un = np.unique(vine.ind_edge_rel[0],return_index=True)  ## Return unique index of ind_edge_rel
            logp = logp + tf.math.reduce_sum(self.logf[:,:,i],1)
            logp_copula = logp_copula + tf.math.reduce_sum(self.logf[:,:,i],1)

        #print('logp',logp)
        logp = tf.cast(logp,tf.float64)
        logp_copula = tf.cast(logp_copula,tf.float64)
        p = tf.exp(logp)
        p_copula = tf.exp(logp_copula)
        return p, p_copula, logf_marginal
        
class vine_obj(object):
    """Vine object.
    """
    def __init__(self, vine_family, families, vine_depth, margin, knots, *args):
        """Create a marginal object.
        Args:
            families: Copula family.
            theta: Correlation factor.
            margin: Margin of the copula.
        """
        self.vine_family = vine_family
        self.families = families
        self.theta1 = []
        self.theta2 = []
        self.rang = None
        self.n_cop = vine_depth
        self.margin = margin
        self.knots = knots
        
        if self.vine_family == 'r-vine':
            self.method = args[0]
            if self.method == 'matrix':
                self.r_matrix = args[1]
                self.E = build_edges(self.r_matrix)
        
        if self.vine_family == 'c-vine':
            self.r_matrix = np.tril(np.tile(np.array(range(self.n_cop,0,-1)),(self.n_cop,1)).T)
            self.E = build_edges(self.r_matrix)
                    
        self.n_cop = None
        self.Mar_G = None
        self.theta = None
        self.Fp = None
        self.logf = None
        self.copulas = None
        
        self.data_u = None
        self.data_s = None
        self.data_x = None
        
        self.points_u = None
        self.points_s = None
        self.points_x = None
        
        self.grid_u = None
        self.grid_s = None
        self.grid_x = None
#         self.err_trace = tf.Variable(initial_value=tf.ones([1],x.dtype),dtype=x.dtype, trainable=False)
#         self.pos_trace = tf.Variable(initial_value=tf.ones([1],x.dtype),dtype=x.dtype, trainable=False)
#         self.iter_err = tf.Variable(1,dtype=tf.int32, trainable=False)
#         self.a = tf.Variable(initial_value=tf.random.uniform(shape=[1], minval=2e-3, maxval=2, dtype=x.dtype), trainable=False)
        
    def fit(self, x, parallel, opt_method):
        
        ## Initialization
        d = x.shape[1]
        self.n_cop = d
        
        ## Make grid
        u_1, ex_u = mk_grid(self.knots,x.dtype)
        trans = Transform(self.n_cop)
        
        ## Grid objects
        self.grid_u = grid_obj(ex_u)
        self.grid_s = grid_obj(trans.forward_u(ex_u))
        
        ## Bivariate normal
        x1_s, x2_s = self.grid_s.axis()
        NORM = biv_norm(x1_s, x2_s)
        
        ## Create Mar_G, theta and Fp
        Mar_G = []
        theta = np.empty([x.shape[0],self.n_cop,self.n_cop],x.dtype)
        for i in range(0,self.n_cop,1):
            ccc = self.margin[i].ker
            mar_p1, mar_s1 = kernel_cdf(ccc, ex_u)
            Mar_G.append([mar_s1, mar_p1])
            theta[:,0,i] = interp1d_np(ccc, mar_s1, mar_p1).numpy()
            del ccc, mar_p1, mar_s1
            
        self.Mar_G = Mar_G
        self.theta = theta
        
        ############### FITTING ####################
        self.copulas = []
        
        for tr in tf.range(0,d-1,1,tf.int32): #d-1
       #     print('Row theta:',tr.numpy())
            
      #      print('theta:',self.theta[:,tr,:])

            # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
            n_cop = d-1-tr
            trans = Transform(n_cop)
     #       print('n_cop in the row:',n_cop.numpy())
            
            data_u = np.empty([theta.shape[0],2,n_cop],x.dtype)  
            
            #### EDGES OF THE VINE
            if self.vine_family == 'r-vine':
                if self.method == 'matrix':   
                    ind_ee = edges_index(self.E,self.r_matrix,tr)
                elif self.method == 'optimal':   
                    if tr == 0:
                        ind_ee, weights = optimal_tree(self.theta[:,tr,:])
                    else:
                        ind_ee, weights = optimal_tree(self.theta[:,tr,:-tr])
            elif self.vine_family == 'c-vine':
                ind_ee = edges_index(self.E,self.r_matrix,tr)
            
            ro = 0
            for ind in ind_ee:
                data_u[:,0,ro] = self.theta[:,tr,ind[0]]
                data_u[:,1,ro] = self.theta[:,tr,ind[1]]
                ro += 1
            
            ## Transform data
            self.data_u = data_u
            self.data_s = trans.forward_u(self.data_u)
            self.data_x = trans.forward_s(self.data_s)
            
            ## Grid on P-Q space
            self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))
            
            #print(data_u)
            
            #self.copulas.append([])
            opt_bw = tf.TensorArray(x.dtype,size=n_cop)
            
            if parallel == True:
                n_cop1 = tf.constant(n_cop,tf.int32)

                grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex}
                data_dict = {'data_s':self.data_s, 'data_x':self.data_x}
                par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,100], 'lr':[0.1, 0.01], 
                                'conv_tol': [0.000001,0.0000001], 'opt_method': opt_method}

                opt = optimization(grid_dict, data_dict, par_dict)
                opt_bw = opt            
            elif parallel == False:
                opt_bw = tf.TensorArray(x.dtype,size=n_cop)
            
                for i in range(0,n_cop,1):
        #            print('col:',i)

                    n_cop1 = tf.constant(1,tf.int32)

                    grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
                    data_dict = {'data_s':self.data_s[:,:,i], 'data_x':self.data_x[:,:,i]}
                    par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,100], 'lr':[0.1, 0.01], 
                                'conv_tol': [0.000001,0.0000001], 'opt_method': opt_method}
    
                    opt = optimization(grid_dict, data_dict, par_dict)
                    opt_bw = opt_bw.write(i,opt)

                opt_bw = opt_bw.stack()         
            
            copula = copula_obj(opt_bw)
            self.copulas.append(copula)
            
            ## UPDATE THETA
            
            grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x}
            data_dict = {'data_s':self.data_s, 'data_x':self.data_x, 'theta':self.theta}
            par_dict = {'copulas': self.copulas[tr], 'n_cop':tf.convert_to_tensor(n_cop), 'batch':tf.constant(2,tf.int32), 'tr':tr}
            
            self.copulas[tr].pd_grid_uv, self.copulas[tr].cdf, self.theta = evaluate_fit(data_dict, grid_dict, par_dict)
            
        return

    def evaluation(self, points):
        
        d = tf.shape(points)[1]
        
        ## Create Fp
        Fp = np.empty([tf.shape(points)[0],self.n_cop,self.n_cop],self.data_u.dtype)
        for i in range(0,self.n_cop,1):
            Fp[:,0,i] = interp1d_np(points[:,i], self.Mar_G[i][0], self.Mar_G[i][1]).numpy()
        self.Fp = Fp
        
        ### Create logf
        en = tf.TensorArray(points.dtype,size=d)
        logf = tf.zeros(tf.shape(points),points.dtype)
        for i in tf.range(0,d,1,tf.int32):
            den1,mden1 = kernel_pdf2(self.margin[i].ker)      
            inter = interp_pdf(points[:,i], mden1, den1) #interp1d_np
            # Product of pdf is the sum of logarithm - Product of pdf margingales evaluated on copula samples
            #logf = logf + tf.math.log(inter)
            logf_tmp = logf[:,0] + tf.math.log(inter)
            logf = update_tensor2D(logf,0,logf_tmp)

            m_diff = mden1[1:] - mden1[:-1]
            m_diff = tf.concat([m_diff, tf.expand_dims(m_diff[-1], 0)], 0)

            # log 2 of the pdf on the grid
            log_pd = tf.py_function(np.log2, [den1], den1.dtype)
            log_pd = replace_inf(log_pd, tf.constant(den1.dtype.min,den1.dtype))
            en = en.write(i,- tf.math.reduce_sum(den1*log_pd*tf.transpose(m_diff),0))  #log_pd
            #vec = tf.linspace(tf.math.reduce_min(vine1.margin[i].ker),tf.math.reduce_max(vine1.margin[i].ker),100)
            #inter_en = interp1d_np(vec, mden1, den1)
            #en = en.write(i,tf.py_function(stats.entropy, [inter_en,vec], vec.dtype))
            del den1,mden1,logf_tmp,m_diff,log_pd
        en = en.stack()
        self.logf = logf.numpy()
        
        for tr in tf.range(0,d-1,1,tf.int32): #d-1
 #           print('Row theta:',tr.numpy())
            
            # TAKE CDF1-CDF2 AND CDF2-CDF3 ...
            n_cop = d-1-tr
            trans = Transform(n_cop)
  #          print('n_cop in the row:',n_cop.numpy())
            
            data_u = np.empty([self.theta.shape[0],2,n_cop],self.data_u.dtype)
            points_u = np.empty([self.Fp.shape[0],2,n_cop],self.data_u.dtype)
            
            #### EDGES OF THE VINE
            if self.vine_family == 'r-vine':
                if self.method == 'matrix':   
                    ind_ee = edges_index(self.E,self.r_matrix,tr)
                elif self.method == 'optimal':   
                    if tr == 0:
                        ind_ee, weights = optimal_tree(self.theta[:,tr,:])
                    else:
                        ind_ee, weights = optimal_tree(self.theta[:,tr,:-tr])
            elif self.vine_family == 'c-vine':
                ind_ee = edges_index(self.E,self.r_matrix,tr)
            
            ro = 0
            for ind in ind_ee:
                data_u[:,0,ro] = self.theta[:,tr,ind[0]]
                data_u[:,1,ro] = self.theta[:,tr,ind[1]]
                points_u[:,0,ro] = self.Fp[:,tr,ind[0]]
                points_u[:,1,ro] = self.Fp[:,tr,ind[1]]
                ro += 1
            
            ## Transform data
            self.data_u = data_u
            self.data_s = trans.forward_u(self.data_u)
            self.data_x = trans.forward_s(self.data_s)
            
            ## Transform points
            self.points_u = points_u
            self.points_s = trans.forward_u(self.points_u)
            self.points_x = trans.forward_s(self.points_s)
            
            ## Grid on P-Q space
            self.grid_x = grid_obj(trans.forward_s(self.grid_s.ex))
            
            #print(data_u)
            grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x}
            data_dict = {'data_s':self.data_s, 'data_x':self.data_x, 'theta':self.theta}
            par_dict = {'copulas': self.copulas[tr], 'n_cop':tf.convert_to_tensor(n_cop), 'batch':tf.constant(2,tf.int32), 'tr':tr}
            
#             pd_grid_uv, cdf1, self.theta = evaluate_fit(data_dict, grid_dict, par_dict)
    
            pd_grid_uv = self.copulas[tr].pd_grid_uv
            cdf1 = self.copulas[tr].cdf
    
            batch_size = tf.constant(2,tf.int32)
            
            for i in range(0,n_cop,1):
                
                ccdf_data = tfp.math.batch_interp_regular_nd_grid(self.data_s[:,:,i],self.grid_s.min,self.grid_s.max,cdf1[:,:,i],axis=-2)
                mar_p1, mar_s1 = kernel_cdf(ccdf_data, self.grid_u.ex)
                
                pd_points, ccdf_points = evaluate_points(self.points_s[:,:,i], batch_size, self.grid_s, cdf1[:,:,i], pd_grid_uv[:,:,i])    
                
                # Update logf
                logftr = tf.math.log(pd_points) 
                logf_tmp = self.logf[:,tr+1] + tf.squeeze(logftr)
                self.logf = update_tensor2D(self.logf,tr+1,logf_tmp)

                # Update Fp
                interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
                self.Fp[:,tr+1,i] = interp_cdf_poi
            
        logp = self.logf[:,0]
        logp_copula = tf.zeros(tf.shape(self.logf[:,0]),points.dtype)
        for i in tf.range(1,d,1,tf.int32):
            #print('loghi',logf[:,i])
            logp = logp + self.logf[:,i]
            logp_copula = logp_copula + self.logf[:,i]
        #print('logp',logp)
        p = tf.exp(logp)
        p_copula = tf.exp(logp_copula)
        return p, p_copula, logp

# File: src/DVC_tensorflow/evalu/.ipynb_checkpoints/cop_eval-checkpoint.py
import tensorflow as tf
from utils.tensor_op import *

############################## COPULA PDF #####################################

@tf.function(experimental_relax_shapes=True)
def eval1(adu11_col1, adu22_1, t2, n_cop):
    # Compute normalization

    I1 = tf.math.reduce_sum(adu22_1*t2,1)
    I2 = tf.math.reduce_sum(adu11_col1*t2,0)
    
    K5 = tf.TensorArray(t2.dtype,size=n_cop) #,element_shape=[tf.shape(t2)[0].eval(),tf.shape(t2)[0].eval()])
    for i in tf.range(0,n_cop,1):
        K1 = tf.tensordot(I1[:,i],I2[:,i],0)
        #print(K1)
        K5 = K5.write(i,K1)
    K5 = K5.stack()
    K5 = tf.transpose(K5, perm=[1,2,0])

    #t2 = t2/K5
    t2 = tf.math.multiply(t2,tf.math.reciprocal(K5))
    
    if tf.reduce_any(tf.math.logical_or(tf.math.is_nan(t2),tf.math.is_inf(t2))) == True:
        t2 = replace_nan_inf(tf.reshape(t2,[-1]))    
        t2 = tf.reshape(t2,[tf.shape(K5)[0],tf.shape(K5)[1],n_cop])             
    return t2

@tf.function(experimental_relax_shapes=True)
def eval_rs_p(adu11, adu22, ker_fit, NORM1, n_cop):
    # Copula normalization for MISE cost function with 100 cycle
    #adu11 = grid_u2.diff1
    #adu22 = grid_u2.diff2
    adu11_col = adu11[...,tf.newaxis]  #Make it a columns vector
    
    #t1 = ker_fit/NORM1     # Projecct on the u-v space
    t1 = tf.math.multiply(ker_fit,tf.math.reciprocal(NORM1))
    
    if tf.math.reduce_any(tf.math.reduce_max(t1) < 1e-6):  ######## THRESHOLD TOP UT ALL COPULA VALUES TO ONE
        t2 = tf.TensorArray(t1.dtype,size=tf.shape(t1)[2])
        for i in range(0,tf.shape(t1)[2],1):
            if tf.math.reduce_max(t1[:,:,i]) < 1e-6:   ######## THRESHOLD TOP UT ALL COPULA VALUES TO ONE
                upd = tf.ones(tf.shape(t1[:,:,i]),t1.dtype)
                t2 = t2.write(i,upd)
            else:
                t2 = t2.write(i,t1[:,:,i])
        t2 = t2.stack()
        t2 = tf.transpose(t2,perm=[1, 2, 0])
        t1 = t2
    
    adu22_1 = adu22[...,tf.newaxis]
    adu22_1 = tf.tile(adu22_1,[1, n_cop])

    adu11_col1 = tf.tile(adu11_col,[1, n_cop])
    adu11_col1 = tf.reshape(adu11_col1,[tf.shape(adu11)[0],1,n_cop])
    
#     t1 = t1*tf.constant(1e-5,t1.dtype)   #### HOUMAN
    
    for i in tf.range(0,50,1,dtype=tf.int32):   #50
        t1 = tf.reshape(eval1(adu11_col1, adu22_1, t1, n_cop),tf.shape(t1))
    
    adu11_col1 = tf.transpose(adu11_col1,perm=[1,0,2])
    II = tf.math.reduce_sum(adu11_col1*tf.math.reduce_sum(adu22_1*t1,1),1)
    t1 = t1/II
    t1 = t1 * NORM1     # Projecct back on the r-s space
    return t1

@tf.function(experimental_relax_shapes=True)
def eval_rs_cop(adu11, adu22, ker_fit, NORM1, n_cop):
    # Copula normalization for MISE cost function with 100 cycle
    #adu11 = grid_u2.diff1
    #adu22 = grid_u2.diff2
    adu11_col = adu11[...,tf.newaxis]  #Make it a columns vector
    
    #t1 = ker_fit/NORM1     # Projecct on the u-v space
    
    t1 = tf.math.multiply(ker_fit,tf.math.reciprocal(NORM1))
    
    if tf.math.reduce_any(tf.math.reduce_max(t1) < 1e-6):  ######## THRESHOLD TOP UT ALL COPULA VALUES TO ONE
        t2 = tf.TensorArray(t1.dtype,size=tf.shape(t1)[2])
        for i in range(0,tf.shape(t1)[2],1):
            if tf.math.reduce_max(t1[:,:,i]) < 1e-6:   ######## THRESHOLD TOP UT ALL COPULA VALUES TO ONE
                upd = tf.ones(tf.shape(t1[:,:,i]),t1.dtype)
                t2 = t2.write(i,upd)
            else:
                t2 = t2.write(i,t1[:,:,i])
        t2 = t2.stack()
        t2 = tf.transpose(t2,perm=[1, 2, 0])
        t1 = t2
    
    adu22_1 = adu22[...,tf.newaxis]
    adu22_1 = tf.tile(adu22_1,[1, n_cop])

    adu11_col1 = tf.tile(adu11_col,[1, n_cop])
    adu11_col1 = tf.reshape(adu11_col1,[tf.shape(adu11)[0],1,n_cop])
    
#     t1 = t1*tf.constant(1e-5,t1.dtype)  ### Houman

    for i in tf.range(0,500,1,dtype=tf.int32): 
        t1 = tf.reshape(eval1(adu11_col1, adu22_1, t1, n_cop),tf.shape(t1))
    
    adu11_col1 = tf.transpose(adu11_col1,perm=[1,0,2])
    II = tf.math.reduce_sum(adu11_col1*tf.math.reduce_sum(adu22_1*t1,1),1)
    t1 = t1/II
    t1 = t1 * NORM1     # Projecct back on the r-s space
    return t1


######################### COPULA CDF #####################################

@tf.function(experimental_relax_shapes=True)
def cdf_grid_fun(pd_grid_uv, ex_u, u1d, u2d, n_cop):
    # Compute the cdf on the grid
    knots = tf.shape(pd_grid_uv)[0]
    u2d = tf.reshape(u2d, [knots,1,1])
    u2d_tile = tf.tile(u2d,[1, knots, n_cop])
    pd_grid_uv_transp = tf.transpose(tf.reshape(pd_grid_uv,[knots, knots, n_cop]),perm=[1, 0, 2])
    integ = tf.math.cumsum(pd_grid_uv_transp*u2d_tile,0)
    norm_p = tf.math.reduce_sum(pd_grid_uv*u2d_tile,0)
    
    #REPLACE ZEROS
    ind_zeros = tf.where(tf.equal(norm_p,0))
    repl_zeros = tf.ones(tf.shape(ind_zeros)[0],u1d.dtype)
    norm_p = tf.tensor_scatter_nd_update(norm_p, ind_zeros, repl_zeros)
    
    cdf1 = tf.transpose(tf.reshape(integ/norm_p,[knots, knots, n_cop]),perm=[1, 0, 2])
    cdf1 = tf.reshape(cdf1, [-1]) 
    cdf1 = check_bound(cdf1,ex_u)
    cdf1 = tf.reshape(cdf1, [tf.shape(u1d)[0],tf.shape(u2d)[0],n_cop])
    return cdf1

# File: src/DVC_tensorflow/evalu/.ipynb_checkpoints/vine_eval-checkpoint.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from utils.prob_op import biv_norm
from optim.local_lik import loclik_batch,loclik_batch_eval
from optim.bandwidth import bandwidth_mul
from evalu.cop_eval import *
from utils.interpolation import nearestInterp2d, interp1d_np
from utils.prob_op import kernel_cdf,kernel_cdf_batch


################# EVALUATE PDF (UV-SPACE), CDF AND THETA ######################

# def evaluate_fit(data_dict, grid_dict, par_dict):
#     grid_u = grid_dict['grid_u']
#     grid_s = grid_dict['grid_s']
#     grid_x = grid_dict['grid_x']
#     adu11 = grid_u.diff1
#     adu22 = grid_u.diff2
    
#     data_s = data_dict['data_s']
#     data_x = data_dict['data_x']
#     theta = data_dict['theta']
    
#     copulas = par_dict['copulas']
#     n_cop1 = par_dict['n_cop']
#     batch_size = par_dict['batch']
#     tr = par_dict['tr']
    
#     ## Bandwidth
#     bw = bandwidth_mul(data_x,2,n_cop1)
# #     bw1 = opt_bw*bw
#     bw1 = np.empty([2,n_cop1],bw.numpy().dtype)
#     for i in range(0,n_cop1,1):
#         if tf.shape(tf.shape(copulas.opt_bw[i])) == 2:
#             bw1[:,i] = tf.squeeze(copulas.opt_bw[i])*bw[:,i]
#         elif tf.shape(tf.shape(copulas.opt_bw[i])) == 0:
#             bw1[:,i] = copulas.opt_bw[i]*tf.transpose(bw[:,i])
#     B = tf.reshape(bw1,[2,n_cop1])
    
#     ## Bivariate normal
#     x1_s, x2_s = grid_s.ax1, grid_s.ax2
#     NORM = biv_norm(x1_s, x2_s)
#     NORM = NORM[...,tf.newaxis]
#     NORM = tf.tile(NORM,[1, 1, n_cop1])
    
#     grid_points = grid_x.ex
# #     grid_x1 = grid_x.ex[...,tf.newaxis]
# #     grid_points = tf.tile(grid_x1,[1, 1, n_cop1])

#     ker_grid_fin = loclik_batch(B, data_x, grid_points, n_cop1, batch_size)  #vine.data_x[:,:,0][...,tf.newaxis]
#     ker_grid_all = tf.transpose(tf.reshape(ker_grid_fin,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop1]),perm=[1, 0, 2])

#     pdf1 = eval_rs_p(adu11, adu22, ker_grid_all, NORM, n_cop1)  #eval_rs_cop

#     pd_grid_uv = pdf1/NORM

#     cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_cop1)

#     # print(theta)

#     for i in tf.range(0,n_cop1,1,tf.int32):
#         ## Update theta  
#         ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s[:,:,i],grid_s.min,grid_s.max,cdf1[:,:,i],axis=-2)
#         #print('ccdf_data',ccdf_data)
#         mar_p1, mar_s1 = kernel_cdf(ccdf_data, grid_u.ex)
#         interp_cdf = interp1d_np(ccdf_data, mar_s1, mar_p1)
#         #print('interp_pdf',interp_cdf)
#     #     theta = update_tensor(theta,interp_cdf,tr+1,i)
#         theta[:,tr+1,i] = interp_cdf
#     return pd_grid_uv, cdf1, theta

# def evaluate_fit(data_dict, grid_dict, par_dict):
#     grid_u = grid_dict['grid_u']
#     grid_s = grid_dict['grid_s']
#     grid_x = grid_dict['grid_x']
#     adu11 = grid_u.diff1
#     adu22 = grid_u.diff2
    
#     data_s = data_dict['data_s']
#     data_x = data_dict['data_x']
#     theta = data_dict['theta']
#     theta_flip = data_dict['theta_flip']
    
#     copulas = par_dict['copulas']
#     n_eval = par_dict['n_eval']
#     batch_size = par_dict['batch']
#     tr = par_dict['tr']
#     ind_edge_rel = par_dict['ind_edge_rel']
#     flip_flag = par_dict['flip_flag']
    
# #     trans = Transform(n_eval)
# #     data_x = trans.forward_s(data_s)
# #     grid_x = trans.forward_s(grid_s.ex)
    
#     ## Bandwidth
#     bw = bandwidth_mul(data_x,2,n_eval)
#     bw1 = np.empty([2,n_eval],bw.numpy().dtype)
#     for i in range(0,n_eval,1):
#         ii = ind_edge_rel[i]
#         if tf.shape(tf.shape(copulas.opt_bw[ii])) == 2:
#             bw1[:,i] = tf.squeeze(copulas.opt_bw[ii])*bw[:,i]
#         elif tf.shape(tf.shape(copulas.opt_bw[ii])) == 0:
#             bw1[:,i] = copulas.opt_bw[ii]*tf.transpose(bw[:,i])
#     B = tf.reshape(bw1,[2,n_eval])
    
#     ## Bivariate normal
#     x1_s, x2_s = grid_s.ax1, grid_s.ax2
#     NORM = biv_norm(x1_s, x2_s)
#     NORM = NORM[...,tf.newaxis]
#     NORM = tf.tile(NORM,[1, 1, n_eval])

#     ker_grid_fin = loclik_batch(B, data_x, grid_x, n_eval, batch_size)  #vine.data_x[:,:,0][...,tf.newaxis]
#     ker_grid_all = tf.transpose(tf.reshape(ker_grid_fin,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_eval]),perm=[1, 0, 2])
    
#     pdf1 = eval_rs_p(adu11, adu22, ker_grid_all, NORM, n_eval)  #eval_rs_cop

#     pd_grid_uv = pdf1/NORM
    
# #     cdf1 = tf.transpose(cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_eval),perm=[1, 0, 2])
#     cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_eval)

#     for i in range(0,n_eval,1):
#         ## Update theta
#         ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s[:,:,i],grid_s.min,grid_s.max,cdf1[:,:,i],axis=-2)

#         mar_p1, mar_s1 = kernel_cdf(ccdf_data, grid_u.ex)
#         interp_cdf = interp1d_np(ccdf_data, mar_s1, mar_p1)

#         if flip_flag[i] == False:
#             theta[:,tr+1,ind_edge_rel[i]] = interp_cdf
#         else:
#             theta_flip[:,tr+1,ind_edge_rel[i]] = interp_cdf
#     return pd_grid_uv, cdf1, theta, theta_flip

def evaluate_fit(data_dict, grid_dict, par_dict):
    grid_u = grid_dict['grid_u']
    grid_s = grid_dict['grid_s']
    grid_x = grid_dict['grid_x']
    adu11,adu22 = grid_u.diff()
    
    data_s = data_dict['data_s']
    data_x = data_dict['data_x']
    theta = data_dict['theta']
    theta_flip = data_dict['theta_flip']
    
    copulas = par_dict['copulas']
    n_eval = par_dict['n_eval']
    batch_size = par_dict['batch']
    batch_size_cdf = par_dict['batch_cdf']
    tr = par_dict['tr']
    ind_edge_rel = par_dict['ind_edge_rel']
    flip_flag = par_dict['flip_flag']

    bw1 = np.zeros([2,n_eval],data_s.dtype)
    for i in range(0,n_eval,1):
        ii = ind_edge_rel[i]
        bw1[:,i] = tf.convert_to_tensor(copulas.opt_bw[:,ii])
    B = tf.reshape(bw1,[2,n_eval])
    
    ## Bivariate normal
    x1_s, x2_s = grid_s.ax1, grid_s.ax2
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM[...,tf.newaxis]
    NORM = tf.tile(NORM,[1, 1, n_eval])
    
#     tf_dtype = B.dtype
#     B = tf.cast(B,tf.float64)
#     data_x = tf.cast(data_x,tf.float64)
#     grid_x = tf.cast(grid_x,tf.float64)
    
    ker_grid_fin = loclik_batch_eval(B, data_x, grid_x, n_eval, batch_size)
    
#     ker_grid_fin = loclik_batch(B, data_x, grid_x, n_eval, batch_size)  #vine.data_x[:,:,0][...,tf.newaxis]
#     ker_grid_fin = tf.cast(ker_grid_fin,tf_dtype)
    
    ker_grid_all = tf.transpose(tf.reshape(ker_grid_fin,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_eval]),perm=[1, 0, 2])
    
    ker_grid_all = ker_grid_all + 1e-30*NORM #1e-30 
    
    pdf1 = eval_rs_cop(adu11, adu22, ker_grid_all, NORM, n_eval)  #eval_rs_p

    pd_grid_uv = pdf1/NORM
    
#     cdf1 = tf.transpose(cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_eval),perm=[1, 0, 2])
    cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_eval)

    for i in range(0,n_eval,1):
        ## Update theta
        ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s[:,:,i],grid_s.min,grid_s.max,cdf1[:,:,i],axis=-2)

#         mar_p1, mar_s1 = kernel_cdf(ccdf_data, grid_u.ex)
#         interp_cdf = interp1d_np(ccdf_data, mar_s1, mar_p1)
        
#         interp_cdf, mar_s1, mar_p1 = kernel_cdf_batch(ccdf_data,ccdf_data,grid_u.ex, batch_size_cdf)
        interp_cdf, mar_s1, mar_p1 = kernel_cdf(ccdf_data,ccdf_data,grid_u.ex)

        if flip_flag[i] == False:
            theta[:,tr+1,ind_edge_rel[i]] = interp_cdf
        else:
            theta_flip[:,tr+1,ind_edge_rel[i]] = interp_cdf
    return pd_grid_uv, cdf1, theta, theta_flip

################# EVALUATE PDF AND CCDF ON THE POINTS

# def evaluate_points(points_u, grid_u, points_s, batch_size, grid_s, cdf1, pd_grid_uv):
# #     pd_points = tf.zeros(1,points_s.dtype)
# #     ccdf_points = tf.zeros(1,points_s.dtype)
#     pd_points = tf.TensorArray(points_s.dtype, size = batch_size)
#     ccdf_points = tf.TensorArray(points_s.dtype, size = batch_size)
#     batch_len = tf.shape(points_s)[0]/batch_size
#     batch_len = tf.cast(batch_len,tf.int32)
    
#     s_ax1 = grid_s.ax1
#     s_ax2 = grid_s.ax2 
#     u_ax1 = grid_u.ax1
#     u_ax2 = grid_u.ax2 
    
#     #print(points_s[:,:,i])

#     for j in tf.range(0,batch_size,1):
#         points_batch = points_s[batch_len*j:batch_len*(j+1),:]
#         points_batch_u = points_u[batch_len*j:batch_len*(j+1),:]
        
#         if tf.math.equal(j,batch_size-1):
#             points_batch = points_s[batch_len*j:,:]
#             points_batch_u = points_u[batch_len*j:,:]
#         pd_points1 = nearestInterp2d(points_batch_u, u_ax1, u_ax2, pd_grid_uv)  #pd_grid_uv[:,:,i]
# #         pd_points1 = tfp.math.batch_interp_regular_nd_grid(points_batch_u,grid_u.min,grid_u.max,pd_grid_uv,axis=-2)
        
#         ccdf_points1 = tfp.math.batch_interp_regular_nd_grid(points_batch,grid_s.min,grid_s.max,cdf1,axis=-2)
#         pd_points = pd_points.write(j, pd_points1)
#         ccdf_points = ccdf_points.write(j, ccdf_points1)
# #         pd_points = tf.concat([pd_points,pd_points1],0)
# #         ccdf_points = tf.concat([ccdf_points,ccdf_points1],0)
    
#     pd_points = pd_points.stack()
#     ccdf_points = ccdf_points.stack()
# #     pd_points = pd_points[1:]
# #     ccdf_points = ccdf_points[1:]
#     pd_points = tf.reshape(pd_points,[-1])
#     ccdf_points = tf.reshape(ccdf_points,[-1])
#     return pd_points, ccdf_points

def evaluate_points(points_s, batch_size, grid_s, cdf1, pd_grid_uv):
#     pd_points = tf.zeros(1,points_s.dtype)
#     ccdf_points = tf.zeros(1,points_s.dtype)
    pd_points = tf.TensorArray(points_s.dtype, size = batch_size)
    ccdf_points = tf.TensorArray(points_s.dtype, size = batch_size)
    batch_len = tf.shape(points_s)[0]/batch_size
    batch_len = tf.cast(batch_len,tf.int32)
    
    s_ax1 = grid_s.ax1
    s_ax2 = grid_s.ax2  
    
    #print(points_s[:,:,i])

    for j in tf.range(0,batch_size,1):
        points_batch = points_s[batch_len*j:batch_len*(j+1),:]
        if tf.math.equal(j,batch_size-1):
            points_batch = points_s[batch_len*j:,:]
        pd_points1 = nearestInterp2d(points_batch, s_ax1, s_ax2, pd_grid_uv)  #pd_grid_uv[:,:,i]
#         pd_points1 = tfp.math.batch_interp_regular_nd_grid(points_batch,grid_s.min,grid_s.max,pd_grid_uv,axis=-2)
        
        ccdf_points1 = tfp.math.batch_interp_regular_nd_grid(points_batch,grid_s.min,grid_s.max,cdf1,axis=-2)
        pd_points = pd_points.write(j, pd_points1)
        ccdf_points = ccdf_points.write(j, ccdf_points1)
#         pd_points = tf.concat([pd_points,pd_points1],0)
#         ccdf_points = tf.concat([ccdf_points,ccdf_points1],0)
    
    pd_points = pd_points.stack()
    ccdf_points = ccdf_points.stack()
#     pd_points = pd_points[1:]
#     ccdf_points = ccdf_points[1:]
    pd_points = tf.reshape(pd_points,[-1])
    ccdf_points = tf.reshape(ccdf_points,[-1])
    return pd_points, ccdf_points


#################### EVALUATE BINNING ###########################

# def evaluate_fit_bin(data_dict, grid_dict, par_dict):
#     grid_u = grid_dict['grid_u']
#     grid_s = grid_dict['grid_s']
#     grid_x = grid_dict['grid_x']
#     adu11 = grid_u.diff1
#     adu22 = grid_u.diff2
    
#     data_s = data_dict['data_s']
#     data_x = data_dict['data_x']
#     bb = data_dict['bin']
# #     theta = data_dict['theta']
    
#     copulas = par_dict['copulas']
#     n_cop1 = par_dict['n_cop']
#     batch_size = par_dict['batch']
#     tr = par_dict['tr']
    
#     ## Bandwidth
#     bw = bandwidth_mul(data_x,2,n_cop1)

#     bw1 = np.empty([2,n_cop1],bw.numpy().dtype)
#     for i in range(0,n_cop1,1):
#         if tf.shape(tf.shape(copulas.opt_bw)) == 3:
#             bw1[:,i] = copulas.opt_bw[:,i,bb]*tf.transpose(bw[:,i])
#         else:
#             if tf.shape(tf.shape(copulas.opt_bw[i])) == 2:
#                 bw1[:,i] = tf.squeeze(copulas.opt_bw[i])*bw[:,i]
#             elif tf.shape(tf.shape(copulas.opt_bw[i])) == 0:
#                 bw1[:,i] = copulas.opt_bw[i]*tf.transpose(bw[:,i])
#     B = tf.reshape(bw1,[2,n_cop1])
    
# #     print('B',B)
    
#     ## Bivariate normal
#     x1_s, x2_s = grid_s.ax1, grid_s.ax2
#     NORM = biv_norm(x1_s, x2_s)
#     NORM = NORM[...,tf.newaxis]
#     NORM = tf.tile(NORM,[1, 1, n_cop1])
    
#     grid_points = grid_x.ex
# #     grid_x1 = grid_x.ex[...,tf.newaxis]
# #     grid_points = tf.tile(grid_x1,[1, 1, n_cop1])
# #     print('grid_points',data_x)
# #     print('grid_points',grid_points)
#     ker_grid_fin = loclik_batch(B, data_x, grid_points, n_cop1, batch_size)  #vine.data_x[:,:,0][...,tf.newaxis]
#     ker_grid_all = tf.transpose(tf.reshape(ker_grid_fin,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop1]),perm=[1, 0, 2])
    
# #     print('ker_grid_all',ker_grid_all[:,:,0])
    
#     pdf1 = eval_rs_p(adu11, adu22, ker_grid_all, NORM, n_cop1)  #eval_rs_cop

#     pd_grid_uv = pdf1/NORM

# #     cdf1 = tf.transpose(cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_cop1),perm=[1, 0, 2])
#     cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_cop1)
    
# #     print('pdf',pd_grid_uv[:,:,0])
# #     print('cdf',cdf1[:,:,0])
#     # print(theta)
#     return pd_grid_uv, cdf1

# def evaluate_fit_bin(data_dict, grid_dict, par_dict):
#     grid_u = grid_dict['grid_u']
#     grid_s = grid_dict['grid_s']
#     grid_x = grid_dict['grid_x']
#     adu11 = grid_u.diff1
#     adu22 = grid_u.diff2
    
#     data_s = data_dict['data_s']
#     data_x = data_dict['data_x']
#     bb = data_dict['bin']

#     data_s = tf.convert_to_tensor(data_s)
#     data_x = tf.convert_to_tensor(data_x)
    
#     copulas = par_dict['copulas']
#     n_cop1 = par_dict['n_cop']
#     batch_size = par_dict['batch']
#     tr = par_dict['tr']
#     ind_edge_rel = par_dict['ind_edge_rel']
    
#     ## Bandwidth
#     bw = bandwidth_mul(data_x,2,n_cop1)

#     bw1 = np.empty([2,n_cop1],bw.numpy().dtype)
#     for i in range(0,n_cop1,1):
#         ii = ind_edge_rel[i]
#         if tf.shape(tf.shape(copulas.opt_bw)) == 3:
#             bw1[:,i] = copulas.opt_bw[:,ii,bb]*tf.transpose(bw[:,i])
#         else:
#             if tf.shape(tf.shape(copulas.opt_bw[ii])) == 2:
#                 bw1[:,i] = tf.squeeze(copulas.opt_bw[ii])*bw[:,i]
#             elif tf.shape(tf.shape(copulas.opt_bw[ii])) == 0:
#                 bw1[:,i] = copulas.opt_bw[ii]*tf.transpose(bw[:,i])
#     B = tf.reshape(bw1,[2,n_cop1])

#     ## Bivariate normal
#     x1_s, x2_s = grid_s.ax1, grid_s.ax2
#     NORM = biv_norm(x1_s, x2_s)
#     NORM = NORM[...,tf.newaxis]
#     NORM = tf.tile(NORM,[1, 1, n_cop1])
    
#     ker_grid_fin = loclik_batch(B, data_x, grid_x, n_cop1, batch_size)  #vine.data_x[:,:,0][...,tf.newaxis]
#     ker_grid_all = tf.transpose(tf.reshape(ker_grid_fin,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop1]),perm=[1, 0, 2])
    
#     pdf1 = eval_rs_p(adu11, adu22, ker_grid_all, NORM, n_cop1)  #eval_rs_cop

#     pd_grid_uv = pdf1/NORM

#     cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_cop1)

#     return pd_grid_uv, cdf1

def evaluate_fit_bin(data_dict, grid_dict, par_dict):
    grid_u = grid_dict['grid_u']
    grid_s = grid_dict['grid_s']
    grid_x = grid_dict['grid_x']
    adu11 = grid_u.diff1
    adu22 = grid_u.diff2
    
    data_s = data_dict['data_s']
    data_x = data_dict['data_x']
#     bb = data_dict['bin']
    
    bw = par_dict['bw']
    n_cop1 = tf.convert_to_tensor(par_dict['n_cop'])
    batch_size = tf.convert_to_tensor(par_dict['batch'])
    tr = par_dict['tr']
    ind_edge_rel = par_dict['ind_edge_rel']
    
    ## Bandwidth

    bw1 = np.empty([2,n_cop1],data_s.dtype)
    for i in range(0,n_cop1,1):
        ii = ind_edge_rel[i]
        bw1[:,i] = bw[:,ii]
    
    B = tf.reshape(tf.convert_to_tensor(bw1),[2,n_cop1])

    ## Bivariate normal
    x1_s, x2_s = grid_s.ax1, grid_s.ax2
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM[...,tf.newaxis]
    NORM = tf.tile(NORM,[1, 1, n_cop1])
    
    data_s = tf.convert_to_tensor(data_s)
    data_x = tf.convert_to_tensor(data_x)
    
    ker_grid_fin = loclik_batch_eval(B, data_x, grid_x, n_cop1, batch_size)
    
#     ker_grid_fin = loclik_batch(B, data_x, grid_x, n_cop1, batch_size)  #vine.data_x[:,:,0][...,tf.newaxis]
    ker_grid_all = tf.transpose(tf.reshape(ker_grid_fin,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop1]),perm=[1, 0, 2])
    
    ker_grid_all = ker_grid_all + 1e-10*NORM
    
    pdf1 = eval_rs_cop(adu11, adu22, ker_grid_all, NORM, n_cop1)  #eval_rs_p

    pd_grid_uv = pdf1/NORM

    cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_cop1)

    return pd_grid_uv, cdf1

# File: src/DVC_tensorflow/evalu/cop_eval.py
import tensorflow as tf
from utils.tensor_op import *

############################## COPULA PDF #####################################

@tf.function(experimental_relax_shapes=True)
def eval1(adu11_col1, adu22_1, t2, n_cop):
    # Compute normalization

    I1 = tf.math.reduce_sum(adu22_1*t2,1)
    I2 = tf.math.reduce_sum(adu11_col1*t2,0)
    
    K5 = tf.TensorArray(t2.dtype,size=n_cop) #,element_shape=[tf.shape(t2)[0].eval(),tf.shape(t2)[0].eval()])
    for i in tf.range(0,n_cop,1):
        K1 = tf.tensordot(I1[:,i],I2[:,i],0)
        #print(K1)
        K5 = K5.write(i,K1)
    K5 = K5.stack()
    K5 = tf.transpose(K5, perm=[1,2,0])

    #t2 = t2/K5
    t2 = tf.math.multiply(t2,tf.math.reciprocal(K5))
    
    if tf.reduce_any(tf.math.logical_or(tf.math.is_nan(t2),tf.math.is_inf(t2))) == True:
        t2 = replace_nan_inf(tf.reshape(t2,[-1]))    
        t2 = tf.reshape(t2,[tf.shape(K5)[0],tf.shape(K5)[1],n_cop])             
    return t2

@tf.function(experimental_relax_shapes=True)
def eval_rs_p(adu11, adu22, ker_fit, NORM1, n_cop):
    # Copula normalization for MISE cost function with 100 cycle
    #adu11 = grid_u2.diff1
    #adu22 = grid_u2.diff2
    adu11_col = adu11[...,tf.newaxis]  #Make it a columns vector
    
    #t1 = ker_fit/NORM1     # Projecct on the u-v space
    t1 = tf.math.multiply(ker_fit,tf.math.reciprocal(NORM1))
    
    if tf.math.reduce_any(tf.math.reduce_max(t1) < 1e-6):  ######## THRESHOLD TOP UT ALL COPULA VALUES TO ONE
        t2 = tf.TensorArray(t1.dtype,size=tf.shape(t1)[2])
        for i in range(0,tf.shape(t1)[2],1):
            if tf.math.reduce_max(t1[:,:,i]) < 1e-6:   ######## THRESHOLD TOP UT ALL COPULA VALUES TO ONE
                upd = tf.ones(tf.shape(t1[:,:,i]),t1.dtype)
                t2 = t2.write(i,upd)
            else:
                t2 = t2.write(i,t1[:,:,i])
        t2 = t2.stack()
        t2 = tf.transpose(t2,perm=[1, 2, 0])
        t1 = t2
    
    adu22_1 = adu22[...,tf.newaxis]
    adu22_1 = tf.tile(adu22_1,[1, n_cop])

    adu11_col1 = tf.tile(adu11_col,[1, n_cop])
    adu11_col1 = tf.reshape(adu11_col1,[tf.shape(adu11)[0],1,n_cop])
    
#     t1 = t1*tf.constant(1e-5,t1.dtype)   #### HOUMAN
    
    for i in tf.range(0,50,1,dtype=tf.int32):   #50
        t1 = tf.reshape(eval1(adu11_col1, adu22_1, t1, n_cop),tf.shape(t1))
    
    adu11_col1 = tf.transpose(adu11_col1,perm=[1,0,2])
    II = tf.math.reduce_sum(adu11_col1*tf.math.reduce_sum(adu22_1*t1,1),1)
    t1 = t1/II
    t1 = t1 * NORM1     # Projecct back on the r-s space
    return t1

@tf.function(experimental_relax_shapes=True)
def eval_rs_cop(adu11, adu22, ker_fit, NORM1, n_cop):
    # Copula normalization for MISE cost function with 100 cycle
    #adu11 = grid_u2.diff1
    #adu22 = grid_u2.diff2
    adu11_col = adu11[...,tf.newaxis]  #Make it a columns vector
    
    #t1 = ker_fit/NORM1     # Projecct on the u-v space
    
    t1 = tf.math.multiply(ker_fit,tf.math.reciprocal(NORM1))
    
    if tf.math.reduce_any(tf.math.reduce_max(t1) < 1e-6):  ######## THRESHOLD TOP UT ALL COPULA VALUES TO ONE
        t2 = tf.TensorArray(t1.dtype,size=tf.shape(t1)[2])
        for i in range(0,tf.shape(t1)[2],1):
            if tf.math.reduce_max(t1[:,:,i]) < 1e-6:   ######## THRESHOLD TOP UT ALL COPULA VALUES TO ONE
                upd = tf.ones(tf.shape(t1[:,:,i]),t1.dtype)
                t2 = t2.write(i,upd)
            else:
                t2 = t2.write(i,t1[:,:,i])
        t2 = t2.stack()
        t2 = tf.transpose(t2,perm=[1, 2, 0])
        t1 = t2
    
    adu22_1 = adu22[...,tf.newaxis]
    adu22_1 = tf.tile(adu22_1,[1, n_cop])

    adu11_col1 = tf.tile(adu11_col,[1, n_cop])
    adu11_col1 = tf.reshape(adu11_col1,[tf.shape(adu11)[0],1,n_cop])
    
#     t1 = t1*tf.constant(1e-5,t1.dtype)  ### Houman

    for i in tf.range(0,500,1,dtype=tf.int32): 
        t1 = tf.reshape(eval1(adu11_col1, adu22_1, t1, n_cop),tf.shape(t1))
    
    adu11_col1 = tf.transpose(adu11_col1,perm=[1,0,2])
    II = tf.math.reduce_sum(adu11_col1*tf.math.reduce_sum(adu22_1*t1,1),1)
    t1 = t1/II
    t1 = t1 * NORM1     # Projecct back on the r-s space
    return t1


######################### COPULA CDF #####################################

@tf.function(experimental_relax_shapes=True)
def cdf_grid_fun(pd_grid_uv, ex_u, u1d, u2d, n_cop):
    # Compute the cdf on the grid
    knots = tf.shape(pd_grid_uv)[0]
    u2d = tf.reshape(u2d, [knots,1,1])
    u2d_tile = tf.tile(u2d,[1, knots, n_cop])
    pd_grid_uv_transp = tf.transpose(tf.reshape(pd_grid_uv,[knots, knots, n_cop]),perm=[1, 0, 2])
    integ = tf.math.cumsum(pd_grid_uv_transp*u2d_tile,0)
    norm_p = tf.math.reduce_sum(pd_grid_uv*u2d_tile,0)
    
    #REPLACE ZEROS
    ind_zeros = tf.where(tf.equal(norm_p,0))
    repl_zeros = tf.ones(tf.shape(ind_zeros)[0],u1d.dtype)
    norm_p = tf.tensor_scatter_nd_update(norm_p, ind_zeros, repl_zeros)
    
    cdf1 = tf.transpose(tf.reshape(integ/norm_p,[knots, knots, n_cop]),perm=[1, 0, 2])
    cdf1 = tf.reshape(cdf1, [-1]) 
    cdf1 = check_bound(cdf1,ex_u)
    cdf1 = tf.reshape(cdf1, [tf.shape(u1d)[0],tf.shape(u2d)[0],n_cop])
    return cdf1

# File: src/DVC_tensorflow/evalu/vine_eval.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from utils.prob_op import biv_norm
from optim.local_lik import loclik_batch,loclik_batch_eval
from optim.bandwidth import bandwidth_mul
from evalu.cop_eval import *
from utils.interpolation import nearestInterp2d, interp1d_np
from utils.prob_op import kernel_cdf,kernel_cdf_batch


################# EVALUATE PDF (UV-SPACE), CDF AND THETA ######################

def evaluate_fit(data_dict, grid_dict, par_dict):
    grid_u = grid_dict['grid_u']
    grid_s = grid_dict['grid_s']
    grid_x = grid_dict['grid_x']
    adu11,adu22 = grid_u.diff()
    
    data_s = data_dict['data_s']
    data_x = data_dict['data_x']
    theta = data_dict['theta']
    theta_flip = data_dict['theta_flip']
    
    copulas = par_dict['copulas']
    n_eval = par_dict['n_eval']
    batch_size = par_dict['batch']
    batch_size_cdf = par_dict['batch_cdf']
    tr = par_dict['tr']
    ind_edge_rel = par_dict['ind_edge_rel']
    flip_flag = par_dict['flip_flag']

    bw1 = np.zeros([2,n_eval],data_s.dtype)
    for i in range(0,n_eval,1):
        ii = ind_edge_rel[i]
        bw1[:,i] = tf.convert_to_tensor(copulas.opt_bw[:,ii])
    B = tf.reshape(bw1,[2,n_eval])
    
    ## Bivariate normal
    x1_s, x2_s = grid_s.ax1, grid_s.ax2
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM[...,tf.newaxis]
    NORM = tf.tile(NORM,[1, 1, n_eval])
    
    ker_grid_fin = loclik_batch_eval(B, data_x, grid_x, n_eval, batch_size)
    
    ker_grid_all = tf.transpose(tf.reshape(ker_grid_fin,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_eval]),perm=[1, 0, 2])
    
    ### The following was added to avoid points with 0 probability but to have it to be very low. Otherwise the log goes to inf
    ker_grid_all = ker_grid_all + 1e-15*NORM # Before it was 1e-30, do not know which one is better 
    
    pdf1 = eval_rs_cop(adu11, adu22, ker_grid_all, NORM, n_eval)  #eval_rs_p

    pd_grid_uv = pdf1/NORM
    
    cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_eval)

    for i in range(0,n_eval,1):
        ## Update theta
        ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s[:,:,i],grid_s.min,grid_s.max,cdf1[:,:,i],axis=-2)
        interp_cdf, mar_s1, mar_p1 = kernel_cdf(ccdf_data,ccdf_data,grid_u.ex)

        if flip_flag[i] == False:
            theta[:,tr+1,ind_edge_rel[i]] = interp_cdf
        else:
            theta_flip[:,tr+1,ind_edge_rel[i]] = interp_cdf
    return pd_grid_uv, cdf1, theta, theta_flip

################# EVALUATE PDF AND CCDF ON THE POINTS

def evaluate_points(points_s, batch_size, grid_s, cdf1, pd_grid_uv):
    pd_points = tf.TensorArray(points_s.dtype, size = batch_size)
    ccdf_points = tf.TensorArray(points_s.dtype, size = batch_size)
    batch_len = tf.shape(points_s)[0]/batch_size
    batch_len = tf.cast(batch_len,tf.int32)
    
    s_ax1 = grid_s.ax1
    s_ax2 = grid_s.ax2  

    for j in tf.range(0,batch_size,1):
        points_batch = points_s[batch_len*j:batch_len*(j+1),:]
        if tf.math.equal(j,batch_size-1):
            points_batch = points_s[batch_len*j:,:]
        pd_points1 = nearestInterp2d(points_batch, s_ax1, s_ax2, pd_grid_uv) 
        
        ccdf_points1 = tfp.math.batch_interp_regular_nd_grid(points_batch,grid_s.min,grid_s.max,cdf1,axis=-2)
        pd_points = pd_points.write(j, pd_points1)
        ccdf_points = ccdf_points.write(j, ccdf_points1)
    
    pd_points = pd_points.stack()
    ccdf_points = ccdf_points.stack()
    pd_points = tf.reshape(pd_points,[-1])
    ccdf_points = tf.reshape(ccdf_points,[-1])
    return pd_points, ccdf_points


#################### EVALUATE BINNING ###########################

def evaluate_fit_bin(data_dict, grid_dict, par_dict):
    grid_u = grid_dict['grid_u']
    grid_s = grid_dict['grid_s']
    grid_x = grid_dict['grid_x']
    adu11 = grid_u.diff1
    adu22 = grid_u.diff2
    
    data_s = data_dict['data_s']
    data_x = data_dict['data_x']
#     bb = data_dict['bin']
    
    bw = par_dict['bw']
    n_cop1 = tf.convert_to_tensor(par_dict['n_cop'])
    batch_size = tf.convert_to_tensor(par_dict['batch'])
    tr = par_dict['tr']
    ind_edge_rel = par_dict['ind_edge_rel']
    
    ## Bandwidth

    bw1 = np.empty([2,n_cop1],data_s.dtype)
    for i in range(0,n_cop1,1):
        ii = ind_edge_rel[i]
        bw1[:,i] = bw[:,ii]
    
    B = tf.reshape(tf.convert_to_tensor(bw1),[2,n_cop1])

    ## Bivariate normal
    x1_s, x2_s = grid_s.ax1, grid_s.ax2
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM[...,tf.newaxis]
    NORM = tf.tile(NORM,[1, 1, n_cop1])
    
    data_s = tf.convert_to_tensor(data_s)
    data_x = tf.convert_to_tensor(data_x)
    
    ker_grid_fin = loclik_batch_eval(B, data_x, grid_x, n_cop1, batch_size)
    
#     ker_grid_fin = loclik_batch(B, data_x, grid_x, n_cop1, batch_size)  #vine.data_x[:,:,0][...,tf.newaxis]
    ker_grid_all = tf.transpose(tf.reshape(ker_grid_fin,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop1]),perm=[1, 0, 2])
    
    ker_grid_all = ker_grid_all + 1e-10*NORM
    
    pdf1 = eval_rs_cop(adu11, adu22, ker_grid_all, NORM, n_cop1)  #eval_rs_p

    pd_grid_uv = pdf1/NORM

    cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_cop1)

    return pd_grid_uv, cdf1

# File: src/DVC_tensorflow/experiments/25gaussians/vine_25gaussians_sample_check.py
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#device = gpu_devices[0]
#tf.config.experimental.set_memory_growth(device, True)
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys

sys.path.append("/Users/safaai/Library/CloudStorage/OneDrive-CompTech/Houman_Work/NPC")

from classes.objects import *
from vine_tree.tree_op import *

from scipy import stats
import pickle

###########

from param.generate_rvine import *
from param.margin_fit import *
from param.margin_op import *
from param.copula_fit import *
from param.cond_copula import *
from pre_proc.preparation import prep_cop
from pred.prediction import*
from sampling.vine_sample import *
from info.info_estimation import vine_entropy

x = np.array([(i,j) for i in range(-4, 5, 2) for j in range(-4, 5, 2)] * 400)

x = x + 0.1 * np.random.randn(*x.shape)

# plt.scatter(x[:,0], x[:,1], s=2.0)
# plt.xlim((-10, 10))
# plt.ylim((-10, 10))
# plt.show()

np.random.shuffle(x)
# plt.scatter(x[:,0], x[:,1], s=2.0)
# plt.xlim((-10, 10))
# plt.ylim((-10, 10))
# plt.show()

print(x.shape)

#### Generate random matrix

cases = 1000        ### Number of samples
vine_type = 'c-vine' # or 'd-vine' or 'c-vine'
method = 'matrix'  # or 'r_matrix'  only with r-vine
binning = False
n_bin = 3
dim = 2                # Dimension of the vine for random r-vine or c-vine or d-vine


if vine_type == 'r-vine':
    
    if method == 'matrix':
        
        ######### REGULAR MATRIX
        r_matrix = np.array([[2, 0, 0, 0, 0],
                             [5, 3, 0, 0, 0],
                             [4, 5, 1, 0, 0],
                             [1, 4, 5, 4, 0],
                             [3, 1, 4, 5, 5]])
        
#         r_matrix = np.array([[3, 0, 0, 0],
#                              [1, 4, 0, 0],
#                              [2, 1, 2, 0],
#                              [4, 2, 1, 1]])
        
#         r_matrix = np.array([[3, 0, 0],
#                              [2, 2, 0],
#                              [1, 1, 1]])

        print(r_matrix)
        
    elif method == 'random':
        
        ##### RANDOM R-MATRIX
        r_matrix, ind_vine, nodes, E = random_r_matrix_gen(dim)
        print(r_matrix)

    E, ind_vine, nodes, matrix_edges = prepare_regular(r_matrix)
    print('matrix_edges',matrix_edges)
    
    ## DEFINE MARGINS
#     margin_fam1 = ['norm','gamma','norm','gamma','norm']
#     theta_fam1 = [[0,1],[2,4],[0,1],[2,4],[0,1]]
    margin_fam1 = ['norm','norm','norm','norm','norm']
    theta_fam1 = [[0,1],[0,1],[0,1],[0,1],[0,1]]
    is_cont1 = [True,True,True,True,True]

    margin_vine = []
    for i in range(0,len(margin_fam1),1):
        mar_p = margin_obj(margin_fam1[i], theta_fam1[i], is_cont1[i])
        margin_vine.append(mar_p)

    for i in range(0,len(margin_fam1),1):
        print(margin_vine[i].dist, end =' ')
        print(margin_vine[i].theta, end =' ')
    
    ######################## DEFINE COPULAS  ###################
    if not binning:
        
#         margin_cop1 = [['gaussian','gaussian','gaussian','gaussian'],
#                        ['gaussian','gaussian','gaussian'],
#                        ['gaussian','gaussian'],
#                        ['gaussian']]
        
#         theta_cop1 = [[0.3, 0.5, 0.7, 0.8],
#                       [0.5, 0.8, 0.4],
#                       [0.5, 0.3],
#                       [0.9]]

#         margin_cop1 = [['gaussian','gaussian','gaussian','gaussian','gaussian'],
#                        ['gaussian','gaussian','gaussian','gaussian'],
#                        ['gaussian','gaussian','gaussian'],
#                        ['gaussian','gaussian'],
#                        ['gaussian']]
        
#         theta_cop1 = [[0.3, 0.5, 0.7, 0.8, -0.8],
#                       [0.5, 0.8, 0.4, -0.2],
#                       [0.5, 0.3, -0.7],
#                       [0.9, 0.6],
#                       [0.5]]
        
        margin_cop1 = [['gaussian','student','clayton','gaussian'],
                       ['student','clayton','gaussian'],
                       ['student','gaussian'],
                       ['clayton']]

        theta_cop1 = [[0.3, [0,0.2], 0.7,  -0.8],
                      [[-0.8,0.2], 4.5,  -0.2],
                      [[0,0.5],  -0.7],
                      [0.9]]
        
#         margin_cop1 = [['gaussian','gaussian','gaussian'],
#                        ['gaussian','gaussian'],
#                        ['gaussian']]

#         theta_cop1 = [[0.7, 0.8, 0.9],
#                       [0.6, 0.5],
#                       [0.7]]

        # margin_cop1 = [['clayton','clayton','clayton'],
        #                ['clayton','clayton'],
        #                ['clayton']]

        # theta_cop1 = [[3.7, 4.8, 5.9],
        #               [6.6, 2.5],
        #               [5.7]]
    else:
        
        margin_cop1 = [['gaussian','gaussian','gaussian','gaussian'],
                       [['gaussian','gaussian','gaussian'],['gaussian','gaussian','gaussian'],['gaussian','gaussian','gaussian']],
                      [['gaussian','gaussian','gaussian'],['gaussian','gaussian','gaussian']],
                      [['gaussian','gaussian','gaussian']]]

        theta_cop1 = [[0.3, 0.5, 0.7, 0.8],
                      [[0.3, 0.4, 0.5],[0.6, 0.7, 0.8],[0.2, 0.3, 0.4]],
                      [[0.3, 0.4, 0.5],[0.3, 0.5, 0.9]],
                      [[0.2, 0.5, 0.9]]] #0.2,0.5,0.9
        
#         margin_cop1 = [['gaussian','gaussian','gaussian'],
#                        [['gaussian','gaussian','gaussian'],['gaussian','gaussian','gaussian']],
#                       [['gaussian','gaussian','gaussian']]]

#         theta_cop1 = [[0.7, 0.8, 0.9],
#                       [[0.3, 0.4, 0.5],[0.6, 0.7, 0.8]],
#                       [[0.2, 0.5, 0.9]]]

#         margin_cop1 = [['gaussian','gaussian'],
#                       [['gaussian','gaussian','gaussian']]]

#         theta_cop1 = [[0.7, 0.8],
#                       [[0.2, 0.5, 0.9]]]


    d = len(r_matrix)
    cop_vine = []
    for tr in range(0,d-1,1):
        cop_vine1 = []
        for col in range(0,d-1-tr,1):
            if (tr == 0) | (binning == False):
                cop_p = cop_par_obj(margin_cop1[tr][col],theta_cop1[tr][col])
                cop_vine1.append(cop_p)
            else:
                cop_vine11 = []
                for bb in range(0,n_bin,1):
                    cop_p = cop_par_obj(margin_cop1[tr][col][bb],theta_cop1[tr][col][bb])
                    cop_vine11.append(cop_p)
                cop_vine1.append(cop_vine11)
        cop_vine.append(cop_vine1)

    for tr in range(0,d-1,1):
        for col in range(0,d-1-tr,1):
            if (tr == 0) | (binning == False):
                print('edge: {} '.format(matrix_edges[tr][col]), 'cop family: {}'.format(cop_vine[tr][col].family), 'theta: {}'.format(cop_vine[tr][col].theta))
            else:
                for bb in range(0,n_bin,1):
                    print('edge: {} '.format(matrix_edges[tr][col]),'bin: {} '.format(bb), 'cop family: {}'.format(cop_vine[tr][col][bb].family), 'theta: {}'.format(cop_vine[tr][col][bb].theta))

    ################# IF YOU WANT TO USE C-VINE OR D-VINE    ################################### 
elif (vine_type == 'c-vine') | (vine_type == 'd-vine'):
    
    
    r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, dim)

    print(r_matrix)

    binning = False
    n_bin = 3
    
    ########## DEFINE MARGINS
    
    margin_vine = []
    for i in range(0,dim,1):
        mar_p = margin_obj('norm', [0,1], True)
        margin_vine.append(mar_p)

    for i in range(0,dim,1):
        print(margin_vine[i].dist, end =' ')
        print(margin_vine[i].theta, end =' ')

    ############## DEFINE COPULAS
    # NN = 0
    tr = 0
    cop_vine = []
    for i in range(dim,1,-1):
        cop_vine1 = []
        for j in range(0,i-1,1):
            if (tr == 0) | (binning == False):
    #             if tr == NN:
    #                 cop_p = cop_par_obj('ind',[])  #
    #                 cop_vine1.append(cop_p)
    #             else:
                cop_p = cop_par_obj('clayton',4.5)  #
                cop_vine1.append(cop_p)
            else:
                cop_vine11 = []
                for bb in range(0,n_bin,1):
                    cop_p = cop_par_obj('gaussian',0.9)
                    cop_vine11.append(cop_p)
                cop_vine1.append(cop_vine11)
        cop_vine.append(cop_vine1)
        tr += 1
    
#     margin_cop1 = [['gaussian','student','clayton','gaussian'],
#                    ['student','clayton','gaussian'],
#                    ['student','gaussian'],
#                    ['clayton']]
        
#     theta_cop1 = [[0.3, [0,0.2], 0.7,  -0.8],
#                   [[-0.8,0.2], 4.5,  -0.2],
#                   [[0,0.5],  -0.7],
#                   [0.9]]
    
#     d = len(r_matrix)
#     cop_vine = []
#     for tr in range(0,d-1,1):
#         cop_vine1 = []
#         for col in range(0,d-1-tr,1):
#             if (tr == 0) | (binning == False):
#                 cop_p = cop_par_obj(margin_cop1[tr][col],theta_cop1[tr][col])
#                 cop_vine1.append(cop_p)
#             else:
#                 cop_vine11 = []
#                 for bb in range(0,n_bin,1):
#                     cop_p = cop_par_obj(margin_cop1[tr][col][bb],theta_cop1[tr][col][bb])
#                     cop_vine11.append(cop_p)
#                 cop_vine1.append(cop_vine11)
#         cop_vine.append(cop_vine1)

    d = len(r_matrix)
    for tr in range(0,d-1,1):
        for col in range(0,d-1-tr,1):
            if (tr == 0) | (binning == False):
                print('edge: {} '.format(matrix_edges[tr][col]), 'cop family: {}'.format(cop_vine[tr][col].family), 'theta: {}'.format(cop_vine[tr][col].theta))
            else:
                for bb in range(0,n_bin,1):
                    print('edge: {} '.format(matrix_edges[tr][col]),'bin: {} '.format(bb), 'cop family: {}'.format(cop_vine[tr][col][bb].family), 'theta: {}'.format(cop_vine[tr][col][bb].theta))

# if binning == True:
#     exc = tf.math.floormod(cases,n_bin)
#     cases = cases - exc

sample, v, v_flip, tau_corr, tau_bins = generate_r_samples(cases, r_matrix, ind_vine, nodes, margin_vine, cop_vine, n_bin, binning)
print(sample)


################### DEFINE VINE #################

vine_type = "d-vine"
method = 'matrix' #'matrix' 'optimal'
families = "kercop"
knots = 50

vine_depth = 2 #len(r_matrix)


margin_vine = []
for i in range(0,vine_depth,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine.append(mar_p)
    
vine = vine_obj_bin(vine_type, families, vine_depth, margin_vine, knots, method, r_matrix)


param = False
binning = False
n_bin = 3

## Make data divisible for bins and k-fold
# x = dat # sample  #dat
# x = sample ## CHANGE THIS IF YOU WANT TO USE DATA THAT ARE LOADED
x = np.array(x,np.float32)


if binning == True:
    if param == False:
        exc = tf.math.floormod(tf.shape(x)[0],n_bin*5)
    else:
        exc = tf.math.floormod(tf.shape(x)[0],n_bin)
    x = x[:tf.shape(x)[0]-exc,:]
else:
    if param == False:
        exc = tf.math.floormod(tf.shape(x)[0],5)
        x = x[:tf.shape(x)[0]-exc,:]

## Prepare copula

sort_n = 'rand'
e = prep_cop(x, vine, sort_n)
print(e)


### FITTING
# Parameters:
# - Data: x
# - Parallel: True or False
# - Bandwidth optimization: LL1 or LL2
# - Binning: True or false.    It can be True only if parallel is false
# - n_bin: Select the number of bins
# parallel = False

gen_dict = {'parallel':True, 'binning':binning, 'param':param, 'vine_depth':vine_depth, 'fitted':False}  #vine_depth
par_dict = {'param_families':["ind","gaussian"]}  #["ind","gaussian","student","clayton","claytonrot90"]
npc_dict = {'opt_method':'LL1','batch_paral':3}
bin_dict = {'n_bin':n_bin}

save_vine = False

vine.fit(x,gen_dict,npc_dict,par_dict,bin_dict)

if save_vine:
    dict_save = {'vine_copulas': vine.copulas, 'r_matrix': vine.r_matrix, 'data': sample}
    pickle_out = open("clay_20_ale","wb")
    pickle.dump(dict_save,pickle_out)
    pickle_out.close()


sample = vine_copula_sample(vine,10000)

# File: src/DVC_tensorflow/experiments/MI_estimators_comparison/MI_estimators_comparison.py


# ### Generate data AWGN channel

import numpy as np
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import pickle
np.random.seed(42)
#tf.set_random_seed(42)
tf.random.set_seed(42)
plt.style.use('ggplot')
import sys

#tf.compat.v1.disable_eager_execution()


#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
### SETUP COPULA
sys.path.append('/n/data2/hms/neurobio/harvey/Houman/NPC/NPC')
#sys.path.append('/Users/safaai/Library/CloudStorage/OneDrive-CompTech/Houman_Work/NPC')
#sys.path.append('C:/Users/alessandromv/Documents/GitHub')
print(sys.path)

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#device = gpu_devices[0]
# tf.config.experimental.set_memory_growth(device, True)
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from classes.objects import *
from vine_tree.tree_op import *

from scipy import stats
import pickle

###########

from param.generate_rvine import *
from param.margin_fit import *
from param.margin_op import *
from param.copula_fit import *
from param.cond_copula import *
from pre_proc.preparation import prep_cop
from pred.prediction import*
from sampling.vine_sample import *
from info.info_estimation import vine_entropy




def sample_AWGN_channel(batch_size, dim, SIGNAL_NOISE = 0.5, SIGNAL_POWER = 2):
    """Simple additive white Gaussian noise channel"""
    x_sample = tf.random.normal((batch_size, dim), stddev = np.sqrt(SIGNAL_POWER))
    y_sample = x_sample + tf.random.normal((batch_size, dim), stddev = np.sqrt(SIGNAL_NOISE))   
    
    return tf.cast(x_sample, tf.float32), tf.cast(y_sample, tf.float32)


### THIS IS TO GENERATE THE DATA BUT IF YOU ALREADY LOADED IS NOT NECESSARY

batch_size = 100
var1, var2 = sample_AWGN_channel(batch_size, 8)
# print(y.shape)


var1 = np.array(var1)
var2 = np.array(var2)

data = np.array([var1,var2]).transpose(1,0,2)
data = data.reshape(data.shape[0],-1)
print(data.shape)

# data = np.array([var1.flatten(),var2.flatten()]).T
# print(data.shape)


# ### Fit copula var1-var2


# 


#### Generate random matrix

#cases = 2000        ### Number of samples
vine_type = 'c-vine' # or 'd-vine' or 'c-vine'
method = 'matrix'  # or 'r_matrix'  only with r-vine
binning = False
n_bin = 3
dim = 16               # Dimension of the vine for random r-vine or c-vine or d-vine
#dim  =8

r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, dim)

print(r_matrix)

binning = False
n_bin = 3

########## DEFINE MARGINS

margin_vine = []
for i in range(0,dim,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine.append(mar_p)

#for i in range(0,dim,1):
#    print(margin_vine[i].dist, end =' ')
#    print(margin_vine[i].theta, end =' ')

############## DEFINE COPULAS

tr = 0
cop_vine = []
for i in range(dim,1,-1):
    cop_vine1 = []
    for j in range(0,i-1,1):
        if (tr == 0) | (binning == False):
            cop_p = cop_par_obj('clayton',4.5)  #
            cop_vine1.append(cop_p)
        else:
            cop_vine11 = []
            for bb in range(0,n_bin,1):
                cop_p = cop_par_obj('gaussian',0.9)
                cop_vine11.append(cop_p)
            cop_vine1.append(cop_vine11)
    cop_vine.append(cop_vine1)
    tr += 1

d = len(r_matrix)
#for tr in range(0,d-1,1):
    #for col in range(0,d-1-tr,1):
        #if (tr == 0) | (binning == False):
            #print('edge: {} '.format(matrix_edges[tr][col]), 'cop family: {}'.format(cop_vine[tr][col].family), 'theta: {}'.format(cop_vine[tr][col].theta))
        #else:
            #for bb in range(0,n_bin,1):
               # print('edge: {} '.format(matrix_edges[tr][col]),'bin: {} '.format(bb), 'cop family: {}'.format(cop_vine[tr][col][bb].family), 'theta: {}'.format(cop_vine[tr][col][bb].theta))



################### DEFINE VINE #################

#vine_type = "c-vine"
method = 'matrix' #'matrix' 'optimal'
families = "kercop"
knots = 50

vine_depth = len(r_matrix)

#vine_depth = 8

margin_vine = []
for i in range(0,vine_depth,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine.append(mar_p)
    
vine = vine_obj_bin(vine_type, families, vine_depth, margin_vine, knots, method, r_matrix)


######### IF YOU WANT TO LOAD THE SAVED VINE --> Put load_pickle = True
load_pickle = False

if load_pickle:
    pickle_in = open("awgn_vine_16","rb")
    dict_save = pickle.load(pickle_in)
    # print(dict_save.keys())
    vine_copulas = dict_save["vine_copulas"]
    data = dict_save["data"]
    r_matrix = dict_save["r_matrix"]
    vine_depth = 16

    ## Here you load the copulas in the vine
    vine.copulas = vine_copulas
    vine.r_matrix = r_matrix
    vine.vine_depth = vine_depth


param = False
binning = False
n_bin = 3

##### Make data divisible for bins and k-fold  #####

# x = dat # sample  #dat
x = data ## CHANGE THIS IF YOU WANT TO USE DATA THAT ARE LOADED
# x = np.array(x,np.float32)


if binning == True:
    if param == False:
        exc = tf.math.floormod(tf.shape(x)[0],n_bin*5)
    else:
        exc = tf.math.floormod(tf.shape(x)[0],n_bin)
    x = x[:tf.shape(x)[0]-exc,:]
else:
    if param == False:
        exc = tf.math.floormod(tf.shape(x)[0],5)
        x = x[:tf.shape(x)[0]-exc,:]

## Prepare copula
sort_n = 'rand'
e = prep_cop(x, vine, sort_n)
print(e)


### FITTING
# Parameters:
# - Data: x
# - Parallel: True or False
# - Bandwidth optimization: LL1 or LL2
# - Binning: True or false.    It can be True only if parallel is false
# - n_bin: Select the number of bins
# parallel = False

#vine_depth=

gen_dict = {'parallel':True, 'binning':binning, 'param':param, 'vine_depth':vine_depth, 'fitted':False}  #vine_depth
par_dict = {'param_families':["ind","gaussian"]}  #["ind","gaussian","student","clayton","claytonrot90"]
npc_dict = {'opt_method':'LL1','batch_paral':4}
bin_dict = {'n_bin':n_bin}

save_vine = False
print('x')
print(x.shape)

vine.fit(x,gen_dict,npc_dict,par_dict,bin_dict)

if save_vine:
    dict_save = {'vine_copulas': vine.copulas, 'r_matrix': vine.r_matrix, 'data': data}
    pickle_out = open("awgn_vine_16","wb")
    pickle.dump(dict_save,pickle_out)
    pickle_out.close()


# ### Fit copula - var2


#### Generate random matrix

#cases = 2000        ### Number of samples
#vine_type = 'c-vine' # or 'd-vine' or 'c-vine'
#method = 'matrix'  # or 'r_matrix'  only with r-vine
#binning = False
#n_bin = 3
dim = 8                # Dimension of the vine for random r-vine or c-vine or d-vine

r_matrix_x2, ind_vine_x2, nodes_x2, matrix_edges_x2 = prepare_vine(vine_type, dim)

print(r_matrix_x2)


########## DEFINE MARGINS

margin_vine_x2 = []
for i in range(0,dim,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine_x2.append(mar_p)

#for i in range(0,dim,1):
#    print(margin_vine_x2[i].dist, end =' ')
#    print(margin_vine_x2[i].theta, end =' ')

############## DEFINE COPULAS

tr = 0
cop_vine_x2 = []
for i in range(dim,1,-1):
    cop_vine1 = []
    for j in range(0,i-1,1):
        if (tr == 0) | (binning == False):
            cop_p = cop_par_obj('clayton',4.5)  #
            cop_vine1.append(cop_p)
        else:
            cop_vine11 = []
            for bb in range(0,n_bin,1):
                cop_p = cop_par_obj('gaussian',0.9)
                cop_vine11.append(cop_p)
            cop_vine1.append(cop_vine11)
    cop_vine_x2.append(cop_vine1)
    tr += 1

d = len(r_matrix_x2)
#for tr in range(0,d-1,1):
#    for col in range(0,d-1-tr,1):
#        if (tr == 0) | (binning == False):
#            print('edge: {} '.format(matrix_edges_x2[tr][col]), 'cop family: {}'.format(cop_vine_x2[tr][col].family), 'theta: {}'.format(cop_vine_x2[tr][col].theta))
#        else:
#            for bb in range(0,n_bin,1):
#                print('edge: {} '.format(matrix_edges_x2[tr][col]),'bin: {} '.format(bb), 'cop family: {}'.format(cop_vine_x2[tr][col][bb].family), 'theta: {}'.format(cop_vine_x2[tr][col][bb].theta))



################### DEFINE VINE #################

#vine_type = "c-vine"
#method = 'matrix' #'matrix' 'optimal'
#families = "kercop"
#knots = 100

vine_depth_x2 = len(r_matrix_x2)

# vine_depth = 20 #len(r_matrix)

margin_vine_x2 = []
for i in range(0,vine_depth_x2,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine_x2.append(mar_p)
    
vine_x2 = vine_obj_bin(vine_type, families, vine_depth_x2, margin_vine_x2, knots, method, r_matrix_x2)

## Here you load the copulas in the vine
# vine_x2.copulas = vine_copulas #_x2

load_pickle = False

if load_pickle:
    pickle_in = open("awgn_vine_x2_8","rb")
    dict_save = pickle.load(pickle_in)
    # print(dict_save.keys())
    vine_copulas_x2 = dict_save["vine_copulas"]
    # sample = dict_save["data"]
    r_matrix_x2 = dict_save["r_matrix"]
    vine_depth_x2 = 8

    ## Here you load the copulas in the vine
    vine_x2.copulas = vine_copulas_x2
    vine_x2.r_matrix = r_matrix_x2
    vine_x2.vine_depth = vine_depth_x2


vine_x2.copulas=vine.copulas.copy()


#param = False
#binning = False
#n_bin = 3

## Make data divisible for bins and k-fold
# x = dat # sample  #dat
# x2 = var2 ## CHANGE THIS IF YOU WANT TO USE DATA THAT ARE LOADED
#x2 = data[:,8:]
x2 = data[:,8:]
# x = np.array(x,np.float32)
print(x2.shape)

if binning == True:
    if param == False:
        exc = tf.math.floormod(tf.shape(x2)[0],n_bin*5)
    else:
        exc = tf.math.floormod(tf.shape(x2)[0],n_bin)
    x2 = x2[:tf.shape(x)[0]-exc,:]
else:
    if param == False:
        exc = tf.math.floormod(tf.shape(x2)[0],5)
        x2 = x2[:tf.shape(x2)[0]-exc,:]

## Prepare copula

sort_n = 'rand'
e_x2 = prep_cop(x2, vine_x2, sort_n)
print(e_x2)


### FITTING
# Parameters:
# - Data: x
# - Parallel: True or False
# - Bandwidth optimization: LL1 or LL2
# - Binning: True or false.    It can be True only if parallel is false
# - n_bin: Select the number of bins
# parallel = False

gen_dict = {'parallel':True, 'binning':binning, 'param':param, 'vine_depth':8, 'fitted':False}  #vine_depth
par_dict = {'param_families':["ind","gaussian"]}  #["ind","gaussian","student","clayton","claytonrot90"]
npc_dict = {'opt_method':'LL1','batch_paral':4}
bin_dict = {'n_bin':n_bin}

save_vine = False

vine_x2.fit(x2,gen_dict,npc_dict,par_dict,bin_dict)


if save_vine:
    dict_save = {'vine_copulas': vine_x2.copulas, 'r_matrix': vine_x2.r_matrix, 'data': x2}
    pickle_out = open("awgn_vine_x2_8","wb")
    pickle.dump(dict_save,pickle_out)
    pickle_out.close()
    
    print('done')



@tf.function
def compute_max(tensor):
    return tf.math.reduce_max(tensor)


def cond_vine_entropy(vine,vine_f2,info_dict):
    alpha = info_dict['alpha']
    cases = info_dict['cases'] #number of samples in each iteration 
    max_iter = info_dict['iterations']
    d = vine.n_cop
    d_f2 = vine_f2.n_cop

    norm_dis = tfd.Normal(loc=0., scale=1.) 
    conf = norm_dis.quantile(1-alpha)
    tim = 0  #Add as parameter if you want to change it

    mo = 0 
    varsum1 = 0 
    infoc1 = 0
    cond_entr = 0
    entr_f2 = 0
    stderr1 = 1e+6
    stderr2 = 1e+6 
    stderr_tot = 1e+6
    erreps = 1e-3
    info=[]
    
 
    mag = tf.math.reduce_max(vine.grid_u.ex)
    mig = tf.math.reduce_min(vine.grid_u.ex)

    mag_f2 = tf.math.reduce_max(vine_f2.grid_u.ex)
    mig_f2 = tf.math.reduce_min(vine_f2.grid_u.ex)

    while ((stderr1 >= erreps) | (stderr2 >= erreps) | (stderr_tot >= erreps) ) & (mo < max_iter):
        mo = mo+1
        info.append(cond_entr-entr_f2)
        print(cond_entr-entr_f2)
        if vine.param == False:

            ## Sample from joint copula and compute prob.
            w = tf.random.uniform([cases,d], minval=0, maxval=1, dtype=vine.data_x.dtype)
            w = (mag-mig)*(w-tf.math.reduce_min(w))/(tf.math.reduce_max(w)-tf.math.reduce_min(w))+mig
            
            sample , u, p1, p2 = vine_copula_sample(vine,cases)
            
            p, p_copula, log_marg_f = vine.evaluation(sample)

            ## Sample from var2 copula and compute prob.
            # w_f2 = tf.random.uniform([cases,d_f2], minval=0, maxval=1, dtype=vine_f2.data_x.dtype)
            # w_f2 = (mag_f2-mig_f2)*(w_f2-tf.math.reduce_min(w_f2))/(tf.math.reduce_max(w_f2)-tf.math.reduce_min(w_f2))+mig_f2
                       
            sample_f2 , u, p1, p2 = vine_copula_sample(vine_f2,cases)

            #sample_f2 = sample[:,0:d_f2]
            --
            p_f2, p_copula_f2, log_marg_f = vine_f2.evaluation(sample_f2)

            ## Compute cond entr.

            p_cond = np.exp(np.log(p.numpy()) - np.log(p_f2.numpy()))
            
            log2_cond = np.log2(p_cond)
            log2_cond[p_cond == 0] = 0 

            cond_entr = cond_entr + ( np.mean(log2_cond) - cond_entr) / mo  #tf.math.reduce_mean

            varsum1 = varsum1 + np.sum((log2_cond - cond_entr)**2)
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
            
            ## Compute entr.
            log2_f2 = np.log2(p_f2.numpy())
            log2_f2[p_f2 == 0] = 0 

            entr_f2 = entr_f2 + ( np.mean(log2_f2) - entr_f2) / mo

        else:
            sample = vine_cop_par_sample(vine,cases)

            # Compute pdf of samples
            p, pcop = vine.evaluation(sample)
            
            log2pp = np.log2(pcop.numpy())
            log2pp[pcop == 0] = 0 

            infoc1 = infoc1 + ( np.mean(log2pp) - infoc1) / mo
            varsum1 = varsum1 + np.sum((log2pp - infoc1)**2)
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
        
    return entr_f2, cond_entr, info


vine.copulas[1].pd_grid_uv.shape


info_dict = {'cases':1000, 'iterations':50, 'alpha': 0.05}
H2,H1_2,info = cond_vine_entropy(vine,vine_x2,info_dict)


print(vine.ind_edge_rel)


print('Entropy H2',H2)
print('Entropy H1|2',H1_2)


MI_XY = -H2 + H1_2
print(MI_XY)


def theoretic_mutual_information_AWGN(power, noise, dim):
    return dim * 0.5 * np.log2(1 + power/noise)

th_mi = theoretic_mutual_information_AWGN(2, 0.5,8)
print(th_mi)


# ### Example sample vine

cases = 1000
sample_vine , u, p1, p2 = vine_copula_sample(vine,cases)

fig, axs = plt.subplots(vine_depth, vine_depth,figsize=(30,30))
for i in range(0,vine_depth,1):
    for j in range(i+1,vine_depth,1):
        axs[i,j].plot(data[:,i],data[:,j],'b.')    
        axs[i,j].plot(sample[:,i],sample[:,j],'r.')    
        axs[i,j].set_title(str(i+1)+","+str(j+1))


plt.figure()
# plt.plot(var1,var2,'.')
plt.plot(var1[:,0],var2[:,0],'.')
plt.plot(sample_vine[:,0],sample_vine[:,8],'.')






# File: src/DVC_tensorflow/experiments/MI_estimators_comparison/MI_estimators_comparison_nested_cleaned.py
#!/usr/bin/env python
# coding: utf-8

# ### Generate data AWGN channel

# In[1]:

import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import pickle
np.random.seed(42)
tf.random.set_seed(42)
plt.style.use('ggplot')
import sys
import numpy

### SETUP COPULA
sys.path.append("D:/NPC")
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
device = gpu_devices[0]
# tf.config.experimental.set_memory_growth(device, True)
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from classes.objects import *
from vine_tree.tree_op import *

from scipy import stats
import pickle
import copy

###########

from param.generate_rvine import *
from param.margin_fit import *
from param.margin_op import *
from param.copula_fit import *
from param.cond_copula import *
from pre_proc.preparation import prep_cop
from pred.prediction import*
from sampling.vine_sample import *
from info.info_estimation import vine_entropy

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

# In[2]:


def sample_AWGN_channel(batch_size, dim, SIGNAL_NOISE = 0.5, SIGNAL_POWER = 2):
    """Simple additive white Gaussian noise channel"""
    x_sample = tf.random.normal((batch_size, dim), stddev = np.sqrt(SIGNAL_POWER))
    y_sample = x_sample + tf.random.normal((batch_size, dim), stddev = np.sqrt(SIGNAL_NOISE))   
    
    return tf.cast(x_sample, tf.float32), tf.cast(y_sample, tf.float32)


# In[3]:


### THIS IS TO GENERATE THE DATA BUT IF YOU ALREADY LOADED IS NOT NECESSARY

batch_size = 2000
var1, var2 = sample_AWGN_channel(batch_size, 8)
# print(y.shape)

var1 = np.array(var1)
var2 = np.array(var2)

data = np.array([var1,var2]).transpose(1,0,2)
data = data.reshape(data.shape[0],-1)
print(data.shape)

# data = np.array([var1.flatten(),var2.flatten()]).T
# print(data.shape)


# ### Fit copula var1-var2

# 

# In[4]:


################################ -  DEFINE THE VINE FOR FITTING - ####################################

### When defining the vine object
### If you use "r-vine", you have to add 'method' and 'r_matrix'
### E.g. using:
### r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, dim)
### and:
### vine = vine_obj_bin(vine_type, families, vine_total_dim, margin_vine, knots, method, r_matrix)

vine_type = "d-vine"
method = 'optimal' #'matrix'#'matrix' 
families = "kercop"
knots = 100

vine_total_dim = data.shape[1]

## Define the margins
margin_vine = []
for i in range(0,vine_total_dim,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine.append(mar_p)

#r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, vine_total_dim)

vine = vine_obj_bin(vine_type, families, vine_total_dim, margin_vine, knots)

######### IF YOU WANT TO LOAD THE SAVED VINE --> Put load_pickle = True
load_pickle = False

if load_pickle:
    pickle_in = open("awgn_vine_16_2000","rb")
    dict_save = pickle.load(pickle_in)
    # print(dict_save.keys())
    vine_copulas = dict_save["vine_copulas"]
    data = dict_save["data"]
    r_matrix = dict_save["r_matrix"]
    vine_depth = 16

    ## Here you load the copulas in the vine
    vine.copulas = vine_copulas
    vine.r_matrix = r_matrix
    vine.vine_depth = vine_depth


# In[5]:


################################ -  VINE FITTING - ####################################
### Vine fitting instructions

# General parameters:
# - parallel: True or False           (Fit in parallel each level)
# - binning: True or False            (It can be True only if parallel is False)
# - param: True or False              (Parametric or Non-parametric)
# - vine_depth: any                   (Max level of vine to fit)
# - Fitted: True or False             (If the vine was already fitted, recompute some needed variables)

# Parametric parameters:
# - param_families: ["ind","gaussian","student","clayton","claytonrot90"]   (Decide which parametric families to fit)

# Non-parametric parameters:
# - opt_method: 'LL1' or 'LL2'

# Binning parameters:
# - n_bin: any                        (Number of bins)

### Parameters

param = False
binning = False
n_bins = 3

### Make data divisible for bins and k-fold

x = np.array(data,np.float32)

if binning == True:
    if param == False:
        exc = x.shape[0] % n_bins*5 #tf.math.floormod(tf.shape(x)[0],n_bin*5)
    else:
        exc = x.shape[0] % n_bins #tf.math.floormod(tf.shape(x)[0],n_bin)
    x = x[:x.shape[0]-exc,:]
else:
    if param == False:
        exc = x.shape[0] % 5 #tf.math.floormod(tf.shape(x)[0],5)
        x = x[:x.shape[0]-exc,:]

### Prepare copula

sort_n = 'rand'
e = prep_cop(x, vine, sort_n)
#print(e)

### FITTING
# Add parameters in a dictionary
vine_depth_fit = 4

gen_dict = {'parallel':True, 'binning':binning, 'param':param, 'vine_depth':vine_depth_fit, 'fitted':False}  #vine_depth
par_dict = {'param_families':["ind","gaussian"]}  #["ind","gaussian","student","clayton","claytonrot90"]
npc_dict = {'opt_method':'LL1','batch_paral':3}
bin_dict = {'n_bin':n_bins}

save_vine = False

vine.fit(x,gen_dict,npc_dict,par_dict,bin_dict)

if save_vine:
    dict_save = {'vine_copulas': vine.copulas, 'r_matrix': vine.r_matrix, 'data': x}
    pickle_out = open("awgn_vine_16_2000","wb")
    pickle.dump(dict_save,pickle_out)
    pickle_out.close()


# ### Fit copula v2

# In[6]:


def copy_copulas(vine):
    new_copula = []
    for tr in range(len(vine.copulas)):
        if tr <= vine.vine_depth:
            new_copula.append(copy.copy(vine.copulas[tr]))
        else:
            list_copula_tmp = []
            for jj in range(len(vine.copulas)-tr):
                list_copula_tmp.append(copy.copy(vine.copulas[tr][jj]))
            new_copula.append(list_copula_tmp)
    return new_copula


# In[7]:


################################ -  DEFINE THE VINE FOR FITTING - ####################################

### When defining the vine object
### If you use "r-vine", you have to add 'method' and 'r_matrix'
### E.g. using:
### r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, dim)
### and:
### vine = vine_obj_bin(vine_type, families, vine_total_dim, margin_vine, knots, method, r_matrix)

#vine_type = "c-vine"
#method = 'matrix' #'matrix' 'optimal'
#families = "kercop"
#knots = 50

var_x2 = data[:,8:]

vine_total_dim_x2 = var_x2.shape[1]

## Define the margins
margin_vine_x2 = []
for i in range(0,vine_total_dim_x2,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine_x2.append(mar_p)
    
vine_x2 = vine_obj_bin(vine_type, families, vine_total_dim_x2, margin_vine_x2, knots)

######### IF YOU WANT TO LOAD THE SAVED VINE --> Put load_pickle = True
# load_pickle = False

# if load_pickle:
#     pickle_in = open("awgn_vine_x2_8","rb")
#     dict_save = pickle.load(pickle_in)
#     # print(dict_save.keys())
#     vine_copulas_x2 = dict_save["vine_copulas"]
#     # sample = dict_save["data"]
#     r_matrix_x2 = dict_save["r_matrix"]
#     vine_depth_x2 = 8

#     ## Here you load the copulas in the vine
#     vine_x2.copulas = vine_copulas_x2
#     vine_x2.r_matrix = r_matrix_x2
#     vine_x2.vine_depth = vine_depth_x2

######################## Nest the vine
## Here you load the copulas in the vine
vine_x2.copulas = copy_copulas(vine)
# vine_x2.copulas = vine.copulas.copy()
# vine_x2.r_matrix = vine.r_matrix.copy()
# vine_x2.vine_depth = 8


# In[8]:


################################ -  VINE FITTING - ####################################
### Vine fitting instructions

# General parameters:
# - parallel: True or False           (Fit in parallel each level)
# - binning: True or False            (It can be True only if parallel is False)
# - param: True or False              (Parametric or Non-parametric)
# - vine_depth: any                   (Max level of vine to fit)
# - Fitted: True or False             (If the vine was already fitted, recompute some needed variables)

# Parametric parameters:
# - param_families: ["ind","gaussian","student","clayton","claytonrot90"]   (Decide which parametric families to fit)

# Non-parametric parameters:
# - opt_method: 'LL1' or 'LL2'

# Binning parameters:
# - n_bin: any                        (Number of bins)

### Parameters

param = False
binning = False
n_bins = 3

### Make data divisible for bins and k-fold

x = np.array(var_x2,np.float32)

if binning == True:
    if param == False:
        exc = x.shape[0] % n_bins*5 #tf.math.floormod(tf.shape(x)[0],n_bin*5)
    else:
        exc = x.shape[0] % n_bins #tf.math.floormod(tf.shape(x)[0],n_bin)
    x = x[:x.shape[0]-exc,:]
else:
    if param == False:
        exc = x.shape[0] % 5 #tf.math.floormod(tf.shape(x)[0],5)
        x = x[:x.shape[0]-exc,:]

### Prepare copula

sort_n = 'rand'
e = prep_cop(x, vine_x2, sort_n)
print(e)

### FITTING
# Add parameters in a dictionary
#vine_depth_fit = 8

gen_dict = {'parallel':True, 'binning':binning, 'param':param, 'vine_depth':vine_depth_fit, 'fitted':False}  #vine_depth
par_dict = {'param_families':["ind","gaussian"]}  #["ind","gaussian","student","clayton","claytonrot90"]
npc_dict = {'opt_method':'LL1','batch_paral':3}
bin_dict = {'n_bin':n_bins}

save_vine = False

vine_x2.fit(x,gen_dict,npc_dict,par_dict,bin_dict)

if save_vine:
    dict_save = {'vine_copulas': vine.copulas, 'r_matrix': vine.r_matrix, 'data': x}
    pickle_out = open("awgn_8_2000","wb")
    pickle.dump(dict_save,pickle_out)
    pickle_out.close()


# In[9]:


def cond_vine_entropy(vine,vine_f2,info_dict):
    alpha = info_dict['alpha']
    cases = info_dict['cases'] #number of samples in each iteration 
    max_iter = info_dict['iterations']
    d = vine.n_cop
    d_f2 = vine_f2.n_cop

    norm_dis = tfd.Normal(loc=0., scale=1.) 
    conf = norm_dis.quantile(1-alpha)
    tim = 0  #Add as parameter if you want to change it

    mo = 0 
    varsum1 = 0 
    infoc1 = 0
    cond_entr = 0
    entr_f2 = 0
    stderr1 = 1e+6
    stderr2 = 1e+6 
    stderr_tot = 1e+6
    erreps = 1e-3

    MI_XY=numpy.zeros(max_iter+1,numpy.float)
    
    mag = tf.math.reduce_max(vine.grid_u.ex)
    mig = tf.math.reduce_min(vine.grid_u.ex)

    mag_f2 = tf.math.reduce_max(vine_f2.grid_u.ex)
    mig_f2 = tf.math.reduce_min(vine_f2.grid_u.ex)

    while ((stderr1 >= erreps) | (stderr2 >= erreps) | (stderr_tot >= erreps) ) & (mo < max_iter):
        mo = mo+1
        if vine.param == False:

            ## Sample from joint copula and compute prob.
            w = tf.random.uniform([cases,d], minval=0, maxval=1, dtype=vine.data_x.dtype)
            w = (mag-mig)*(w-tf.math.reduce_min(w))/(tf.math.reduce_max(w)-tf.math.reduce_min(w))+mig
            
            sample = vine_copula_sample(vine,cases)
            
            p, p_copula = vine.evaluation(sample)

            ## Sample from var2 copula and compute prob.
    
            sample_f2 = sample[:,d_f2:]
            # sample_f2 = vine_copula_sample(vine_f2,cases)
            
            p_f2, p_copula_f2 = vine_f2.evaluation(sample_f2)

            ## Compute cond entr.

            p_cond = np.exp(np.log(p.numpy()) - np.log(p_f2.numpy()))
            
            log2_cond = np.log2(p_cond)
            log2_cond[p_cond == 0] = 0 

            cond_entr = cond_entr + ( np.mean(log2_cond) - cond_entr) / mo  #tf.math.reduce_mean

            varsum1 = varsum1 + np.sum((log2_cond - cond_entr)**2)
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
            
            ## Compute entr.
            log2_f2 = np.log2(p_f2.numpy())
            log2_f2[p_f2 == 0] = 0 

            entr_f2 = entr_f2 + ( np.mean(log2_f2) - entr_f2) / mo
            print('MI: ', -(entr_f2 - cond_entr))

            MI_XY[mo-1] = -(entr_f2 - cond_entr)
        else:
            sample = vine_cop_par_sample(vine,cases)

            # Compute pdf of samples
            p, pcop = vine.evaluation(sample)
            
            log2pp = np.log2(pcop.numpy())
            log2pp[pcop == 0] = 0 

            infoc1 = infoc1 + ( np.mean(log2pp) - infoc1) / mo
            varsum1 = varsum1 + np.sum((log2pp - infoc1)**2)
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
    
    
    return MI_XY, entr_f2, cond_entr


# In[ ]:


info_dict = {'cases':10000, 'iterations':20, 'alpha': 0.05}
MI_XY, H2,H1_2 = cond_vine_entropy(vine,vine_x2,info_dict)
print('MI: ', MI_XY)
plt.plot(MI_XY[:20])


# In[12]:


plt.plot(MI_XY[:20])


# In[28]:


def theoretic_mutual_information_AWGN(power, noise, dim):
    return dim * 0.5 * np.log2(1 + power/noise)

th_mi = theoretic_mutual_information_AWGN(2, 0.5,8)
print(th_mi)


# ### Example sample vine

# In[13]:


cases = 2000
sample = vine_copula_sample(vine,cases)

fig, axs = plt.subplots(vine.n_cop, vine.n_cop,figsize=(30,30))
for i in range(0,vine.n_cop,1):
    for j in range(i+1,vine.n_cop,1):
        axs[i,j].plot(data[:,i],data[:,j],'b.')    
        axs[i,j].plot(sample[:,i],sample[:,j],'r.')    
        axs[i,j].set_title(str(i+1)+","+str(j+1))


# In[37]:


cases = 2000
sample_x2 = vine_copula_sample(vine_x2,cases)

fig, axs = plt.subplots(vine_x2.n_cop, vine_x2.n_cop,figsize=(30,30))
for i in range(0,vine_x2.n_cop,1):
    for j in range(i+1,vine_x2.n_cop,1):
        axs[i,j].plot(data[:,vine_x2.n_cop+i],data[:,vine_x2.n_cop+j],'b.')    
        axs[i,j].plot(sample_x2[:,i],sample_x2[:,j],'r.')    
        axs[i,j].set_title(str(i+1)+","+str(j+1))



# File: src/DVC_tensorflow/experiments/VAE/autoencode_gaussian.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.path.append("/Users/safaai/Library/CloudStorage/OneDrive-CompTech/Houman_Work/NPC")
from classes.objects import *
from vine_tree.tree_op import *
from pre_proc.preparation import prep_cop
import cProfile

# Data generation for two labels
def generate_data(num_samples_label1, num_samples_label2, label1, label2):
    mat1 = np.eye(10)
    mat2 = np.array([[0, 0.1, 0.02, 0.01, 0, 0, 0, 0, 0, 0],
                     [0.01, 0, 0.04, 0.03, 0.01, 0, 0, 0, 0, 0],
                     [0.02, 0.04, 0, 0.05, 0.03, 0.01, 0, 0, 0, 0],
                     [0.01, 0.03, 0.05, 0, 0.04, 0.02, 0.01, 0, 0, 0],
                     [0, 0.01, 0.03, 0.04, 0, 0.1, 0.03, 0.01, 0, 0],
                     [0, 0, -0.01, 0.02, -0.05, 0, -0.04, 0.02, -0.01, 0],
                     [0, 0, 0, -0.01, -0.03, -0.4, 0, -0.05, -0.03, -0.01],
                     [0, 0, 0, 0, -0.01, -0.02, -0.05, 0, -0.04, -0.02],
                     [0, 0, 0, 0, 0, 0.01, -0.03, -0.04, 0, -0.05],
                     [0, 0, 0, 0, 0, 0, 0.01, -0.02, -0.05, 0]]) * 2 
    mat3 = np.transpose(mat2)
    mat4 = np.eye(10)

    top_row = np.concatenate((mat1, mat2), axis=1)
    bottom_row = np.concatenate((mat3, mat4), axis=1)
    cov_matrix_label1 = np.concatenate((top_row, bottom_row), axis=0)
    cov_matrix_label2 = -cov_matrix_label1   # or np.abs(cov_matrix_label1) if needed
    for i in range(cov_matrix_label2.shape[0]):
        cov_matrix_label2[i, i] = 1

    mean = np.zeros(20)
    data_label1 = np.random.multivariate_normal(mean, cov_matrix_label1, size=num_samples_label1)
    data_label2 = np.random.multivariate_normal(mean, cov_matrix_label2, size=num_samples_label2)

    data = np.concatenate((data_label1, data_label2), axis=0)
    labels = np.concatenate((np.full(num_samples_label1, label1), np.full(num_samples_label2, label2)))

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    return data[indices], labels[indices]


def shuffle_dimensions(data, labels):
    unique_labels = np.unique(labels)
    shuffled_data = []

    for label in unique_labels:
        # Extract data for the current label
        data_label = data[labels == label]

        # Shuffle each dimension of the data for this label
        shuffled_label_data = np.copy(data_label)
        for i in range(data_label.shape[1]):
            np.random.shuffle(shuffled_label_data[:, i])

        # Append the shuffled data for this label
        shuffled_data.append(shuffled_label_data)

    # Concatenate the shuffled data for all labels
    return np.concatenate(shuffled_data, axis=0)

def calculate_cdf(data, labels):
    unique_labels = np.unique(labels)
    cdf_data = np.zeros_like(data)

    for label in unique_labels:
        # Extract data for the current label
        data_label = data[labels == label]

        # Compute CDF for each dimension of the data for this label
        cdf_label_data = np.argsort(np.argsort(data_label, axis=0), axis=0) / float(data_label.shape[0] - 1)
        
        # Assign the computed CDFs back to the corresponding positions in the overall CDF array
        cdf_data[labels == label] = cdf_label_data

    return cdf_data

# Define the sample_z function
def sample_z(args):
    mu, log_sigma, epsilon = args
    sigma = tf.exp(0.5 * log_sigma)
    return mu + sigma * epsilon

input_dim = 20
latent_dim = 50  # Latent dimension

# Define the encoder
input_x_shuffled = Input(shape=(input_dim,))
input_u = Input(shape=(input_dim,))
input_label = Input(shape=(2,))
x1 = Concatenate()([input_x_shuffled, input_label])
x1 = Dense(64, activation='relu')(x1)
x1 = Dense(32, activation='relu')(x1)
x1 = Dropout(0.2)(x1)
mu = Dense(latent_dim, name='mu')(x1)
log_sigma = Dense(latent_dim, name='log_sigma')(x1)
x2 = Concatenate()([input_u, input_label])
x2 = Dense(128, activation='relu')(x2)
x2 = Dense(64, activation='relu')(x2)
x2 = Dense(32, activation='relu')(x2)
x2 = Dropout(0.2)(x2)
epsilon = Dense(latent_dim, activation='sigmoid', name='epsilon')(x2)
z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([mu, log_sigma, epsilon])
encoder = Model([input_x_shuffled, input_u, input_label], [mu, log_sigma, z], name='encoder')

# Define the decoder
decoder_inputs = Input(shape=(latent_dim,))
decoder_input = Concatenate()([decoder_inputs, input_label])
x = Dense(32, activation="relu")(decoder_input)
x = Dense(64, activation="relu")(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
decoder_outputs = Dense(input_dim, activation='linear')(x)
decoder = Model([decoder_inputs, input_label], decoder_outputs, name="decoder")


class CustomVAE(Model):
    def __init__(self, encoder, decoder):
        super(CustomVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def vae_loss(self, y, reconstructed, z_mean, z_log_var):
        reconstruction_loss = tf.reduce_mean(tf.square(y - reconstructed))
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        return reconstruction_loss + kl_loss

    def train_step(self, data):
        x, y = data
        x_shuffled, u, labels = x
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([x_shuffled, u, labels])
            reconstructed = self.decoder([z, labels])
            loss = self.vae_loss(y, reconstructed, z_mean, z_log_var)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': loss}
    
    def call(self, inputs):
        input_x_shuffled, input_u, input_label = inputs
        z_mean, z_log_var, z = self.encoder([input_x_shuffled, input_u, input_label])
        reconstructed = self.decoder([z, input_label])
        return reconstructed

# Example usage
num_samples_label1 = 7000
num_samples_label2 = 7000
label1 = 0
label2 = 1
data, labels = generate_data(num_samples_label1, num_samples_label2, label1, label2)
labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=2)
x_train_shuffled = shuffle_dimensions(data, labels)
u_train = calculate_cdf(data,labels)

# Instantiate and compile the custom VAE
vae = CustomVAE(encoder, decoder)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
vae.compile(optimizer=optimizer)


# Adjust batch size or dataset size to be compatible
batch_size = 128
dataset_size = data.shape[0]
adjusted_size = (dataset_size // batch_size) * batch_size  # Truncate dataset to fit batch size

# Use the adjusted dataset for training
vae.fit([x_train_shuffled[:adjusted_size], u_train[:adjusted_size], labels_one_hot[:adjusted_size]], 
        data[:adjusted_size], epochs=50, batch_size=batch_size)

# Visualization
cases = len(data)
sample = vae.predict([x_train_shuffled[:cases], u_train[:cases], labels_one_hot[:cases]])
data_sample = data[:cases, :]


label_classes = labels

dim = input_dim // 2
n = dim * 2

def compute_and_plot_correlation(data, sample, label_class):
    corr_sample = np.zeros((n, n), dtype=np.float64)
    corr_data = np.zeros((n, n), dtype=np.float64)

    data_class = data[label_classes == label_class]
    sample_class = sample[label_classes == label_class]

    fig, axs = plt.subplots(n, n, figsize=(n, n))
    for i in range(n):
        for j in range(n):
            if i < j:
                axs[i, j].plot(data_class[:, i], data_class[:, j], 'b.', markersize=1)
                axs[i, j].plot(sample_class[:, i], sample_class[:, j], 'r.', markersize=1)
                corr_sample[i, j] = stats.spearmanr(sample_class[:, i], sample_class[:, j])[0]
                corr_data[i, j] = stats.spearmanr(data_class[:, i], data_class[:, j])[0]
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    plt.tight_layout()
    plt.show()

    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    # Correlation Data
    fig, axs = plt.subplots(1, 1)
    plt.imshow(corr_data, cmap=cmap, vmin=-0.1, vmax=0.1)
    plt.colorbar()
    plt.show()

    # Correlation Sample
    fig, axs = plt.subplots(1, 1)
    plt.imshow(corr_sample, cmap=cmap, vmin=-0.1, vmax=0.1)
    plt.colorbar()
    plt.show()

# Call the function for each label class
for label_class in np.unique(label_classes):
    compute_and_plot_correlation(data, sample, label_class)

##################################################################
##################################################################
##################################################################
########################## VINE PART #############################
##################################################################
##################################################################

dat = np.array(data[:1000,:], np.float64)

vine_type = "d-vine"
method = 'matrix' #'matrix' 'optimal'
families = "kercop"
knots = 50
vine_total_dim = dat.shape[1]
margin_vine = []
for i in range(0,vine_total_dim,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine.append(mar_p) 

r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, vine_total_dim)
vine = vine_obj_bin(vine_type, families, vine_total_dim, margin_vine, knots, method, r_matrix)

sort_n = 'rand' 
e = prep_cop(dat, vine, sort_n)

param = False
binning = False
n_bins = 3
parallel = False        

vine_depth_fit = vine_total_dim + 1

gen_dict = {'parallel':parallel, 'binning':binning, 'param':param, 'vine_depth':vine_depth_fit, 'fitted':False}  
par_dict = {'param_families':["ind","gaussian"]}  #["ind","gaussian","student","clayton","claytonrot90"]
npc_dict = {'opt_method':'LL1','batch_paral':5}
bin_dict = {'n_bin':n_bins}

cProfile.run('vine.fit(dat,gen_dict,npc_dict,par_dict,bin_dict)')

from pred.prediction import*

exp_dim = 100
dim = 0
points = create_points(dat,dim,exp_dim)
min_dim = tf.math.reduce_min(dat[:,dim])
max_dim = tf.math.reduce_max(dat[:,dim])
y_vec = tf.linspace(min_dim-2e-16+1e-5,max_dim+2e-16,exp_dim)

p, p_cop, logp = vine.evaluation(points)
p1 = tf.where(tf.math.is_nan(p), tf.zeros_like(p), p)

p1 = reshaped_tensor = tf.reshape(p1, (dat.shape[0], exp_dim))
y_ML, y_EM = predict_response(p1, y_vec)
p, y_ml, y_em = predict_vine(dat,vine,dim,exp_dim)

from scipy import stats
#corr = stats.pearsonr(dat[:,dim], y_ml)
corre = stats.pearsonr(dat[:,dim], y_em)
#CORR = stats.pearsonr(dat[:,dim], y_ML)
CORRE = stats.pearsonr(dat[:,dim], y_EM)
print('Correlation: ' +  str(corre[0]) + ' , ' + str(CORRE[0]))
plt.figure()
#plt.plot(y_ml, dat[:,dim], 'g.')
plt.plot(y_em, dat[:,dim], 'r.')
plt.plot(y_EM, dat[:,dim], 'b.')
plt.show()

from sampling.vine_sample import *

sample, u, sample_pdf, sample_pdsmple = vine_copula_sample(vine,10000)

fig, axs = plt.subplots(vine.n_cop, vine.n_cop,figsize=(20,20))
for i in range(0,vine.n_cop,1):
    for j in range(i+1,vine.n_cop,1):
        axs[i,j].plot(data[:,i],data[:,j],'b.', markersize=1)    
        axs[i,j].plot(sample[:,i],sample[:,j],'r.', markersize=1)    
        axs[i,j].set_title(str(i+1)+","+str(j+1))

#for label_class in np.unique(label_classes):
#    compute_and_plot_correlation(data, sample, label_class)

# Call the function for each label class
for label_class in np.unique(label_classes):
    compute_and_plot_correlation(data, sample, label_class)

# File: src/DVC_tensorflow/experiments/VAE/autoencode_gaussian_vanilla.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

# Data generation for two labels
def generate_data(num_samples_label1, num_samples_label2, label1, label2):
    mat1 = np.eye(10)
    mat2 = np.array([[0, 0.1, 0.02, 0.01, 0, 0, 0, 0, 0, 0],
                     [0.01, 0, 0.04, 0.03, 0.01, 0, 0, 0, 0, 0],
                     [0.02, 0.04, 0, 0.05, 0.03, 0.01, 0, 0, 0, 0],
                     [0.01, 0.03, 0.05, 0, 0.04, 0.02, 0.01, 0, 0, 0],
                     [0, 0.01, 0.03, 0.04, 0, 0.1, 0.03, 0.01, 0, 0],
                     [0, 0, -0.01, 0.02, -0.05, 0, -0.04, 0.02, -0.01, 0],
                     [0, 0, 0, -0.01, -0.03, -0.4, 0, -0.05, -0.03, -0.01],
                     [0, 0, 0, 0, -0.01, -0.02, -0.05, 0, -0.04, -0.02],
                     [0, 0, 0, 0, 0, 0.01, -0.03, -0.04, 0, -0.05],
                     [0, 0, 0, 0, 0, 0, 0.01, -0.02, -0.05, 0]]) * 2 
    mat3 = np.transpose(mat2)
    mat4 = np.eye(10)

    top_row = np.concatenate((mat1, mat2), axis=1)
    bottom_row = np.concatenate((mat3, mat4), axis=1)
    cov_matrix_label1 = np.concatenate((top_row, bottom_row), axis=0)
    cov_matrix_label2 = -cov_matrix_label1   # or np.abs(cov_matrix_label1) if needed
    for i in range(cov_matrix_label2.shape[0]):
        cov_matrix_label2[i, i] = 1

    mean = np.zeros(20)
    data_label1 = np.random.multivariate_normal(mean, cov_matrix_label1, size=num_samples_label1)
    data_label2 = np.random.multivariate_normal(mean, cov_matrix_label2, size=num_samples_label2)

    data = np.concatenate((data_label1, data_label2), axis=0)
    labels = np.concatenate((np.full(num_samples_label1, label1), np.full(num_samples_label2, label2)))

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    return data[indices], labels[indices]



# Define the sample_z function
def sample_z(args):
    mu, log_sigma = args
    sigma = tf.exp(0.5 * log_sigma)
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(mu))
    return mu + sigma * epsilon

input_dim = 20
latent_dim = 50  # Latent dimension

# Define the encoder
input_data = Input(shape=(input_dim,))
x = Dense(64, activation='relu')(input_data)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
mu = Dense(latent_dim, name='mu')(x)
log_sigma = Dense(latent_dim, name='log_sigma')(x)
z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([mu, log_sigma])
encoder = Model(input_data, [mu, log_sigma, z], name='encoder')

# Define the decoder
decoder_inputs = Input(shape=(latent_dim,))
x = Dense(32, activation="relu")(decoder_inputs)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
decoder_outputs = Dense(input_dim, activation='linear')(x)
decoder = Model(decoder_inputs, decoder_outputs, name="decoder")


class CustomVAE(Model):
    def __init__(self, encoder, decoder):
        super(CustomVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def vae_loss(self, y, reconstructed, z_mean, z_log_var):
        reconstruction_loss = tf.reduce_mean(tf.square(y - reconstructed))
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        return reconstruction_loss + kl_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstructed = self.decoder(z)
            loss = self.vae_loss(y, reconstructed, z_mean, z_log_var)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': loss}
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed



# Example usage
num_samples_label1 = 7000
num_samples_label2 = 7000
label1 = 0
label2 = 1
data, labels = generate_data(num_samples_label1, num_samples_label2, label1, label2)
labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=2)

# Instantiate and compile the custom VAE
vae = CustomVAE(encoder, decoder)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
vae.compile(optimizer=optimizer)

# Train the model
batch_size = 128
dataset_size = data.shape[0]
adjusted_size = (dataset_size // batch_size) * batch_size  # Truncate dataset to fit batch size
vae.fit(data[:adjusted_size], data[:adjusted_size], epochs=50, batch_size=batch_size)

# Visualization
cases = len(data)
sample = vae.predict(data[:cases])
data_sample = data[:cases, :]


label_classes = labels

dim = input_dim // 2
n = dim * 2

def compute_and_plot_correlation(data, sample, label_class):
    corr_sample = np.zeros((n, n), dtype=np.float64)
    corr_data = np.zeros((n, n), dtype=np.float64)

    data_class = data[label_classes == label_class]
    sample_class = sample[label_classes == label_class]

    fig, axs = plt.subplots(n, n, figsize=(n, n))
    for i in range(n):
        for j in range(n):
            if i < j:
                axs[i, j].plot(data_class[:, i], data_class[:, j], 'b.', markersize=1)
                axs[i, j].plot(sample_class[:, i], sample_class[:, j], 'r.', markersize=1)
                corr_sample[i, j] = stats.spearmanr(sample_class[:, i], sample_class[:, j])[0]
                corr_data[i, j] = stats.spearmanr(data_class[:, i], data_class[:, j])[0]
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    plt.tight_layout()
    plt.show()

    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    # Correlation Data
    fig, axs = plt.subplots(1, 1)
    plt.imshow(corr_data, cmap=cmap, vmin=-0.1, vmax=0.1)
    plt.colorbar()
    plt.show()

    # Correlation Sample
    fig, axs = plt.subplots(1, 1)
    plt.imshow(corr_sample, cmap=cmap, vmin=-0.1, vmax=0.1)
    plt.colorbar()
    plt.show()

# Call the function for each label class
for label_class in np.unique(label_classes):
    compute_and_plot_correlation(data, sample, label_class)

# File: src/DVC_tensorflow/grid/.ipynb_checkpoints/grid_class-checkpoint.py
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

# File: src/DVC_tensorflow/grid/.ipynb_checkpoints/grid_op-checkpoint.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from utils.bijector import *

@tf.function
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

# File: src/DVC_tensorflow/grid/grid_class.py
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

# File: src/DVC_tensorflow/grid/grid_op.py
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
 

# File: src/DVC_tensorflow/info/.ipynb_checkpoints/info_estimation-checkpoint.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from utils.prob_op import kernel_pdf2
from utils.tensor_op import update_tensor, update_tensor2D, replace_nan_inf
from pre_proc.preparation import prep_copula
from sampling.vine_sample import *

########################## INFO ESTIMATION ##################################

def vine_entropy(vine,info_dict):
    alpha = info_dict['alpha']
    cases = info_dict['cases'] #number of samples in each iteration 
    max_iter = info_dict['iterations']
    d = vine.n_cop
    
#     if vine.binning == True:
#         exc = tf.math.floormod(cases,vine.n_bin)
#         cases = cases - exc

    norm_dis = tfd.Normal(loc=0., scale=1.) 
    conf = norm_dis.quantile(1-alpha)
    tim = 0  #Add as parameter if you want to change it

    mo = 0 
    varsum1 = 0 
    infoc1 = 0
    stderr1 = 1e+6
    stderr2 = 1e+6 
    stderr_tot = 1e+6
    erreps = 1e-3

    mag = tf.math.reduce_max(vine.grid_u.ex)
    mig = tf.math.reduce_min(vine.grid_u.ex)

    while ((stderr1 >= erreps) | (stderr2 >= erreps) | (stderr_tot >= erreps) ) & (mo < max_iter):
        mo = mo+1
        if vine.param == False:
            w = tf.random.uniform([cases,d], minval=0, maxval=1, dtype=vine.data_x.dtype)
            w = (mag-mig)*(w-tf.math.reduce_min(w))/(tf.math.reduce_max(w)-tf.math.reduce_min(w))+mig
            
            sample = vine_copula_sample(vine,cases)
            
#             if vine.binning == True:
#                 if vine.param == False:
#                     exc = tf.math.floormod(tf.shape(sample)[0],5)
#                 sample = sample[:tf.shape(sample)[0]-exc,:]
            
            p, p_copula = vine.evaluation(sample)
            
            log2pp = np.log2(p_copula.numpy())
            log2pp[p_copula == 0] = 0 
            
#             log2pp =  tf.py_function(np.log2, [p_copula+1e-20], vine.data_x.dtype) #1e-20 because one p = 0

#             log2pp = replace_nan_inf(log2pp).numpy()

            infoc1 = infoc1 + ( np.mean(log2pp) - infoc1) / mo  #tf.math.reduce_mean

            varsum1 = varsum1 + np.sum((log2pp - infoc1)**2)
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
        else:
            sample = vine_cop_par_sample(vine,cases)

            # Compute pdf of samples
            p, pcop = vine.evaluation(sample)
            
            log2pp = np.log2(pcop.numpy())
            log2pp[pcop == 0] = 0 
            
#             log2pp = tf.py_function(np.log2, [pcop], vine.data_u.dtype)

#             log2pp = replace_nan_inf(log2pp).numpy()

            infoc1 = infoc1 + ( np.mean(log2pp) - infoc1) / mo
            varsum1 = varsum1 + np.sum((log2pp - infoc1)**2)
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
            
    return infoc1

# File: src/DVC_tensorflow/info/info_estimation.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from utils.prob_op import kernel_pdf2
from utils.tensor_op import update_tensor, update_tensor2D, replace_nan_inf
from pre_proc.preparation import prep_copula
from sampling.vine_sample import vine_copula_sample, vine_cop_par_sample

########################## INFO ESTIMATION ##################################

def vine_entropy(vine,info_dict):
    alpha = info_dict['alpha']
    cases = info_dict['cases'] #number of samples in each iteration 
    max_iter = info_dict['iterations']
    d = vine.n_cop

    norm_dis = tfd.Normal(loc=0., scale=1.) 
    conf = norm_dis.quantile(1-alpha)
    tim = 0  #Add as parameter if you want to change it

    mo = 0 
    varsum1 = 0 
    infoc1 = 0
    stderr1 = 1e+6
    stderr2 = 1e+6 
    stderr_tot = 1e+6
    erreps = 1e-3

    mag = tf.math.reduce_max(vine.grid_u.ex)
    mig = tf.math.reduce_min(vine.grid_u.ex)

    while ((stderr1 >= erreps) | (stderr2 >= erreps) | (stderr_tot >= erreps) ) & (mo < max_iter):
        mo = mo+1
        if vine.param == False:
            w = tf.random.uniform([cases,d], minval=0, maxval=1, dtype=vine.data_x.dtype)
            w = (mag-mig)*(w-tf.math.reduce_min(w))/(tf.math.reduce_max(w)-tf.math.reduce_min(w))+mig
            
            sample = vine_copula_sample(vine,cases)

            #sample = tf.convert_to_tensor(sample)
            #print(type(sample))      

            p, p_copula, plog = vine.evaluation(sample)
            
            log2pp = np.log2(p_copula.numpy())
            log2pp[p_copula == 0] = 0 

            infoc1 = infoc1 + ( np.mean(log2pp) - infoc1) / mo  #tf.math.reduce_mean

            varsum1 = varsum1 + np.sum((log2pp - infoc1)**2)
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
        else:
            sample, _, _, _ = vine_cop_par_sample(vine,cases)

            # Compute pdf of samples
            p, pcop, _ = vine.evaluation(sample)
            
            log2pp = np.log2(pcop.numpy())
            log2pp[pcop == 0] = 0 

            infoc1 = infoc1 + ( np.mean(log2pp) - infoc1) / mo
            varsum1 = varsum1 + np.sum((log2pp - infoc1)**2)
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
            
    return infoc1

# File: src/DVC_tensorflow/main_all.py
import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

########### Packages from the vine library

from pre_proc.define_copulas import define_copulas
from param.generate_rvine import generate_r_samples
from pre_proc.preparation import prep_cop
from sampling.vine_sample import vine_copula_sample
from utils.tensor_op import create_points
from classes.objects import vine_obj_bin, margin_obj
from pred.prediction import predict_vine
from info.info_estimation import vine_entropy


################################ -  DEFINE THE VINE FOR GENERATING THE DATA - ####################################

#### Generate random matrix
cases = 1000        ### Number of samples
vine_type = 'c-vine' # or 'd-vine' or 'r-vine'
method = 'matrix'  # or 'r_matrix'  only with r-vine
binning = False
n_bin = 3
dim = 5                # Dimension of the vine for random r-vine or c-vine or d-vine

### Define copulas and vine. Please look at the function in ./pre_proc/define_copulas and change directly in there
r_matrix, cop_vine, ind_vine, nodes, matrix_edges, margin_vine = define_copulas(vine_type, method, binning, n_bin, dim)

# if binning == True:
#     exc = tf.math.floormod(cases,n_bin)
#     cases = cases - exc

sample, v, v_flip, tau_corr, tau_bins = generate_r_samples(cases, r_matrix, ind_vine, nodes, margin_vine, cop_vine, n_bin, binning)
print(sample)

## Plot one example of sample
plt.figure()
plt.plot(sample[:,0],sample[:,1],'.')
plt.show()

################################ -  LOAD MATLAB OR PICKLE FILE - ####################################

#  If you want to load a matlab file or a vine saved in a pickle file
load_mat = False
load_pickle = False #True

if load_mat:
    mat_contents = sio.loadmat('stu_ex_01.mat') #sim_vine #sim17_10000.mat')   #X_sim17.mat')
    # print(mat_contents)
    dat = mat_contents.get('x') #X_sim17
    dat = np.array(dat,np.float64)

    pdf_cop = mat_contents.get('pdf_cop') #X_sim17
    pdf_cop = np.array(pdf_cop,np.float32)
    
if load_pickle:
    pickle_in = open("clay_20_ale","rb")  #clay_20_ale
    dict_save = pickle.load(pickle_in)
    vine_copulas = dict_save["vine_copulas"]
    x = dict_save["data"] #x
    r_matrix = dict_save["r_matrix"]
    vine_depth = 20

x = sample
################################ -  DEFINE THE VINE FOR FITTING - ####################################

vine_type = "d-vine"
method = 'matrix' #'matrix' 'optimal'
families = "kercop"
knots = 50

vine_depth = len(r_matrix)

## Define the margins
margin_vine = []
for i in range(0,vine_depth,1):
    mar_p = margin_obj('norm', [0,1], True)
    margin_vine.append(mar_p)
    
vine = vine_obj_bin(vine_type, families, vine_depth, margin_vine, knots, method, r_matrix)

if load_pickle:
    vine.copulas = vine_copulas

################################ -  VINE FITTING - ####################################
### Vine fitting instructions

# General parameters:
# - parallel: True or False           (Fit in parallel each level)
# - binning: True or False            (It can be True only if parallel is False)
# - param: True or False              (Parametric or Non-parametric)
# - vine_depth: any                   (Max level of vine to fit)
# - Fitted: True or False             (If the vine was already fitted, recompute some needed variables)

# Parametric parameters:
# - param_families: ["ind","gaussian","student","clayton","claytonrot90"]   (Decide which parametric families to fit)

# Non-parametric parameters:
# - opt_method: 'LL1' or 'LL2'

# Binning parameters:
# - n_bin: any                        (Number of bins)

### Parameters

param = False
binning = False
n_bin = 3

### Make data divisible for bins and k-fold

x = np.array(x,np.float32)

if binning == True:
    if param == False:
        exc = x.shape[0] % n_bins*5 #tf.math.floormod(tf.shape(x)[0],n_bin*5)
    else:
        exc = x.shape[0] % n_bins #tf.math.floormod(tf.shape(x)[0],n_bin)
    x = x[:x.shape[0]-exc,:]
else:
    if param == False:
        exc = x.shape[0] % 5 #tf.math.floormod(tf.shape(x)[0],5)
        x = x[:x.shape[0]-exc,:]

### Prepare copula

sort_n = 'rand'
e = prep_cop(x, vine, sort_n)
print(e)

### FITTING
# Add parameters in a dictionary
tf.config.experimental_run_functions_eagerly(True)

gen_dict = {'parallel':True, 'binning':binning, 'param':param, 'vine_depth':vine_depth, 'fitted':False}  #vine_depth
par_dict = {'param_families':["ind","gaussian"]}  #["ind","gaussian","student","clayton","claytonrot90"]
npc_dict = {'opt_method':'LL1','batch_paral':3}
bin_dict = {'n_bin':n_bin}

save_vine = False

vine.fit(x,gen_dict,npc_dict,par_dict,bin_dict)

if save_vine:
    dict_save = {'vine_copulas': vine.copulas, 'r_matrix': vine.r_matrix, 'data': x}
    pickle_out = open("clay_20_ale","wb")
    pickle.dump(dict_save,pickle_out)
    pickle_out.close()

################################ -  VINE SAMPLING - ####################################

sample = vine_copula_sample(vine,2000)

################################ -  VINE EVALUATION - ####################################

### Create points for evaluation

exp_dim = 100
dim = 0
 
points = create_points(x,dim,exp_dim)
print(points)
print(type(points))

p, p_cop, plog = vine.evaluation(points)



################################ -  PREDICT VINE - ####################################
dim = 0
exp_dim = 100

p, y_ml, y_em = predict_vine(x,vine,dim,exp_dim)


## To remove this, check why a NaN is generated. One point only gives this problem, probably at the boundary
print(np.where(np.isnan(y_ml)))
print(np.where(np.isnan(y_em)))

#### IF NAN
from utils.tensor_op import replace_nan_inf

y_em = replace_nan_inf(y_em)

## Compute correlation and plot

from scipy import stats
corr = stats.pearsonr(x[:,dim], y_em)

print(corr[0])

plt.figure()
plt.plot(x[:,dim], y_ml, 'r.')
plt.plot(x[:,dim], y_em, 'b.')
plt.title('Correlation: ' + str(corr[0]))
plt.show()

################################ -  MI ESTIMATION - ####################################
info_dict = {'cases':1000, 'iterations':10, 'alpha': 0.05}
MI = vine_entropy(vine,info_dict)
print(MI)

# File: src/DVC_tensorflow/optim/.ipynb_checkpoints/MISE-checkpoint.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from evalu.cop_eval import eval_rs_p
from optim.local_lik import loclik_batch

############################## MISE COST FUNCTION ################################

@tf.function(experimental_relax_shapes=True)
def MISE_mul(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, all_x, data_x_train, data_s_test, n_cop, batch_size, NORM1, norm_flag):
    if tf.math.equal(tf.shape(tf.shape(all_x)),2):
        all_x = all_x[...,tf.newaxis]
    
    bw1 = tf.abs(a*bw)
    n_splits = tf.shape(data_x_train)[3]  

    if tf.math.equal(tf.shape(tf.shape(grid_x)),2):
        grid_x = grid_x[...,tf.newaxis]

    ker_grid_all = loclik_batch(bw1, all_x, grid_x, n_cop, batch_size) #data_x
    if norm_flag == True:
        ker_grid_all = tf.transpose(tf.reshape(ker_grid_all,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop]),perm=[1, 0, 2])
        pd_grid = eval_rs_p(adu11, adu22, ker_grid_all, NORM1, n_cop)
    else:
        pd_grid = tf.zeros([tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop],bw.dtype)
        
    kkk_fin = tf.TensorArray(grid_x.dtype,size=n_splits)
    for k in tf.range(0,n_splits,1,tf.int32):
        ker_grid_fin = loclik_batch(bw1, data_x_train[:,:,:,k], grid_x, n_cop, batch_size)
        pd_grid1 = tf.transpose(tf.reshape(ker_grid_fin,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop]),perm=[1, 0, 2])
        if norm_flag == True:
            pd_grid1 = eval_rs_p(adu11, adu22, pd_grid1, NORM1, n_cop)
            
        interp_data = tf.TensorArray(data_x_train.dtype, size=n_cop)
        for kk in tf.range(0,n_cop,1,tf.int32):
            interp_data1 = tfp.math.batch_interp_regular_nd_grid(data_s_test[:,:,kk,k], min_s, max_s, pd_grid1[:,:,kk], axis=-2)
            interp_data = interp_data.write(kk,interp_data1)
        interp_data = interp_data.stack()
        interp_data = tf.transpose(interp_data)
        if norm_flag == True:
            interp_data = interp_data / tf.math.reduce_sum(pd_grid * step_s,[0,1])
        else:
            interp_data = interp_data / tf.math.reduce_sum(ker_grid_fin * step_s,0)
        kkk_fin = kkk_fin.write(k,interp_data)
    kkk_fin = kkk_fin.stack()
    kkk_fin = tf.reshape(kkk_fin,[tf.shape(kkk_fin)[0]*tf.shape(kkk_fin)[1],n_cop]) #kkk_fin

    if norm_flag == True:
        err = tf.math.reduce_sum(pd_grid**2 * step_s,[0,1]) - 2 *tf.math.reduce_mean(kkk_fin,0)

    else:
        pd_grid = ker_grid_all / (tf.math.reduce_sum(ker_grid_all * step_s,0))
        err = tf.math.reduce_sum(pd_grid**2 * step_s,0) - 2 *tf.math.reduce_mean(kkk_fin,0)

    ### Put to err +- 0.001*err if out of bounds
    ind_err = tf.where(tf.math.logical_or(tf.math.less_equal(a,1e-4),tf.math.greater_equal(a,2)))
    ind_err = tf.cast(ind_err,tf.int32)
    new_err = tf.TensorArray(grid_x.dtype,size=tf.shape(err)[0])
    for ind in tf.range(0,tf.shape(err)[0],1,tf.int32):
        if tf.math.reduce_any(tf.math.equal(ind,ind_err)):
            if tf.math.sign(err[ind]) > 0:
                new_err_tmp = err[ind][...,tf.newaxis]+err[ind][...,tf.newaxis]*0.001 #tf.constant([0.1],grid_x.dtype)
            else:
                new_err_tmp = err[ind][...,tf.newaxis]-err[ind][...,tf.newaxis]*0.001
        else:
            new_err_tmp = err[ind][...,tf.newaxis]
        new_err = new_err.write(ind,new_err_tmp)
    new_err = tf.squeeze(new_err.stack())
    err = new_err
    if tf.shape(tf.shape(err)) == 0:
        err = err[...,tf.newaxis]
    return err


# File: src/DVC_tensorflow/optim/.ipynb_checkpoints/bandwidth-checkpoint.py
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

# File: src/DVC_tensorflow/optim/.ipynb_checkpoints/local_lik-checkpoint.py
import tensorflow as tf
import math as m
from utils.tensor_op import replace_nan_inf

########################## COMPUTE LOCAL LIKELIHOOD ################################

@tf.function(experimental_relax_shapes=True)
def loclik_batch(B, data, grid_points, n_cop, batch_size):
#     tf_dtype = B.dtype
#     B = tf.cast(B,tf.float64)
#     data = tf.cast(data,tf.float64)
#     grid_points = tf.cast(grid_points,tf.float64)
    
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
    
    #gr1_tile = tf.tile(grid_point,[d_n, 1, 1]) #[:,0]
    #gr1_tile = tf.reshape(gr1_tile, [d_n, tf.shape(grid_point)[0], 2, 2])
    
    gr1_tile = tf.reshape(grid_point,[1, tf.shape(grid_point)[0],  tf.shape(grid_point)[1], d])
    gr1_tile = tf.tile(gr1_tile,[tf.shape(data_p)[0], 1, 1, 1])
    
    d1_tile = tf.reshape(data_p,[tf.shape(data_p)[0], 1,  tf.shape(grid_point)[1], d])
    #d2_tile = tf.tile(data_x[:,0][...,tf.newaxis],[1, tf.shape(grid_point)[0]])
    d1_tile = tf.tile(d1_tile,[1, tf.shape(grid_point)[0], 1, 1])
    
    #d1_tile = tf.tile(data_p, [d_n, 1, 1])
    
    c = gr1_tile - d1_tile
    
    d_n = tf.cast(d_n,data_p.dtype)  #64
    pi = tf.cast(m.pi,data_p.dtype)  #64
    
    #a1 = tf.exp(-(c[:,:,0,:]**2) / (2*B[0,:]**2))
    #a2 = tf.exp(-(c[:,:,1,:]**2) / (2*B[1,:]**2))
    #a = (a1*a2)/(2*pi*B[0,:]*B[1,:]*d_n)

    a = tf.exp(-(c[:,:,0,:]**2) / (2*B[0,:]**2)) * tf.exp(-(c[:,:,1,:]**2) / (2*B[1,:]**2)) / (2*pi*B[0,:]*B[1,:]*d_n) 
    
    ker_grid1 = tf.math.reduce_sum(a, 0)
    ker_grid2 = tf.math.reduce_sum(a*c[:,:,0,:], 0)
    ker_grid3 = tf.math.reduce_sum(a*c[:,:,1,:], 0)
    ker_grid4 = tf.math.reduce_sum(a*c[:,:,0,:]**2, 0)
    ker_grid5 = tf.math.reduce_sum(a*c[:,:,1,:]**2, 0)
    return ker_grid1,ker_grid2,ker_grid3,ker_grid4,ker_grid5 #ker_grid_fin

# File: src/DVC_tensorflow/optim/.ipynb_checkpoints/nadam-checkpoint.py
import tensorflow as tf

from utils.tensor_op import check_bound3, replace_nan_inf
from optim.MISE import MISE_mul

############################# NADAM OPTIMIZATION #################################

@tf.function(experimental_relax_shapes=True)
def fit_ban(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag, pos_trace, max_iter, convergence_tol, lr):
    eps = 1e-6
    
    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace = tf.ones([n_cop],bw.dtype)

    err = err_trace + 10*convergence_tol
    
    err = MISE_mul(pos_trace, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
#     print('pos_trace:',pos_trace.numpy())
#     print('err:',err.numpy())
#     print('  ')
#     print('band:',a.numpy())
    m = tf.zeros(tf.shape(bw)[1],bw.dtype)
    #print(m)
    v = tf.zeros(tf.shape(bw)[1],bw.dtype)
    m_hat = tf.zeros(tf.shape(bw)[1],bw.dtype)
    v_hat = tf.zeros(tf.shape(bw)[1],bw.dtype)
    beta_1 = tf.constant(0.9,bw.dtype)
    beta_2 = tf.constant(0.999,bw.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):
#         err_trace.assign(err)
        err_trace = err
        err_trace = tf.reshape(err_trace, [tf.shape(bw)[1]])
        
        err = MISE_mul(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
        
        grad = (err - err_trace)/(a-pos_trace)
        grad = replace_nan_inf(grad)
        ###if tf.math.logical_or(tf.math.is_nan(grad),tf.math.is_inf(grad)):
        ###    grad = tf.constant(-0.001,tf.float64)
        
#         pos_trace.assign(a)
        pos_trace = a
        
#         print('err:',err.numpy())
#         print('grad:',grad.numpy())
        iter1 = tf.cast(iter_err,bw.dtype)
        #print(beta_1 * m + (1 - beta_1) * grad)
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, [tf.shape(bw)[1]])
        ###print('m',m.numpy())
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, [tf.shape(bw)[1]])
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)
        ###print('diff:',diff.numpy())
        
#         a.assign(a + diff)
        
        #### To be sure that bw does not go lower than 5e-3
        bw_n = tf.abs(a*bw)
        ind = tf.where(tf.math.less(bw_n[1,:],tf.constant(1e-2,bw.dtype)))  ##It was 5e-3 but too low
        if tf.shape(ind)[0] > 0:
            bu1 = tf.tile(tf.constant([5e-3],bw.dtype),[tf.shape(ind)[0]])
            gat = tf.gather_nd(bw[1,:],ind)
            aa1 = bu1/gat
            a = tf.tensor_scatter_nd_update(a,ind,aa1)
        
        a = a + diff
        
        a_new = check_bound3(a,tf.constant(4,bw.dtype),tf.constant(1e-2,bw.dtype))  ##It was 5e-3 but too low
        a = a_new
        
#         a.assign(a_new)
        a = tf.reshape(a, [tf.shape(bw)[1]])

#         print('   ')
#         print('iter:',iter_err.numpy())
#         print('band:',a.numpy())
        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)

@tf.function(experimental_relax_shapes=True)
def fit_banLL2(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag, pos_trace, max_iter, convergence_tol, lr):
    eps = 1e-6

    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace_x1y = tf.ones([n_cop],bw.dtype)
    err_trace_xy1 = tf.ones([n_cop],bw.dtype)
    
    err = err_trace_x1y + 10*convergence_tol
    
    m = tf.zeros(tf.shape(bw),bw.dtype)
    #print(m)
    v = tf.zeros(tf.shape(bw),bw.dtype)
    m_hat = tf.zeros(tf.shape(bw),bw.dtype)
    v_hat = tf.zeros(tf.shape(bw),bw.dtype)
    beta_1 = tf.constant(0.9,bw.dtype)
    beta_2 = tf.constant(0.999,bw.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),
                              tf.math.logical_or(
                                  tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_x1y),convergence_tol)),
                                  tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_xy1),convergence_tol))
                              )
                             ):
        
#         print('pos_trace:',pos_trace.numpy())
#         print('a:',a.numpy())
        
        x1y = tf.concat([tf.gather(pos_trace,[0]),tf.gather(a,[1])],0) #tf.concat([pos_trace[0],a[1]],0)
        xy1 = tf.concat([tf.gather(a,[0]),tf.gather(pos_trace,[1])],0) #tf.concat([a[0],pos_trace[1]],0)
#         print('x1y:',x1y.numpy())
#         print('xy1:',xy1.numpy())
        err_x1y = MISE_mul(x1y, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
        err_xy1 = MISE_mul(xy1, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
#         print('err_x1y:',err_x1y.numpy())
#         print('err_xy1:',err_xy1.numpy())
#         print('  ')
#         print('band:',a.numpy())

        err_trace_x1y = err_x1y
        err_trace_xy1 = err_xy1
        err_trace_x1y = tf.reshape(err_trace_x1y, [tf.shape(bw)[1]])
        err_trace_xy1 = tf.reshape(err_trace_xy1, [tf.shape(bw)[1]])
        
        err = MISE_mul(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
        
        err = tf.reshape(err, [tf.shape(bw)[1]])
#         print('err:',err.numpy())
        
        grad_x1y = (err - err_trace_x1y)/(tf.gather(a,[0])-tf.gather(x1y,[0]))
        grad_xy1 = (err - err_trace_xy1)/(tf.gather(a,[1])-tf.gather(xy1,[1]))
#         print('grad_x1y:',grad_x1y) #.numpy())
#         print('grad_xy1:',grad_xy1) #.numpy())
        if tf.shape(tf.shape(grad_x1y)) == 1:
            grad_x1y[...,tf.newaxis]
            grad_xy1[...,tf.newaxis]
            
        grad = tf.concat([grad_x1y,grad_xy1],0)
        grad = replace_nan_inf(grad)

        pos_trace = a
        
#         print('grad:',grad.numpy())
        iter1 = tf.cast(iter_err,bw.dtype)
        
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(bw))
#         print('m',m.numpy())
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(bw))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        
        a_new = check_bound3(a,tf.constant(2,bw.dtype),tf.constant(1e-2,bw.dtype))  ##It was 5e-3 but too low

        a = a_new
        a = tf.reshape(a, tf.shape(bw))

#         print('----------------------------------------')
#         print('iter:',iter_err.numpy())
        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.logical_or(
                                  tf.math.reduce_any(tf.math.less(tf.abs(err-err_trace_x1y),convergence_tol)),
                                  tf.math.reduce_any(tf.math.less(tf.abs(err-err_trace_xy1),convergence_tol))
                              )

# File: src/DVC_tensorflow/optim/.ipynb_checkpoints/vine_fit-checkpoint.py
import tensorflow as tf
from utils.prob_op import biv_norm
from optim.bandwidth import bandwidth_mul
from optim.nadam import fit_ban
from optim.nadam import fit_banLL2
from sklearn.model_selection import KFold
from time import perf_counter
from utils.dataset_op import *
from param.copula_fit import *


# def optimization(var_dict,grid_dict, data_dict,par_dict):
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
    
    print('opt1',optim1)
    
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
    print('opt2',optim2)
    
    time_fit = perf_counter() - start_time
    print('time_fit2:', time_fit)
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

# def parametric_fit(u, families):
#     u = tf.convert_to_tensor(u)
#     u = check_bound3(u,tf.constant(1-1e-7,u.dtype),tf.constant(-1+1e-7,u.dtype))

#     theta = []
#     logp = []
#     aic = []
#     for fam in families:
        
#         if fam == 'ind':
#             theta_est = []
#             theta.append(theta_est)
#             p = tf.constant(1,u.dtype)
#             err = -tf.math.reduce_sum(tf.math.log(p))
#             logp.append(err)
#             aic1 = 2*tf.cast(tf.size(theta_est),u.dtype) + 2*err
#             aic.append(aic1.numpy())
        
#         if fam == 'gaussian':
#             pos_trace = tf.constant([0.5], dtype = u.dtype)
#             lr = tf.constant(0.005,u.dtype)
#             conv_tol = tf.constant(1e-3,u.dtype)
#             max_iter = tf.constant(100,tf.int32)
#             a = pos_trace + lr

#             theta_est, err, n_iter, conv_flag = fit_gaussian(u, a, pos_trace, conv_tol, lr, max_iter)
#             theta.append(theta_est.numpy())
#             logp.append(err)
#             aic1 = 2*tf.cast(tf.size(theta_est),theta_est.dtype) + 2*err
#             aic.append(aic1.numpy())

#         if fam == 'student':
#             pos_trace = tf.constant([0.5,3], dtype = u.dtype)
#             lr = tf.constant(0.1,u.dtype)
#             conv_tol = tf.constant(1e-2,u.dtype)
#             max_iter = tf.constant(100,tf.int32)
#             a = pos_trace + lr

#             theta_est, err, n_iter, conv_flag = fit_student(u, a, pos_trace, conv_tol, lr, max_iter)
#             theta.append(theta_est.numpy())
#             logp.append(err)
#             aic1 = 2*tf.cast(tf.size(theta_est),theta_est.dtype) + 2*err
#             aic.append(aic1[0].numpy())

#         if (fam == 'clayton') | (fam == 'claytonrot90'):
#             pos_trace = tf.constant([3], dtype = u.dtype)
#             lr = tf.constant(0.2,u.dtype)
#             conv_tol = tf.constant(1e-2,u.dtype)
#             max_iter = tf.constant(100,tf.int32)
#             a = pos_trace + lr

#             if fam == 'clayton':
#                 theta_est, err, n_iter, conv_flag = fit_clayton(u, a, pos_trace, conv_tol, lr, max_iter)
#             if fam == 'claytonrot90':
#                 theta_est, err, n_iter, conv_flag = fit_claytonrot90(u, a, pos_trace, conv_tol, lr, max_iter)
#             theta.append(theta_est.numpy())
#             logp.append(err)
#             aic1 = 2*tf.cast(tf.size(theta_est),theta_est.dtype) + 2*err
#             aic.append(aic1[0].numpy())
#     return aic, theta, logp

# File: src/DVC_tensorflow/optim/MISE.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from evalu.cop_eval import eval_rs_p
from optim.local_lik import loclik_batch

############################## MISE COST FUNCTION ################################

#@tf.function(experimental_relax_shapes=True)
def MISE_mul(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, all_x, data_x_train, data_s_test, n_cop, batch_size, NORM1, norm_flag):
    if tf.math.equal(tf.shape(tf.shape(all_x)),2):
        all_x = all_x[...,tf.newaxis]
    
    bw1 = tf.abs(a*bw)
    n_splits = tf.shape(data_x_train)[3]  

    if tf.math.equal(tf.shape(tf.shape(grid_x)),2):
        grid_x = grid_x[...,tf.newaxis]

    ker_grid_all = loclik_batch(bw1, all_x, grid_x, n_cop, batch_size) #data_x
    if norm_flag == True:
        ker_grid_all = tf.transpose(tf.reshape(ker_grid_all,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop]),perm=[1, 0, 2])
        pd_grid = eval_rs_p(adu11, adu22, ker_grid_all, NORM1, n_cop)
    else:
        pd_grid = tf.zeros([tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop],bw.dtype)
        
    kkk_fin = tf.TensorArray(grid_x.dtype,size=n_splits)
    for k in tf.range(0,n_splits,1,tf.int32):
        ker_grid_fin = loclik_batch(bw1, data_x_train[:,:,:,k], grid_x, n_cop, batch_size)
        pd_grid1 = tf.transpose(tf.reshape(ker_grid_fin,[tf.shape(adu11)[0], tf.shape(adu11)[0], n_cop]),perm=[1, 0, 2])
        if norm_flag == True:
            pd_grid1 = eval_rs_p(adu11, adu22, pd_grid1, NORM1, n_cop)
            
        interp_data = tf.TensorArray(data_x_train.dtype, size=n_cop)
        for kk in tf.range(0,n_cop,1,tf.int32):
            interp_data1 = tfp.math.batch_interp_regular_nd_grid(data_s_test[:,:,kk,k], min_s, max_s, pd_grid1[:,:,kk], axis=-2)
            interp_data = interp_data.write(kk,interp_data1)
        interp_data = interp_data.stack()
        interp_data = tf.transpose(interp_data)
        if norm_flag == True:
            interp_data = interp_data / tf.math.reduce_sum(pd_grid * step_s,[0,1])
        else:
            interp_data = interp_data / tf.math.reduce_sum(ker_grid_fin * step_s,0)
        kkk_fin = kkk_fin.write(k,interp_data)
    kkk_fin = kkk_fin.stack()
    kkk_fin = tf.reshape(kkk_fin,[tf.shape(kkk_fin)[0]*tf.shape(kkk_fin)[1],n_cop]) #kkk_fin

    if norm_flag == True:
        err = tf.math.reduce_sum(pd_grid**2 * step_s,[0,1]) - 2 *tf.math.reduce_mean(kkk_fin,0)

    else:
        pd_grid = ker_grid_all / (tf.math.reduce_sum(ker_grid_all * step_s,0))
        err = tf.math.reduce_sum(pd_grid**2 * step_s,0) - 2 *tf.math.reduce_mean(kkk_fin,0)

    ### Put to err +- 0.001*err if out of bounds
    ind_err = tf.where(tf.math.logical_or(tf.math.less_equal(a,1e-4),tf.math.greater_equal(a,2)))
    ind_err = tf.cast(ind_err,tf.int32)
    new_err = tf.TensorArray(grid_x.dtype,size=tf.shape(err)[0])
    for ind in tf.range(0,tf.shape(err)[0],1,tf.int32):
        if tf.math.reduce_any(tf.math.equal(ind,ind_err)):
            if tf.math.sign(err[ind]) > 0:
                new_err_tmp = err[ind][...,tf.newaxis]+err[ind][...,tf.newaxis]*0.001 #tf.constant([0.1],grid_x.dtype)
            else:
                new_err_tmp = err[ind][...,tf.newaxis]-err[ind][...,tf.newaxis]*0.001
        else:
            new_err_tmp = err[ind][...,tf.newaxis]
        new_err = new_err.write(ind,new_err_tmp)
    new_err = tf.squeeze(new_err.stack())
    err = new_err
    if tf.shape(tf.shape(err)) == 0:
        err = err[...,tf.newaxis]
    return err


# File: src/DVC_tensorflow/optim/bandwidth.py
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

# File: src/DVC_tensorflow/optim/local_lik.py
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

# File: src/DVC_tensorflow/optim/nadam.py
import tensorflow as tf

from utils.tensor_op import check_bound3, replace_nan_inf
from optim.MISE import MISE_mul

############################# NADAM OPTIMIZATION #################################

#@tf.function(experimental_relax_shapes=True)
def fit_ban(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag, pos_trace, max_iter, convergence_tol, lr):
    eps = 1e-6
    
    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace = tf.ones([n_cop],bw.dtype)

    err = err_trace + 10*convergence_tol
    
    err = MISE_mul(pos_trace, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)

    m = tf.zeros(tf.shape(bw)[1],bw.dtype)
    v = tf.zeros(tf.shape(bw)[1],bw.dtype)
    m_hat = tf.zeros(tf.shape(bw)[1],bw.dtype)
    v_hat = tf.zeros(tf.shape(bw)[1],bw.dtype)
    beta_1 = tf.constant(0.9,bw.dtype)
    beta_2 = tf.constant(0.999,bw.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):
        err_trace = err
        err_trace = tf.reshape(err_trace, [tf.shape(bw)[1]])
        
        err = MISE_mul(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
        
        grad = (err - err_trace)/(a-pos_trace)
        grad = replace_nan_inf(grad)

        pos_trace = a
        iter1 = tf.cast(iter_err,bw.dtype)

        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, [tf.shape(bw)[1]])
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, [tf.shape(bw)[1]])
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)
        
        #### To be sure that bw does not go lower than 5e-3
        bw_n = tf.abs(a*bw)
        ind = tf.where(tf.math.less(bw_n[1,:],tf.constant(1e-2,bw.dtype)))  ##It was 5e-3 but too low
        if tf.shape(ind)[0] > 0:
            bu1 = tf.tile(tf.constant([5e-3],bw.dtype),[tf.shape(ind)[0]])
            gat = tf.gather_nd(bw[1,:],ind)
            aa1 = bu1/gat
            a = tf.tensor_scatter_nd_update(a,ind,aa1)
        
        a = a + diff
        
        a_new = check_bound3(a,tf.constant(4,bw.dtype),tf.constant(1e-2,bw.dtype))  ##It was 5e-3 but too low
        a = a_new
        
        a = tf.reshape(a, [tf.shape(bw)[1]])

        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)

@tf.function(experimental_relax_shapes=True)
def fit_banLL2(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag, pos_trace, max_iter, convergence_tol, lr):
    eps = 1e-6

    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace_x1y = tf.ones([n_cop],bw.dtype)
    err_trace_xy1 = tf.ones([n_cop],bw.dtype)
    
    err = err_trace_x1y + 10*convergence_tol
    
    m = tf.zeros(tf.shape(bw),bw.dtype)
    v = tf.zeros(tf.shape(bw),bw.dtype)
    m_hat = tf.zeros(tf.shape(bw),bw.dtype)
    v_hat = tf.zeros(tf.shape(bw),bw.dtype)
    beta_1 = tf.constant(0.9,bw.dtype)
    beta_2 = tf.constant(0.999,bw.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),
                              tf.math.logical_or(
                                  tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_x1y),convergence_tol)),
                                  tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_xy1),convergence_tol))
                              )
                             ):
        
        x1y = tf.concat([tf.gather(pos_trace,[0]),tf.gather(a,[1])],0)
        xy1 = tf.concat([tf.gather(a,[0]),tf.gather(pos_trace,[1])],0) 
        err_x1y = MISE_mul(x1y, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
        err_xy1 = MISE_mul(xy1, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)

        err_trace_x1y = err_x1y
        err_trace_xy1 = err_xy1
        err_trace_x1y = tf.reshape(err_trace_x1y, [tf.shape(bw)[1]])
        err_trace_xy1 = tf.reshape(err_trace_xy1, [tf.shape(bw)[1]])
        
        err = MISE_mul(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
        
        err = tf.reshape(err, [tf.shape(bw)[1]])
        
        grad_x1y = (err - err_trace_x1y)/(tf.gather(a,[0])-tf.gather(x1y,[0]))
        grad_xy1 = (err - err_trace_xy1)/(tf.gather(a,[1])-tf.gather(xy1,[1]))

        if tf.shape(tf.shape(grad_x1y)) == 1:
            grad_x1y[...,tf.newaxis]
            grad_xy1[...,tf.newaxis]
            
        grad = tf.concat([grad_x1y,grad_xy1],0)
        grad = replace_nan_inf(grad)

        pos_trace = a
        
        iter1 = tf.cast(iter_err,bw.dtype)
        
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(bw))
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(bw))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        
        a_new = check_bound3(a,tf.constant(2,bw.dtype),tf.constant(1e-2,bw.dtype))  ##It was 5e-3 but too low

        a = a_new
        a = tf.reshape(a, tf.shape(bw))

        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.logical_or(
                                  tf.math.reduce_any(tf.math.less(tf.abs(err-err_trace_x1y),convergence_tol)),
                                  tf.math.reduce_any(tf.math.less(tf.abs(err-err_trace_xy1),convergence_tol))
                              )

# File: src/DVC_tensorflow/optim/vine_fit.py
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


# File: src/DVC_tensorflow/param/.ipynb_checkpoints/cond_copula-checkpoint.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from scipy import stats

from utils.bijector import *
from param.margin_pdf import *

############################################ COPULA PDF      ############################################

def copulapdf(vine_par,u):
    c = np.zeros(np.shape(u)[0],u.dtype)
    u = tf.convert_to_tensor(u)
    
    if vine_par.family == 'ind':
        c = tf.ones(tf.shape(u)[0],u.dtype)
    elif vine_par.family == 'gaussian':
        theta = tf.convert_to_tensor(vine_par.theta,u.dtype)
        c = gaussian_pdf(u,theta)
    if vine_par.family == 'student':
        theta = tf.convert_to_tensor(vine_par.theta,u.dtype)
        c = student_pdf(u,theta)
    if vine_par.family == 'clayton':
        theta = tf.convert_to_tensor(vine_par.theta,u.dtype)
        c = clayton_pdf(u,theta)
    if vine_par.family == 'claytonrot90':
        theta = tf.convert_to_tensor(vine_par.theta,u.dtype)
        c = claytonrot90_pdf(u,theta)
    return c

############################################ COPULA CONDITIONED CDF      ############################################

def copulaccdf(vine_par,u):
    loc = 0
    scale = 1
    
    u[u>=1-1e-7] = 1-1e-7
    u[u<=1e-7] = +1e-7
    
    c = np.zeros(np.shape(u)[0],u.dtype)
    if vine_par.family == 'ind':
        c = u[:,0]
    elif vine_par.family == 'gaussian':
        x = NormalCDF.forward(NormalCDF(loc,scale), u)
        theta = vine_par.theta
        tmp = (x[:,0] - theta * x[:,1]) / np.sqrt(1-theta**2)
        c = NormalCDF.inverse(NormalCDF(loc,scale), tmp)
    elif vine_par.family == 'student':
        theta1 = vine_par.theta[0]
        theta2 = vine_par.theta[1]
        x = stats.t.ppf(u, theta2, loc, scale)
#         print('x',x)
        tmp = np.sqrt((theta2+1) / (theta2+x[:,1]**2)) * (x[:,0] - theta1 * x[:,1]) / (np.sqrt(1-theta1**2)) #, theta[1]+1)
#         print('tmp',tmp)
        c = stats.t.cdf(tmp, theta2+1, loc, scale)
    elif vine_par.family == 'clayton':   
        theta = vine_par.theta
        if theta == 0:
            c = u[:,0]
        else:
            c = np.maximum(u[:,1]**(-1-theta) * (u[:,0]**(-theta) + u[:,1]**(-theta) - 1) ** (-1-1/theta),0)
    elif vine_par.family == 'claytonrot90':   
        theta = vine_par.theta
        if theta == 0:
            c = u[:,0]
        else:
            c = np.maximum((1-u[:,1])**(-1-theta) * (u[:,0]**(-theta) + (1-u[:,1])**(-theta) - 1) ** (-1-1/theta),0)
    
#     ind = np.where(np.isnan(c))
#     for ii in range(0,len(ind),1):
#         if len(ind) > 1:
#             print(ind[ii])
#         print(u[ind[ii],:])
#     c = replace_nan2(c)
    return c

############################################ COPULA INVERSE CONDITIONED CDF      ############################################

def copulainvccdf(vine_par,u):
    loc = 0
    scale = 1
    
    u[u>=1-1e-7] = 1-1e-7
    u[u<=1e-7] = +1e-7
    c = np.zeros(np.shape(u)[0],u.dtype)
    
    if vine_par.family == 'ind':
        c = u[:,0]
    elif vine_par.family == 'gaussian':
        x = NormalCDF.forward(NormalCDF(loc,scale), u)
        theta = vine_par.theta
        tmp = x[:,0] * np.math.sqrt(1-theta**2) + theta * x[:,1]        
        c = NormalCDF.inverse(NormalCDF(loc,scale), tmp)
    if vine_par.family == 'student':
        theta1 = vine_par.theta[0]
        theta2 = vine_par.theta[1]
        x = stats.t.ppf(u, theta2, loc, scale)
        param = theta2 + 1 
        tmp_inv = stats.t.ppf(u[:,0], param, loc, scale)
        tmp = np.sqrt( ((1-theta1**2) * (theta2 + x[:,1]**2)) / (theta2+1) ) * tmp_inv + theta1 * x[:,1]
        c = stats.t.cdf(tmp, theta2, loc, scale)
    if vine_par.family == 'clayton':
        theta = vine_par.theta
        if theta == 0:
            c = u[:,0]
        else:           
            c = (1 - u[:,1]**(-theta) + (u[:,0] * (u[:,1]**(1+theta)))**(-theta/(1+theta)))**(-1/theta)
    if vine_par.family == 'claytonrot90':
        theta = vine_par.theta
        if theta == 0:
            c = u[:,0]
        else:           
            c = (1 - (1 - u[:,1]) **(-theta) + (u[:,0] * ((1 - u[:,1])**(1+theta)))**(-theta/(1+theta)))**(-1/theta)
#     c = replace_nan2(c)

#     ind = np.where(np.isnan(c))
#     for ii in range(0,len(ind),1):
#         if len(ind) > 1:
#             print(ind[ii])
#         print(u[ind[ii],:])
    return c

# File: src/DVC_tensorflow/param/.ipynb_checkpoints/copula_fit-checkpoint.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from utils.tensor_op import check_bound3, replace_nan_inf
from param.margin_cost import *

################################# GAUSSIAN FITTING ###############################################

@tf.function(experimental_relax_shapes=True)
def fit_gaussian(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    eps = tf.constant(1e-6,a.dtype)
    
    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace = tf.ones([n_cop],u.dtype)
    err = err_trace + 10*convergence_tol
    
#     print('pos',pos_trace)
#     print('u',u)
    
#     norm_dis = tfd.Normal(loc=tf.constant(0,u.dtype), scale=tf.constant(1,u.dtype))
    err = gaussian_cost(u, pos_trace)
    
#     print('err',err)
    
    m = tf.zeros(tf.shape(a),u.dtype)
    v = tf.zeros(tf.shape(a),u.dtype)
    m_hat = tf.zeros(tf.shape(a),u.dtype)
    v_hat = tf.zeros(tf.shape(a),u.dtype)
    beta_1 = tf.constant(0.9,u.dtype)
    beta_2 = tf.constant(0.999,u.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):
#         err_trace.assign(err)
        err_trace = err
#         print('err_trace', err_trace)
        err_trace = tf.reshape(err_trace, [n_cop])
        
        err = gaussian_cost(u, a)
        
        grad = (err - err_trace)/(a-pos_trace)
        grad = replace_nan_inf(grad)

        pos_trace = a
        

        iter1 = tf.cast(iter_err,u.dtype)
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(a))
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(a))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        a_new = check_bound3(a,tf.constant(1-1e-3,u.dtype),tf.constant(0+1e-3,u.dtype))
        a = a_new
        a = tf.reshape(a, tf.shape(a))
        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)

# @tf.function
# def fit_gaussian(u, a, pos_trace, convergence_tol, lr, max_iter):
#     eps = 1e-6
    
#     iter_err = tf.constant(1,dtype=tf.int32)
#     err_trace = tf.ones([1],u.dtype)
#     err = err_trace + 10*convergence_tol
    
#     norm_dis = tfd.Normal(loc=tf.constant(0,u.dtype), scale=tf.constant(1,u.dtype))
#     err = gaussian_cost(u,pos_trace, norm_dis)

#     m = tf.zeros(tf.shape(a),u.dtype)
#     v = tf.zeros(tf.shape(a),u.dtype)
#     m_hat = tf.zeros(tf.shape(a),u.dtype)
#     v_hat = tf.zeros(tf.shape(a),u.dtype)
#     beta_1 = tf.constant(0.9,u.dtype)
#     beta_2 = tf.constant(0.999,u.dtype)

#     while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):
# #         err_trace.assign(err)
#         err_trace = err
#         err_trace = tf.reshape(err_trace, [1])
        
#         err = gaussian_cost(u, a, norm_dis)
        
#         grad = (err - err_trace)/(a-pos_trace)
#         grad = replace_nan_inf(grad)

#         pos_trace = a
        

#         iter1 = tf.cast(iter_err,u.dtype)
#         m = beta_1 * m + (1 - beta_1) * grad
#         m = tf.reshape(m, tf.shape(a))
#         v = beta_2 * v + (1 - beta_2) * grad**2
#         v = tf.reshape(v, tf.shape(a))
        
#         m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
#         v_hat = v / (1 - beta_2**iter1)
#         diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

#         a = a + diff
#         a_new = check_bound3(a,tf.constant(1,u.dtype),tf.constant(0,u.dtype))
#         a = a_new
#         a = tf.reshape(a, tf.shape(a))
#         iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
#     return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)


################################# STUDENT FITTING ###############################################

# @tf.function(experimental_relax_shapes=True)
def fit_student(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    eps = 1e-6

    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace_x1y = tf.ones([n_cop],u.dtype)
    err_trace_xy1 = tf.ones([n_cop],u.dtype)
    
    err = err_trace_x1y + 10*convergence_tol
    
    m = tf.zeros(tf.shape(a),u.dtype)
    #print(m)
    v = tf.zeros(tf.shape(a),u.dtype)
    m_hat = tf.zeros(tf.shape(a),u.dtype)
    v_hat = tf.zeros(tf.shape(a),u.dtype)
    beta_1 = tf.constant(0.9,u.dtype)
    beta_2 = tf.constant(0.999,u.dtype)
    

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),
#                               tf.math.logical_or(
#                                 tf.equal( tf.shape(tf.where(tf.equal(tf.math.greater(tf.abs(err-err_trace_x1y),convergence_tol),True)))[0] , n_cop),
#                                 tf.equal( tf.shape(tf.where(tf.equal(tf.math.greater(tf.abs(err-err_trace_xy1),convergence_tol),True)))[0] , n_cop)
#                                                 )
                              tf.math.logical_or(
                                  tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_x1y),convergence_tol)),
                                  tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_xy1),convergence_tol))
                              )
                             ):
        
#         print('pos_trace:',pos_trace.numpy())
#         print('a:',a.numpy())
        
#         x1y = tf.concat([tf.gather(pos_trace,[0]),tf.gather(a,[1])],0) #tf.concat([pos_trace[0],a[1]],0)
#         xy1 = tf.concat([tf.gather(a,[0]),tf.gather(pos_trace,[1])],0) #tf.concat([a[0],pos_trace[1]],0)

    
        x1y = tf.concat([pos_trace[:,0][...,tf.newaxis],a[:,1][...,tf.newaxis]],1)
        xy1 = tf.concat([a[:,0][...,tf.newaxis],pos_trace[:,1][...,tf.newaxis]],1)
        
#         print('x1y:',x1y.numpy())
#         print('xy1:',xy1.numpy())
        err_x1y = student_cost(u, x1y)
        err_xy1 = student_cost(u, xy1)
#         print('err_x1y:',err_x1y.numpy())
#         print('err_xy1:',err_xy1.numpy())
#         print('  ')
#         print('band:',a.numpy())

        err_trace_x1y = err_x1y
        err_trace_xy1 = err_xy1
        err_trace_x1y = tf.reshape(err_trace_x1y, [n_cop])
        err_trace_xy1 = tf.reshape(err_trace_xy1, [n_cop])
        
        err = student_cost(u, a)
        
        err = tf.reshape(err, [n_cop])
#         print('err:',err.numpy())
        
#         grad_x1y = (err - err_trace_x1y)/(tf.gather(a,[0])-tf.gather(x1y,[0]))
#         grad_xy1 = (err - err_trace_xy1)/(tf.gather(a,[1])-tf.gather(xy1,[1]))
        
        grad_x1y = (err - err_trace_x1y)/(a[:,0]-x1y[:,0])
        grad_xy1 = (err - err_trace_xy1)/(a[:,1]-xy1[:,1])
#         print('grad_x1y:',grad_x1y) #.numpy())
#         print('grad_xy1:',grad_xy1) #.numpy())
        if tf.shape(tf.shape(grad_x1y)) == 1:
            grad_x1y[...,tf.newaxis]
            grad_xy1[...,tf.newaxis]
            
        grad = tf.concat([grad_x1y[...,tf.newaxis],grad_xy1[...,tf.newaxis]],1)
#         print('grad',grad)
        grad = replace_nan_inf(grad)

        pos_trace = a
        
#         print('grad:',grad.numpy())
        iter1 = tf.cast(iter_err,u.dtype)
        
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(a))
#         print('m',m.numpy())
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(a))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        
#         a_new1 = check_bound3([a[0]],tf.constant(1,u.dtype),tf.constant(-1,u.dtype))
#         a_new2 = check_bound3([a[1]],tf.constant(1000,u.dtype),tf.constant(1e-3,u.dtype))
        a_new1 = check_bound3(a[:,0][...,tf.newaxis],tf.constant(1,u.dtype),tf.constant(-1,u.dtype))
        a_new2 = check_bound3(a[:,1][...,tf.newaxis],tf.constant(1000,u.dtype),tf.constant(1e-3,u.dtype))
        a_new1 = tf.reshape(a_new1,[n_cop,1])
        a_new2 = tf.reshape(a_new2,[n_cop,1])
#         print('a1',a_new1)
#         a_new = tf.concat([a_new1,a_new2],0)
        a_new = tf.concat([a_new1,a_new2],1)
#         print('aa',a_new)
#         a_new = tf.reshape(a_new, tf.shape(a))
        a = a_new
        a = tf.reshape(a, [n_cop,2]) # tf.shape(a))

#         print('----------------------------------------')
#         print('iter:',iter_err.numpy())
        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
        
    return a, err, iter_err, tf.math.logical_or(
                                tf.equal( tf.shape(tf.where(tf.equal(tf.math.less(tf.abs(err-err_trace_x1y),convergence_tol),True)))[0] , n_cop),
                                tf.equal( tf.shape(tf.where(tf.equal(tf.math.less(tf.abs(err-err_trace_xy1),convergence_tol),True)))[0] , n_cop)
                                                )

# @tf.function
# def fit_student(u, a, pos_trace, convergence_tol, lr, max_iter):
#     eps = 1e-6

#     iter_err = tf.constant(1,dtype=tf.int32)
#     err_trace_x1y = tf.ones([1],u.dtype)
#     err_trace_xy1 = tf.ones([1],u.dtype)
    
#     err = err_trace_x1y + 10*convergence_tol
    
#     m = tf.zeros(tf.shape(a),u.dtype)
#     #print(m)
#     v = tf.zeros(tf.shape(a),u.dtype)
#     m_hat = tf.zeros(tf.shape(a),u.dtype)
#     v_hat = tf.zeros(tf.shape(a),u.dtype)
#     beta_1 = tf.constant(0.9,u.dtype)
#     beta_2 = tf.constant(0.999,u.dtype)

#     while tf.math.logical_and(tf.math.less(iter_err, max_iter),
#                               tf.math.logical_or(
#                                   tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_x1y),convergence_tol)),
#                                   tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_xy1),convergence_tol))
#                               )
#                              ):
        
# #         print('pos_trace:',pos_trace.numpy())
# #         print('a:',a.numpy())
        
#         x1y = tf.concat([tf.gather(pos_trace,[0]),tf.gather(a,[1])],0) #tf.concat([pos_trace[0],a[1]],0)
#         xy1 = tf.concat([tf.gather(a,[0]),tf.gather(pos_trace,[1])],0) #tf.concat([a[0],pos_trace[1]],0)
# #         print('x1y:',x1y.numpy())
# #         print('xy1:',xy1.numpy())
#         err_x1y = student_cost(u, x1y)
#         err_xy1 = student_cost(u, xy1)
# #         print('err_x1y:',err_x1y.numpy())
# #         print('err_xy1:',err_xy1.numpy())
# #         print('  ')
# #         print('band:',a.numpy())

#         err_trace_x1y = err_x1y
#         err_trace_xy1 = err_xy1
#         err_trace_x1y = tf.reshape(err_trace_x1y, [1])
#         err_trace_xy1 = tf.reshape(err_trace_xy1, [1])
        
#         err = student_cost(u, a)
        
#         err = tf.reshape(err, [1])
# #         print('err:',err.numpy())
        
#         grad_x1y = (err - err_trace_x1y)/(tf.gather(a,[0])-tf.gather(x1y,[0]))
#         grad_xy1 = (err - err_trace_xy1)/(tf.gather(a,[1])-tf.gather(xy1,[1]))
# #         print('grad_x1y:',grad_x1y) #.numpy())
# #         print('grad_xy1:',grad_xy1) #.numpy())
#         if tf.shape(tf.shape(grad_x1y)) == 1:
#             grad_x1y[...,tf.newaxis]
#             grad_xy1[...,tf.newaxis]
            
#         grad = tf.concat([grad_x1y,grad_xy1],0)
#         grad = replace_nan_inf(grad)

#         pos_trace = a
        
# #         print('grad:',grad.numpy())
#         iter1 = tf.cast(iter_err,u.dtype)
        
#         m = beta_1 * m + (1 - beta_1) * grad
#         m = tf.reshape(m, tf.shape(a))
# #         print('m',m.numpy())
#         v = beta_2 * v + (1 - beta_2) * grad**2
#         v = tf.reshape(v, tf.shape(a))
        
#         m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
#         v_hat = v / (1 - beta_2**iter1)
#         diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

#         a = a + diff
        
#         a_new1 = check_bound3([a[0]],tf.constant(1,u.dtype),tf.constant(-1,u.dtype))
#         a_new2 = check_bound3([a[1]],tf.constant(1000,u.dtype),tf.constant(1e-3,u.dtype))
        
#         a_new = tf.concat([a_new1,a_new2],0)
        
#         a = a_new
#         a = tf.reshape(a, tf.shape(a))

# #         print('----------------------------------------')
# #         print('iter:',iter_err.numpy())
#         iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
#     return a, err, iter_err, tf.math.logical_or(
#                                   tf.math.reduce_any(tf.math.less(tf.abs(err-err_trace_x1y),convergence_tol)),
#                                   tf.math.reduce_any(tf.math.less(tf.abs(err-err_trace_xy1),convergence_tol))
#                               )


################################# CLAYTON FITTING ###############################################

@tf.function(experimental_relax_shapes=True)
def fit_clayton(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    eps = 1e-6
    
#     print('0')
    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace = tf.ones([n_cop],u.dtype)
    
#     err_trace = tf.ones(1,u.dtype)
    err = err_trace + 10*convergence_tol
    
    err = clayton_cost(u, pos_trace)
    err = tf.reshape(err, [n_cop])
#     print('err_trace',err.numpy())
#     print('pos_trace',pos_trace.numpy())
    
    m = tf.zeros(tf.shape(a),u.dtype)
    v = tf.zeros(tf.shape(a),u.dtype)
    m_hat = tf.zeros(tf.shape(a),u.dtype)
    v_hat = tf.zeros(tf.shape(a),u.dtype)
    beta_1 = tf.constant(0.9,u.dtype)
    beta_2 = tf.constant(0.999,u.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):
#         err_trace.assign(err)
        err_trace = err
        err_trace = tf.reshape(err_trace, [n_cop])
        
        err = clayton_cost(u, a)
        
#         if tf.math.is_inf(err):
#             sign = tf.math.sign(err)
#             err = err_trace + sign*err_trace*0.1 #u.dtype.max
            
        err = tf.reshape(err, [n_cop])
#         print('err',err.numpy())
#         print('a',a.numpy())
        
        grad = (err - err_trace)/(a-pos_trace)
        grad = replace_nan_inf(grad)
        
#         print('grad',grad)
        
        pos_trace = a
        
        iter1 = tf.cast(iter_err,u.dtype)
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(a))
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(a))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        a_new = check_bound3(a,tf.constant(20,u.dtype),tf.constant(1e-1,u.dtype))
        a = a_new
        a = tf.reshape(a, tf.shape(a))
        
#         print('differr',tf.abs(err-err_trace))
#         print('----------------------------')
        
        
        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)

# @tf.function
# def fit_clayton(u, a, pos_trace, convergence_tol, lr, max_iter):
#     eps = 1e-6
    
# #     print('0')
#     iter_err = tf.constant(1,dtype=tf.int32)
#     err_trace = tf.ones([1],u.dtype)
    
# #     err_trace = tf.ones(1,u.dtype)
#     err = err_trace + 10*convergence_tol
    
#     err = clayton_cost(u, pos_trace)
#     err = tf.reshape(err, [1])
# #     print('err_trace',err.numpy())
# #     print('pos_trace',pos_trace.numpy())
    
#     m = tf.zeros(tf.shape(a),u.dtype)
#     v = tf.zeros(tf.shape(a),u.dtype)
#     m_hat = tf.zeros(tf.shape(a),u.dtype)
#     v_hat = tf.zeros(tf.shape(a),u.dtype)
#     beta_1 = tf.constant(0.9,u.dtype)
#     beta_2 = tf.constant(0.999,u.dtype)

#     while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):
# #         err_trace.assign(err)
#         err_trace = err
#         err_trace = tf.reshape(err_trace, [1])
        
#         err = clayton_cost(u, a)
        
# #         if tf.math.is_inf(err):
# #             sign = tf.math.sign(err)
# #             err = err_trace + sign*err_trace*0.1 #u.dtype.max
            
#         err = tf.reshape(err, [1])
# #         print('err',err.numpy())
# #         print('a',a.numpy())
        
#         grad = (err - err_trace)/(a-pos_trace)
#         grad = replace_nan_inf(grad)
        
# #         print('grad',grad)
        
#         pos_trace = a
        
#         iter1 = tf.cast(iter_err,u.dtype)
#         m = beta_1 * m + (1 - beta_1) * grad
#         m = tf.reshape(m, tf.shape(a))
#         v = beta_2 * v + (1 - beta_2) * grad**2
#         v = tf.reshape(v, tf.shape(a))
        
#         m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
#         v_hat = v / (1 - beta_2**iter1)
#         diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

#         a = a + diff
#         a_new = check_bound3(a,tf.constant(20,u.dtype),tf.constant(1e-1,u.dtype))
#         a = a_new
#         a = tf.reshape(a, tf.shape(a))
        
# #         print('differr',tf.abs(err-err_trace))
# #         print('----------------------------')
        
        
#         iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
#     return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)

################################# CLAYTON ROT 90 FITTING ###############################################

@tf.function(experimental_relax_shapes=True)
def fit_claytonrot90(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    eps = 1e-6
    
#     print('0')
    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace = tf.ones([n_cop],u.dtype)
    
#     err_trace = tf.ones(1,u.dtype)
    err = err_trace + 10*convergence_tol
    
    err = claytonrot90_cost(u, pos_trace)
    err = tf.reshape(err, [n_cop])
#     print('err_trace',err.numpy())
#     print('pos_trace',pos_trace.numpy())
    
    m = tf.zeros(tf.shape(a),u.dtype)
    v = tf.zeros(tf.shape(a),u.dtype)
    m_hat = tf.zeros(tf.shape(a),u.dtype)
    v_hat = tf.zeros(tf.shape(a),u.dtype)
    beta_1 = tf.constant(0.9,u.dtype)
    beta_2 = tf.constant(0.999,u.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):
#         err_trace.assign(err)
        err_trace = err
        err_trace = tf.reshape(err_trace, [n_cop])
        
        err = claytonrot90_cost(u, a)
        
#         if tf.math.is_inf(err):
#             sign = tf.math.sign(err)
#             err = err_trace + sign*err_trace*0.1 #u.dtype.max
            
        err = tf.reshape(err, [n_cop])
#         print('err',err.numpy())
#         print('a',a.numpy())
        
        grad = (err - err_trace)/(a-pos_trace)
        grad = replace_nan_inf(grad)
        
#         print('grad',grad)
        
        pos_trace = a
        
        iter1 = tf.cast(iter_err,u.dtype)
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(a))
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(a))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        a_new = check_bound3(a,tf.constant(20,u.dtype),tf.constant(1e-1,u.dtype))
        a = a_new
        a = tf.reshape(a, tf.shape(a))
        
#         print('differr',tf.abs(err-err_trace))
#         print('----------------------------')
        
        
        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)

# @tf.function
# def fit_claytonrot90(u, a, pos_trace, convergence_tol, lr, max_iter):
#     eps = 1e-6
    
# #     print('0')
#     iter_err = tf.constant(1,dtype=tf.int32)
#     err_trace = tf.ones([1],u.dtype)
    
# #     err_trace = tf.ones(1,u.dtype)
#     err = err_trace + 10*convergence_tol
    
#     err = claytonrot90_cost(u, pos_trace)
#     err = tf.reshape(err, [1])
# #     print('err_trace',err.numpy())
# #     print('pos_trace',pos_trace.numpy())
    
#     m = tf.zeros(tf.shape(a),u.dtype)
#     v = tf.zeros(tf.shape(a),u.dtype)
#     m_hat = tf.zeros(tf.shape(a),u.dtype)
#     v_hat = tf.zeros(tf.shape(a),u.dtype)
#     beta_1 = tf.constant(0.9,u.dtype)
#     beta_2 = tf.constant(0.999,u.dtype)

#     while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):
# #         err_trace.assign(err)
#         err_trace = err
#         err_trace = tf.reshape(err_trace, [1])
        
#         err = claytonrot90_cost(u, a)
        
# #         if tf.math.is_inf(err):
# #             sign = tf.math.sign(err)
# #             err = err_trace + sign*err_trace*0.1 #u.dtype.max
            
#         err = tf.reshape(err, [1])
# #         print('err',err.numpy())
# #         print('a',a.numpy())
        
#         grad = (err - err_trace)/(a-pos_trace)
#         grad = replace_nan_inf(grad)
        
# #         print('grad',grad)
        
#         pos_trace = a
        
#         iter1 = tf.cast(iter_err,u.dtype)
#         m = beta_1 * m + (1 - beta_1) * grad
#         m = tf.reshape(m, tf.shape(a))
#         v = beta_2 * v + (1 - beta_2) * grad**2
#         v = tf.reshape(v, tf.shape(a))
        
#         m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
#         v_hat = v / (1 - beta_2**iter1)
#         diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

#         a = a + diff
#         a_new = check_bound3(a,tf.constant(20,u.dtype),tf.constant(1e-1,u.dtype))
#         a = a_new
#         a = tf.reshape(a, tf.shape(a))
        
# #         print('differr',tf.abs(err-err_trace))
# #         print('----------------------------')
        
        
#         iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
#     return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)

# File: src/DVC_tensorflow/param/.ipynb_checkpoints/generate_rvine-checkpoint.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from scipy import stats

from classes.objects import margin_obj
from param.margin_pdf import *
from param.margin_op import *
from param.cond_copula import *
from utils.dataset_op import create_bins
from utils.prob_op import kernel_cdf
from vine_tree.tree_op import parent_var
from grid.grid_op import mk_grid


###################### GENERATE SAMPLES ##############################

def generate_r_samples(cases, r_matrix, ind_vine, nodes, margin_vine, cop_vine, n_bin, binning):
    d = len(r_matrix)
    n = len(r_matrix) -1

    w = np.random.uniform(0,1,(cases,d))
    w = w.astype(np.float32)
    
    tau_bins = []
    tau_corr = []
    for tr in range(0,d-1,1):
        tau_bins1 = []
        tau_corr1 = []
        for col in range(0,d-1-tr,1):
            tau_bins11 = []
            for bb in range(0,n_bin,1):
                tau_bins11.append([])
            tau_bins1.append(tau_bins11)
            tau_corr1.append([])
        tau_bins.append(tau_bins1)
        tau_corr.append(tau_corr1)

    

    v = np.zeros([cases,d,d],w.dtype)
    v_flip = np.zeros([cases,d,d],w.dtype)
    v[:,0,0] = w[:,0]
    
    for i in range(1,d,1):
        v[:,i,i] = w[:,i]

        c = 0
        for k in range(i-1,-1,-1):
            tr = k
            col = i-k-1
            ind_now = ind_vine[k][c]

            if k == 0:
                tr1 = n-k
                col1 = n-i
                ind1 = r_matrix[tr1,col1] #- 1
                ind1 = np.where(nodes == ind1)
                ind1 = ind1[0][0]

                v2 = v[:,k,ind1][...,np.newaxis]
                v1 = v[:,k+1,i][...,np.newaxis]
                vv = np.concatenate((v1,v2),1)
                v[:,k,i] = copulainvccdf(cop_vine[tr][col],vv)
                
                tau, p_value = stats.kendalltau(vv[:,1],v[:,k,i])
                tau_corr[tr][col] = tau
                tau_bins[tr][col] = tau
            else:

                parent, inx1, inx2 = parent_var(k,ind_vine,ind_now)

                if ind_vine[k-1][ind_now[0]][0] != parent: 
                    v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
                else:
                    v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

                v1 = v[:,k+1,i][...,np.newaxis]
                vv = np.concatenate((v1,v2),1)
                if binning == False:
                    v[:,k,i] = copulainvccdf(cop_vine[tr][col],vv)
                    
                    tau, p_value = stats.kendalltau(vv[:,1],v[:,k,i])
                    tau_corr[tr][col] = tau
                    
                else:
                    
                    ind1 = parent
                    
                    if k == 1:
                        ind1 = np.where(nodes == ind1 +1)
                        ind1 = ind1[0][0]
                        bins = create_bins(v[:,k-1,ind1],n_bin)
                        val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                    else:
                        ind_par_now = ind_vine[k-1][ind_now[1]]
                        parent22, inx1, inx2 = parent_var(k-1,ind_vine,ind_par_now)  

                        ind1 = ind1 + k - 1
                        if (ind_vine[k-2][ind_par_now[0]][0] == parent22):
                            bins = create_bins(v[:,k-1,ind1],n_bin)
                            val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                        else:
                            bins = create_bins(v_flip[:,k-1,ind1],n_bin)
                            val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1

                    vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                    for bb in range(0,n_bin,1):
                        mask = np.where(val_to_bin == bb)
                        vv_bin = vv[mask[0],:]
                        
                        ### CDF FORCE UNIFORM
                        vv_bin_new = vv_bin
                        u_1, ex_u = mk_grid(tf.convert_to_tensor(50),vv_bin.dtype)
                        for zz in range(0,2,1):
                            vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],ex_u)
                        vv_bin = vv_bin_new
                        ###
                        
                        v[mask[0],k,i] = copulainvccdf(cop_vine[tr][col][bb],vv_bin)
                        
                        tau, p_value = stats.kendalltau(vv_bin[:,1],v[mask[0],k,i])
                        print('Tau value bin -',bb, '- is: ', tau)
                        tau_bins[tr][col][bb] = tau
#                         tau_binned.append(tau)
                        
                        corr = stats.pearsonr(vv_bin[:,1],v[mask[0],k,i])
                        print('Corr value  UV space, bin(',bb,')',corr[0])
            c += 1
        
        print('-----------')
        
        if i < d -1:
            for ii in range(1,i+1,1):
                for j in range(0,ii,1):
                    tr = j
                    col = ii-j-1

                    ind_now = ind_vine[j][ii-1-j]

                    if j == n-2:
                        ind_sup = ind_vine[j+1][0]
                    else:
                        ind_sup = ind_vine[j+1][i-1-j]

                    if j == 0:
                        tr1 = n-j
                        col1 = n-ii
                        ind1 = r_matrix[tr1,col1] #- 1
                        ind1 = np.where(nodes == ind1)
                        ind1 = ind1[0][0]

                        v2 = v[:,j,ind1][...,np.newaxis]
                    else:
                        parent1, inx1, inx2 = parent_var(j,ind_vine,ind_now)

                        if ind_vine[j-1][ind_now[0]][0] != parent1:
                            v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
                        else:
                            v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

                    v1 = v[:,j,ii][...,np.newaxis]

                    vv = np.concatenate((v1,v2),1)

                    parent, inx1, inx2 = parent_var(j+1,ind_vine,ind_sup)                
                    u_edge = {ind_now[0], ind_now[1]}

                    if (j == 0) | (binning == False):
                        if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                vv = np.concatenate((v2,v1),1)
                                v_flip[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
                            else:
                                v[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
                        else:
                            v[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
                    else:
                        if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                vv = np.concatenate((v2,v1),1)
                                flip_1 = True
                            else:
                                flip_1 = False
                        else:
                            flip_1 = False

                        ind1 = parent1
                        
                        if j == 1:
                            ind1 = np.where(nodes == ind1 +1)
                            ind1 = ind1[0][0]
                            bins = create_bins(v[:,j-1,ind1],n_bin)
                            val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                        else:
                            ind_par_now = ind_vine[j-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(j-1,ind_vine,ind_par_now)  

                            ind1 = ind1 + j - 1
                            if (ind_vine[j-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(v[:,j-1,ind1],n_bin)
                                val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                            else:
                                bins = create_bins(v_flip[:,j-1,ind1],n_bin)
                                val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1
                            
                        vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                        for bb in range(0,n_bin,1):
                            mask = np.where(val_to_bin == bb)
                            vv_bin = vv[mask[0],:]
                            
                            ### CDF FORCE UNIFORM
                            vv_bin_new = vv_bin
                            u_1, ex_u = mk_grid(tf.convert_to_tensor(50),vv_bin.dtype)
                            for zz in range(0,2,1):
                                vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],ex_u)
                            vv_bin = vv_bin_new
                            ###
                        
                            tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])

                            if flip_1 == True:
                                v_flip[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)
                            else:
                                v[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)
    
    u = np.reshape(v[:,0,:],np.shape(w))
    u1 = np.zeros(np.shape(u),u.dtype)
    c= 0
    for i in range(d-1,-1,-1):
        ind = r_matrix[i,i]-1
        u1[:,ind] = u[:,c]
        c += 1
    u = u1

    sample = np.zeros((cases,np.shape(u)[1]),w.dtype)
    for i in range(0,np.shape(u)[1],1):
        sample[:,i] = margininv(margin_vine[i],u[:,i])
    return sample, v, v_flip, tau_corr, tau_bins

# def generate_r_samples(cases, r_matrix, ind_vine, nodes, margin_vine, cop_vine, n_bin, binning):
#     d = len(r_matrix)
#     n = len(r_matrix) -1

#     w = np.random.uniform(0,1,(cases,d))
#     w = w.astype(np.float32)
    
#     tau_bins = []
#     tau_corr = []
#     for tr in range(0,d-1,1):
#         tau_bins1 = []
#         tau_corr1 = []
#         for col in range(0,d-1-tr,1):
#             tau_bins11 = []
#             for bb in range(0,n_bin,1):
#                 tau_bins11.append([])
#             tau_bins1.append(tau_bins11)
#             tau_corr1.append([])
#         tau_bins.append(tau_bins1)
#         tau_corr.append(tau_corr1)

    

#     v = np.zeros([cases,d,d],w.dtype)
#     v_flip = np.zeros([cases,d,d],w.dtype)
#     v[:,0,0] = w[:,0]
    
#     for i in range(1,d,1):
#         v[:,i,i] = w[:,i]

#         c = 0
#         for k in range(i-1,-1,-1):
#             tr = k
#             col = i-k-1
#             ind_now = ind_vine[k][c]

#             if k == 0:
#                 tr1 = n-k
#                 col1 = n-i
#                 ind1 = r_matrix[tr1,col1] #- 1
#                 ind1 = np.where(nodes == ind1)
#                 ind1 = ind1[0][0]

#                 v2 = v[:,k,ind1][...,np.newaxis]
#                 v1 = v[:,k+1,i][...,np.newaxis]
#                 vv = np.concatenate((v1,v2),1)
#                 v[:,k,i] = copulainvccdf(cop_vine[tr][col],vv)
                
#                 tau, p_value = stats.kendalltau(vv[:,1],v[:,k,i])
#                 print('Tau value is: ', tau)
#                 tau_corr[tr][col] = tau
#                 tau_bins[tr][col] = tau
#             else:

#                 parent, inx1, inx2 = parent_var(k,ind_vine,ind_now)

#                 if ind_vine[k-1][ind_now[0]][0] != parent: 
#                     v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
#                 else:
#                     v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

#                 v1 = v[:,k+1,i][...,np.newaxis]
#                 vv = np.concatenate((v1,v2),1)
#                 if binning == False:
#                     v[:,k,i] = copulainvccdf(cop_vine[tr][col],vv)
                    
#                     tau, p_value = stats.kendalltau(vv[:,1],v[:,k,i])
#                     print('Tau value is: ', tau)
#                     tau_corr[tr][col] = tau
                    
#                 else:
                    
#                     ind1 = parent
                    
#                     if (ind_vine[k-1][ind_now[0]][0] == parent) | (k == 1): 
#                         if k == 1:
#                             ind1 = np.where(nodes == ind1 +1)
#                             ind1 = ind1[0][0]
#                         else:
#                             ind1 = ind1 + k - 1
#                         bins = create_bins(v[:,k-1,ind1],n_bin)
#                         val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
#                     else:
#                         ind1 = ind1 + k - 1
#                         bins = create_bins(v_flip[:,k-1,ind1],n_bin)
#                         val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1
                        
#                     vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                     for bb in range(0,n_bin,1):
#                         mask = np.where(val_to_bin == bb)
#                         vv_bin = vv[mask[0],:]
                        
#                         ### CDF FORCE UNIFORM
#                         vv_bin_new = vv_bin
#                         u_1, ex_u = mk_grid(tf.convert_to_tensor(50),vv_bin.dtype)
#                         for zz in range(0,2,1):
#                             vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],ex_u)
#                         vv_bin = vv_bin_new
#                         ###
                        
#                         v[mask[0],k,i] = copulainvccdf(cop_vine[tr][col][bb],vv_bin)
                        
#                         tau, p_value = stats.kendalltau(vv_bin[:,1],v[mask[0],k,i])
#                         print('Tau value bin -',bb, '- is: ', tau)
#                         tau_bins[tr][col][bb] = tau
# #                         tau_binned.append(tau)
                        
# #                         corr = stats.pearsonr(vv_bin[:,1],v[mask[0],k,i])
# #                         print('Corr value  UV space, bin(',bb,')',corr[0])
#             c += 1
        
#         if i < d -1:
#             for ii in range(1,i+1,1):
#                 for j in range(0,ii,1):
#                     tr = j
#                     col = ii-j-1

#                     ind_now = ind_vine[j][ii-1-j]

#                     if j == n-2:
#                         ind_sup = ind_vine[j+1][0]
#                     else:
#                         ind_sup = ind_vine[j+1][i-1-j]

#                     if j == 0:
#                         tr1 = n-j
#                         col1 = n-ii
#                         ind1 = r_matrix[tr1,col1] #- 1
#                         ind1 = np.where(nodes == ind1)
#                         ind1 = ind1[0][0]

#                         v2 = v[:,j,ind1][...,np.newaxis]
#                     else:
#                         parent1, inx1, inx2 = parent_var(j,ind_vine,ind_now)

#                         if ind_now[0] != parent1:
#                             v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
#                         else:
#                             v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

#                     v1 = v[:,j,ii][...,np.newaxis]

#                     vv = np.concatenate((v1,v2),1)

#                     parent, inx1, inx2 = parent_var(j+1,ind_vine,ind_sup)                
#                     u_edge = {ind_now[0], ind_now[1]}
                    
#                     if (j == 0) | (binning == False):
#                         if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
#                             if ind_now[0] != parent:
#                                 vv = np.concatenate((v2,v1),1)
#                                 v_flip[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
#                             else:
#                                 v[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
#                         else:
#                             v[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
#                     else:
                        
#                         if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
#                             if ind_now[0] != parent:
#                                 vv = np.concatenate((v2,v1),1)
#                                 flip_1 = True
#                             else:
#                                 flip_1 = False
#                         else:
#                             flip_1 = False
                        
#                         ind1 = parent1
#                         if (ind_vine[j-1][ind_now[0]][0] == parent1) | (j == 1): 
#                             ind1 = np.where(nodes == ind1 + 1)
#                             ind1 = ind1[0][0]
#                             bins = create_bins(v[:,j-1,ind1],n_bin)
#                             val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
#                         else:
#                             ind1 = ind1 + j -1
#                             bins = create_bins(v_flip[:,j-1,ind1],n_bin)
#                             val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1
                            
#                         vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                         for bb in range(0,n_bin,1):
#                             mask = np.where(val_to_bin == bb)
#                             vv_bin = vv[mask[0],:]
                            
#                             ### CDF FORCE UNIFORM
#                             vv_bin_new = vv_bin
#                             u_1, ex_u = mk_grid(tf.convert_to_tensor(50),vv_bin.dtype)
#                             for zz in range(0,2,1):
#                                 vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],ex_u)
#                             vv_bin = vv_bin_new
#                             ###
                        
#                             tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])

#                             if flip_1 == True:
#                                 v_flip[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)
#                             else:
#                                 v[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)

    
#     u = np.reshape(v[:,0,:],np.shape(w))
#     u1 = np.zeros(np.shape(u),u.dtype)
#     c= 0
#     for i in range(d-1,-1,-1):
#         ind = r_matrix[i,i]-1
#         u1[:,ind] = u[:,c]
#         c += 1
#     u = u1

#     sample = np.zeros((cases,np.shape(u)[1]),w.dtype)
#     for i in range(0,np.shape(u)[1],1):
#         sample[:,i] = margininv(margin_vine[i],u[:,i])
#     return sample, v, v_flip, tau_corr, tau_bins

# def generate_r_samples(cases, r_matrix, ind_vine, nodes, margin_vine, cop_vine, n_bin, binning):
#     d = len(r_matrix)
#     n = len(r_matrix) -1

#     w = np.random.uniform(0,1,(cases,d))
#     w = w.astype(np.float32)

#     v = np.zeros([cases,d,d],w.dtype)
#     v_flip = np.zeros([cases,d,d],w.dtype)
#     v[:,0,0] = w[:,0]
    
#     for i in range(1,d,1):
#         v[:,i,i] = w[:,i]

#         c = 0
#         for k in range(i-1,-1,-1):
#             tr = k
#             col = i-k-1
#             ind_now = ind_vine[k][c]

#             if k == 0:
#                 tr1 = n-k
#                 col1 = n-i
#                 ind1 = r_matrix[tr1,col1] #- 1
#                 ind1 = np.where(nodes == ind1)
#                 ind1 = ind1[0][0]

#                 v2 = v[:,k,ind1][...,np.newaxis]
#                 v1 = v[:,k+1,i][...,np.newaxis]
#                 vv = np.concatenate((v1,v2),1)
#                 v[:,k,i] = copulainvccdf(cop_vine[tr][col],vv)
#             else:

#                 parent, inx1, inx2 = parent_var(k,ind_vine,ind_now)
# #                 print('par_p1',parent)

#                 if ind_vine[k-1][ind_now[0]][0] != parent: 
#                     v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
#                 else:
#                     v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

#                 v1 = v[:,k+1,i][...,np.newaxis]
#                 vv = np.concatenate((v1,v2),1)
#                 if binning == False:
#                     v[:,k,i] = copulainvccdf(cop_vine[tr][col],vv)
#                 else:
# #                     parent11 = r_matrix[n-k+1,n-i]  #n-k,n-2-i
# #                     print('indr1',n-k+1)
# #                     print('indr2',n-i)
# #                     print('ind_now',ind_now)
# #                     print('ind_bef',ind_vine[k-1][ind_now[0]])
# #                     print('ind_bef',ind_vine[k-1][ind_now[1]])
                    
# #                     parent_p, inx1, inx2 = parent_var(k,ind_vine,ind_now)
# #                     print('par_p2',parent_p)
# #                     print('parent',parent11)
# #                     print('nodes',nodes)
# #                     ind1 = np.where(nodes == parent11)
# #                     ind1 = ind1[0][0]
                    
#                     ind1 = parent
                    
#                     bins = create_bins(v[:,k-1,ind1],n_bin)
#                     val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
#                     vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                     for bb in range(0,n_bin,1):
#                         mask = np.where(val_to_bin == bb)
#                         vv_bin = vv[mask[0],:]
#                         v[mask[0],k,i] = copulainvccdf(cop_vine[tr][col][bb],vv_bin)
#             c += 1

#         if i < d-1:
#             for ii in range(1,i+1,1):

#                 for j in range(0,ii,1):
#                     tr = j
#                     col = ii-j-1

#                     ind_now = ind_vine[j][ii-1-j]

#                     if j == n-2:
#                         ind_sup = ind_vine[j+1][0]
#                     else:
#                         ind_sup = ind_vine[j+1][i-1-j]

#                     if j == 0:
#                         tr1 = n-j
#                         col1 = n-ii
#                         ind1 = r_matrix[tr1,col1] #- 1
#                         ind1 = np.where(nodes == ind1)
#                         ind1 = ind1[0][0]

#                         v2 = v[:,j,ind1][...,np.newaxis]
#                     else:
#                         parent1, inx1, inx2 = parent_var(j,ind_vine,ind_now)

#                         if ind_now[0] != parent1:
#                             v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
#                         else:
#                             v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

#                     v1 = v[:,j,ii][...,np.newaxis]

#                     vv = np.concatenate((v1,v2),1)

#                     parent, inx1, inx2 = parent_var(j+1,ind_vine,ind_sup)                
#                     u_edge = {ind_now[0], ind_now[1]}
                    
#                     if (j == 0) | (binning == False):
#                         if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
#                             if ind_now[0] != parent:
#                                 vv = np.concatenate((v2,v1),1)
#                                 v_flip[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
#                             else:
#                                 v[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
#                         else:
#                             v[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
#                     else:
# #                         parent11 = r_matrix[n-j+1,n-ii]  #n-k,n-2-i
# #                         print('indr1',n-j+1)
# #                         print('indr2',n-ii)
                        
# # #                         [n-j,n-2-ii]
# #                         print('ind_now',ind_now)
# #                         print('parent',parent)
#                         if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
#                             if ind_now[0] != parent:
#                                 vv = np.concatenate((v2,v1),1)
                                
# #                                 ind1 = np.where(nodes == parent11)
# #                                 ind1 = ind1[0][0]
                                
#                                 ind1 = parent1

#                                 bins = create_bins(v[:,j-1,ind1],n_bin)
#                                 val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
#                                 vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                                 for bb in range(0,n_bin,1):
#                                     mask = np.where(val_to_bin == bb)
#                                     vv_bin = vv[mask[0],:]
#                                     tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])
#                                     v_flip[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)
#                             else:
#                                 ind1 = np.where(nodes == parent11)
#                                 ind1 = ind1[0][0]
                                
#                                 ind1 = parent1

#                                 bins = create_bins(v[:,j-1,ind1],n_bin)
#                                 val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
#                                 vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                                 for bb in range(0,n_bin,1):
#                                     mask = np.where(val_to_bin == bb)
#                                     vv_bin = vv[mask[0],:]
#                                     tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])
#                                     v_flip[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)
#                         else:
#                             ind1 = np.where(nodes == parent11)
#                             ind1 = ind1[0][0]
                            
#                             ind1 = parent1
                            
#                             bins = create_bins(v[:,j-1,ind1],n_bin)
#                             val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
#                             vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                             for bb in range(0,n_bin,1):
#                                 mask = np.where(val_to_bin == bb)
#                                 vv_bin = vv[mask[0],:]
#                                 tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])
#                                 v_flip[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)
                    
#     u = np.reshape(v[:,0,:],np.shape(w))
#     u1 = np.zeros(np.shape(u),u.dtype)
#     c= 0
#     for i in range(d-1,-1,-1):
#         ind = r_matrix[i,i]-1
#         u1[:,ind] = u[:,c]
#         c += 1
#     u = u1

#     sample = np.zeros((cases,np.shape(u)[1]),w.dtype)
#     for i in range(0,np.shape(u)[1],1):
#         sample[:,i] = margininv(margin_vine[i],u[:,i])
#     return sample, v, v_flip

# def generate_r_samples(cases, r_matrix, ind_vine, nodes, margin_vine, cop_vine, n_bin, binning):
#     d = len(r_matrix)
#     n = len(r_matrix) -1

#     w = np.random.uniform(0,1,(cases,d))
#     w = w.astype(np.float32)

#     v = np.zeros([cases,d,d],w.dtype)
#     v_flip = np.zeros([cases,d,d],w.dtype)
#     v[:,0,0] = w[:,0]
    
#     for i in range(1,d,1):
#         v[:,i,i] = w[:,i]

#         c = 0
#         for k in range(i-1,-1,-1):
#             tr = k
#             col = i-k-1
#             ind_now = ind_vine[k][c]

#             if k == 0:
#                 tr1 = n-k
#                 col1 = n-i
#                 ind1 = r_matrix[tr1,col1] #- 1
#                 ind1 = np.where(nodes == ind1)
#                 ind1 = ind1[0][0]

#                 v2 = v[:,k,ind1][...,np.newaxis]
#                 v1 = v[:,k+1,i][...,np.newaxis]
#                 vv = np.concatenate((v1,v2),1)
#                 v[:,k,i] = copulainvccdf(cop_vine[tr][col],vv)
#             else:

#                 parent, inx1, inx2 = parent_var(k,ind_vine,ind_now)

#                 if ind_vine[k-1][ind_now[0]][0] != parent: 
#                     v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
#                 else:
#                     v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

#                 v1 = v[:,k+1,i][...,np.newaxis]
#                 vv = np.concatenate((v1,v2),1)
#                 if binning == False:
#                     v[:,k,i] = copulainvccdf(cop_vine[tr][col],vv)
#                 else:
#                     parent11 = r_matrix[n-k+1,n-i]  #n-k,n-2-i
# #                     print('indr1',n-k+1)
# #                     print('indr2',n-i)
# #                     print('ind_now',ind_now)
# #                     print('parent',parent11)
# #                     print('nodes',nodes)
#                     ind1 = np.where(nodes == parent11)
#                     ind1 = ind1[0][0]
                    
#                     bins = create_bins(v[:,0,ind1],n_bin)
#                     val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                     vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                     for bb in range(0,n_bin,1):
#                         mask = np.where(val_to_bin == bb)
#                         vv_bin = vv[mask[0],:]
#                         v[mask[0],k,i] = copulainvccdf(cop_vine[tr][col][bb],vv_bin)
#             c += 1

#         if i < d-1:
#             for ii in range(1,i+1,1):

#                 for j in range(0,ii,1):
#                     tr = j
#                     col = ii-j-1

#                     ind_now = ind_vine[j][ii-1-j]

#                     if j == n-2:
#                         ind_sup = ind_vine[j+1][0]
#                     else:
#                         ind_sup = ind_vine[j+1][i-1-j]

#                     if j == 0:
#                         tr1 = n-j
#                         col1 = n-ii
#                         ind1 = r_matrix[tr1,col1] #- 1
#                         ind1 = np.where(nodes == ind1)
#                         ind1 = ind1[0][0]

#                         v2 = v[:,j,ind1][...,np.newaxis]
#                     else:
#                         parent1, inx1, inx2 = parent_var(j,ind_vine,ind_now)

#                         if ind_now[0] != parent1:
#                             v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
#                         else:
#                             v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

#                     v1 = v[:,j,ii][...,np.newaxis]

#                     vv = np.concatenate((v1,v2),1)

#                     parent, inx1, inx2 = parent_var(j+1,ind_vine,ind_sup)                
#                     u_edge = {ind_now[0], ind_now[1]}
                    
#                     if (j == 0) | (binning == False):
#                         if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
#                             if ind_now[0] != parent:
#                                 vv = np.concatenate((v2,v1),1)
#                                 v_flip[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
#                             else:
#                                 v[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
#                         else:
#                             v[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
#                     else:
#                         parent11 = r_matrix[n-j+1,n-ii]  #n-k,n-2-i
# #                         print('indr1',n-j+1)
# #                         print('indr2',n-ii)
                        
# # #                         [n-j,n-2-ii]
# #                         print('ind_now',ind_now)
# #                         print('parent',parent)
#                         if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
#                             if ind_now[0] != parent:
#                                 vv = np.concatenate((v2,v1),1)
                                
#                                 ind1 = np.where(nodes == parent11)
#                                 ind1 = ind1[0][0]

#                                 bins = create_bins(v[:,0,ind1],n_bin)
#                                 val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                                 vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                                 for bb in range(0,n_bin,1):
#                                     mask = np.where(val_to_bin == bb)
#                                     vv_bin = vv[mask[0],:]
#                                     tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])
#                                     print('Tau value bin -',bb, '- is: ', tau)
#                                     corr = stats.pearsonr(vv_bin[:,0],vv_bin[:,1])
#                                     print('Corr value  UV space: ',corr[0])
#                                     v_flip[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)
#                             else:
#                                 ind1 = np.where(nodes == parent11)
#                                 ind1 = ind1[0][0]

#                                 bins = create_bins(v[:,0,ind1],n_bin)
#                                 val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                                 vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                                 for bb in range(0,n_bin,1):
#                                     mask = np.where(val_to_bin == bb)
#                                     vv_bin = vv[mask[0],:]
#                                     tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])
#                                     print('Tau value bin -',bb, '- is: ', tau)
#                                     corr = stats.pearsonr(vv_bin[:,0],vv_bin[:,1])
#                                     print('Corr value  UV space: ',corr[0])
#                                     v_flip[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)
#                         else:
#                             ind1 = np.where(nodes == parent11)
#                             ind1 = ind1[0][0]

#                             bins = create_bins(v[:,0,ind1],n_bin)
#                             val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                             vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                             for bb in range(0,n_bin,1):
#                                 mask = np.where(val_to_bin == bb)
#                                 vv_bin = vv[mask[0],:]
#                                 tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])
#                                 print('Tau value bin -',bb, '- is: ', tau)
#                                 corr = stats.pearsonr(vv_bin[:,0],vv_bin[:,1])
#                                 print('Corr value  UV space: ',corr[0])
#                                 v_flip[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)
                    
#     u = np.reshape(v[:,0,:],np.shape(w))
#     u1 = np.zeros(np.shape(u),u.dtype)
#     c= 0
#     for i in range(d-1,-1,-1):
#         ind = r_matrix[i,i]-1
#         u1[:,ind] = u[:,c]
#         c += 1
#     u = u1

#     sample = np.zeros((cases,np.shape(u)[1]),w.dtype)
#     for i in range(0,np.shape(u)[1],1):
#         sample[:,i] = margininv(margin_vine[i],u[:,i])
#     return sample, v, v_flip

# File: src/DVC_tensorflow/param/.ipynb_checkpoints/margin_cost-checkpoint.py
import tensorflow as tf
from utils.tensor_op import replace_nan_inf,replace_nan_with, replace_inf_with
from param.margin_pdf import *

############################################ GAUSSIAN COST FUNCTION  ######################################

@tf.function(experimental_relax_shapes=True)
def gaussian_cost(u,theta_par):
    p = gaussian_pdf(u,theta_par)
    p = replace_nan_with(p,tf.constant(1,u.dtype))
    eps = tf.constant(2.220446049250313e-16,u.dtype)
    err = -tf.math.reduce_sum(tf.math.log(p+eps),[0])
    return err

# @tf.function
# def gaussian_cost(u,theta_par, norm_dis):
#     p = gaussian_pdf(u,theta_par, norm_dis)
#     #p = replace_nan2(p)
#     eps = tf.constant(2.220446049250313e-16,u.dtype)
#     err = -tf.math.reduce_sum(tf.math.log(p+eps))
#     return err

############################################ STUDENT COST FUNCTION  ######################################

# @tf.function(experimental_relax_shapes=True)
def student_cost(u,theta):
    p = student_pdf(u,theta)
    eps = tf.constant(2.220446049250313e-16,u.dtype)
    err = -tf.math.reduce_sum(tf.math.log(p+eps),[0])
    return err

# @tf.function
# def student_cost(u,theta):
#     p = student_pdf(u,theta)
#     eps = 2.220446049250313e-16
#     err = -tf.math.reduce_sum(tf.math.log(p+eps))
#     cond = tf.math.logical_or(
#                 tf.math.logical_or(
#                     tf.math.less_equal(theta[0],-1),
#                     tf.math.greater(theta[0],1)),
#                 tf.math.logical_or(
#                     tf.math.less_equal(theta[1],1e-1),
#                     tf.math.greater(theta[0],1000)))
#     err = tf.cond(cond, lambda: u.dtype.max, lambda: err)
#     return err

############################################ CLAYTON COST FUNCTION  ######################################

@tf.function(experimental_relax_shapes=True)
def clayton_cost(u,theta_cla1):
    p = clayton_pdf(u,theta_cla1)
    p = replace_nan_with(p,tf.constant(1,u.dtype))
    p = replace_inf_with(p, tf.constant(p.dtype.max,u.dtype))
    eps = tf.constant(2.220446049250313e-16,u.dtype)
    err = -tf.math.reduce_sum(tf.math.log(p+eps),[0])
    return err

# @tf.function
# def clayton_cost(u,theta_cla1):
#     p = clayton_pdf(u,theta_cla1)
#     eps = tf.constant(2.220446049250313e-16,u.dtype)
#     err = -tf.math.reduce_sum(tf.math.log(p+eps))
#     return err

############################################ CLAYTON ROT 90 COST FUNCTION  ######################################

@tf.function(experimental_relax_shapes=True)
def claytonrot90_cost(u,theta_cla1):
    p = claytonrot90_pdf(u,theta_cla1)
    p = replace_nan_with(p,tf.constant(1,u.dtype))
    p = replace_inf_with(p, tf.constant(p.dtype.max,u.dtype))
    eps = tf.constant(2.220446049250313e-16,u.dtype)
#     log_p = tf.math.log(p+eps)
#     log_p = replace_nan_inf(log_p)
#     err = - tf.math.reduce_sum(log_p)
    err = -tf.math.reduce_sum(tf.math.log(p+eps),[0])
    return err

# @tf.function
# def claytonrot90_cost(u,theta_cla1):
#     p = claytonrot90_pdf(u,theta_cla1)
#     eps = tf.constant(2.220446049250313e-16,u.dtype)
#     log_p = tf.math.log(p+eps)
#     log_p = replace_nan_inf(log_p)
#     err = - tf.math.reduce_sum(log_p)
#     return err


# File: src/DVC_tensorflow/param/.ipynb_checkpoints/margin_fit-checkpoint.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from scipy import stats

from classes.objects import margin_obj
from param.margin_pdf import *
from param.margin_op import marginpdf


##################### PARAMETRIC MARGIN FITTING   ###############################

# @tf.function
def marginfit(fam,x):   
    theta = []
    if fam == 'norm':
        loc, scale = stats.norm.fit(x)
        theta = [loc,scale]
    elif fam == 'gamma':
        concentration, loc, rate = stats.gamma.fit(x)
        theta = [concentration, rate]
#     elif tf.equal(fam,"poiss"):
#         theta = tf.math.reduce_mean(x)
#         theta = tf.reshape(theta,[1])
#     elif tf.equal(fam,"bin"):
#         mu1 = tf.math.reduce_max(x)
#         mu2 = tf.math.reduce_mean(x/mu1)
#         mu1 = mu1[...,tf.newaxis]
#         mu2 = mu2[...,tf.newaxis]
#         theta = tf.concat([mu1,mu2],0)
    return theta

def marginfit_all(x):
    if np.any(x < 0):
        families = ["norm"]
    else:
        families = ["gamma"]
    aic = []
    theta = []
    k = 0
    
    for i in families:
#         if i == 'norm':
#             theta1 = marginfit(i,x)
#         elif i == 'gamma':
        theta1 = marginfit(i,x)
        iscont = True
        
        mar_pp  = margin_obj(i, theta1, iscont)
        
        logp = np.sum(np.log(marginpdf(mar_pp,x) + 1e-30))

        aic1 = 2*tf.cast(tf.size(theta1),x.dtype) - 2*logp
        aic = aic.append(aic1)
        theta.append(theta1)
        k = k+1
    
    ind_min = np.argmin(aic)
    return families[ind_min], theta[ind_min]

# File: src/DVC_tensorflow/param/.ipynb_checkpoints/margin_op-checkpoint.py


# File: src/DVC_tensorflow/param/.ipynb_checkpoints/margin_pdf-checkpoint.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from scipy.special import gammaln
from scipy.stats import t
from utils.tensor_op import replace_nan_inf,update_tensor2D
import math as m

############################################ GAUSSIAN MARGIN PDF  ######################################

@tf.function(experimental_relax_shapes=True)
def gaussian_pdf(u,theta_par):
    norm_dis = tfd.Normal(loc=tf.constant(0.,u.dtype), scale=tf.constant(1.,u.dtype))
    x = norm_dis.quantile(u)
    p = tf.exp((2*theta_par*x[:,0,:]*x[:,1,:] - (theta_par**2) * (x[:,0,:]**2 + x[:,1,:]**2))/(2*(1-theta_par**2))) / tf.math.sqrt(1-theta_par**2)
    return p

# @tf.function
# def gaussian_pdf(u,theta_par,norm_dis):
#     x = norm_dis.quantile(u)
#     p = tf.exp((2*theta_par*x[:,0]*x[:,1] - (theta_par**2) * (x[:,0]**2 + x[:,1]**2))/(2*(1-theta_par**2))) / tf.math.sqrt(1-theta_par**2)
#     return p

############################################ CLAYTON MARGIN PDF  ######################################

@tf.function(experimental_relax_shapes=True)
def clayton_pdf(u,theta):
    p = (1 + theta) * (u[:,0,:] * u[:,1,:])**(-1-theta) * (u[:,0,:]**(-theta) + u[:,1,:]**(-theta) - 1)**(-1/theta-2)
    if tf.shape(tf.shape(theta))[0] == 0:
        theta = theta[...,tf.newaxis]
    cond = tf.math.equal(theta,0)
    ind = tf.where(tf.equal(cond,True))
    ind = tf.cast(ind,tf.int32)

    for i in ind:
        newval = tf.ones(tf.shape(u)[0],u.dtype)
        p = update_tensor2D(p, i[0] , newval)
    return p

# @ tf.function
# def clayton_pdf(u,theta):
#     def f1(): return tf.constant(1,u.dtype)
#     def f2(): return (1 + theta) * (u[:,0] * u[:,1])**(-1-theta) * (u[:,0]**(-theta) + u[:,1]**(-theta) - 1)**(-1/theta-2)
#     p = tf.cond(tf.math.equal(theta,0),f1,f2)
#     p = replace_nan_inf(p)
#     return p

############################################ CLAYTON ROT 90 MARGIN PDF  ######################################

@tf.function(experimental_relax_shapes=True)
def claytonrot90_pdf(u,theta):
    p = (1 + theta) * (u[:,0,:] * (1-u[:,1,:]))**(-1-theta) * (u[:,0,:]**(-theta) + (1-u[:,1,:])**(-theta) - 1)**(-1/theta-2)
    if tf.shape(tf.shape(theta))[0] == 0:
        theta = theta[...,tf.newaxis]
    cond = tf.math.equal(theta,0)
    ind = tf.where(tf.equal(cond,True))
    ind = tf.cast(ind,tf.int32)

    for i in ind:
        newval = tf.ones(tf.shape(u)[0],u.dtype)
        p = update_tensor2D(p, i[0] , newval)
    return p

# @tf.function
# def claytonrot90_pdf(u,theta):
#     def f1(): return tf.constant(1,u.dtype)
#     def f2(): return (1 + theta) * (u[:,0] * (1-u[:,1]))**(-1-theta) * (u[:,0]**(-theta) + (1-u[:,1])**(-theta) - 1)**(-1/theta-2)
#     p = tf.cond(tf.math.equal(theta,0),f1,f2)
#     p = replace_nan_inf(p)
#     return p

############################################ STUDENT MARGIN PDF  ######################################

# @tf.function(experimental_relax_shapes=True)
def gammaln1(x):
    return tf.py_function(gammaln, [x], x.dtype)

# @tf.function(experimental_relax_shapes=True)
def tpdf(x,vk):
    term = tf.exp(gammaln1((vk + 1) / 2) - gammaln1(vk/2))
    pi = tf.constant(m.pi,x.dtype)
    y = term / (tf.math.sqrt(vk*pi) * (1 + (x**2) / vk) ** ((vk + 1)/2))
    return y

# @tf.function(experimental_relax_shapes=True)
def student_pdf(u,theta):
    if tf.shape(tf.shape(theta)) == 1:
        theta = theta[tf.newaxis,...]
    
    df = theta[:,1]
    loc = tf.constant(0,u.dtype)
    scale = tf.constant(1,u.dtype)
    pi = tf.constant(m.pi,u.dtype)

    x = tf.py_function(t.ppf, [u, theta[:,1], loc, scale], u.dtype)
    factor1 = gammaln1(theta[:,1]/2+1)
    factor2 = -gammaln1(theta[:,1]/2) - tf.math.log(pi) - tf.math.log(theta[:,1]) - tf.math.log(1-theta[:,0]**2)/2 - tf.math.log(tpdf(x[:,0,:],theta[:,1])) - tf.math.log(tpdf(x[:,1,:],theta[:,1]))

    factor3 = (-(theta[:,1]+2)/2) * tf.math.log(1 + (x[:,0,:]**2 + x[:,1,:]**2 - theta[:,0] * x[:,0,:] * x[:,1,:]) / (theta[:,1]*(1-theta[:,0]**2)))
    p = tf.exp(factor1 + factor2 + factor3)
    p = replace_nan_inf(p)
    return p

# @tf.function
# def gammaln1(x):
#     return tf.py_function(gammaln, [x], x.dtype)

# @tf.function
# def tpdf(x,vk):
#     term = tf.exp(gammaln1((vk + 1) / 2) - gammaln1(vk/2))
#     pi = tf.constant(m.pi,x.dtype)
#     y = term / (tf.math.sqrt(vk*pi) * (1 + (x**2) / vk) ** ((vk + 1)/2))
#     return y

# @tf.function
# def student_pdf(u,theta):
#     df = theta[1]
#     loc = tf.constant(0,u.dtype)
#     scale = tf.constant(1,u.dtype)
#     pi = tf.constant(m.pi,u.dtype)
#     var = tf.math.divide(df,df-2)
#     t_dis = tfd.StudentT(df=df,loc=tf.constant(0,u.dtype), scale=var) #var

#     x = tf.py_function(t.ppf, [u, theta[1], loc, scale], u.dtype)
#     factor1 = gammaln1(theta[1]/2+1)
#     factor2 = -gammaln1(theta[1]/2) - tf.math.log(pi) - tf.math.log(theta[1]) - tf.math.log(1-theta[0]**2)/2 - tf.math.log(tpdf(x[:,0],theta[1])) - tf.math.log(tpdf(x[:,1],theta[1]))

#     factor3 = (-(theta[1]+2)/2) * tf.math.log(1 + (x[:,0]**2 + x[:,1]**2 - theta[0] * x[:,0] * x[:,1]) / (theta[1]*(1-theta[0]**2)))
#     p = tf.exp(factor1 + factor2 + factor3)
#     p = replace_nan_inf(p)
#     return p

# File: src/DVC_tensorflow/param/cond_copula.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from scipy import stats

from utils.bijector import *
from param.margin_pdf import *

############################################ COPULA PDF      ############################################

def copulapdf(vine_par,u):
    c = np.zeros(np.shape(u)[0],u.dtype)
    u = tf.convert_to_tensor(u)
    
    if vine_par.family == 'ind':
        c = tf.ones(tf.shape(u)[0],u.dtype)
    elif vine_par.family == 'gaussian':
        theta = tf.convert_to_tensor(vine_par.theta,u.dtype)
        c = gaussian_pdf(u,theta)
    if vine_par.family == 'student':
        theta = tf.convert_to_tensor(vine_par.theta,u.dtype)
        c = student_pdf(u,theta)
    if vine_par.family == 'clayton':
        theta = tf.convert_to_tensor(vine_par.theta,u.dtype)
        c = clayton_pdf(u,theta)
    if vine_par.family == 'claytonrot90':
        theta = tf.convert_to_tensor(vine_par.theta,u.dtype)
        c = claytonrot90_pdf(u,theta)
    return c

############################################ COPULA CONDITIONED CDF      ############################################

def copulaccdf(vine_par,u):
    loc = 0
    scale = 1
    
    u[u>=1-1e-7] = 1-1e-7
    u[u<=1e-7] = +1e-7
    
    c = np.zeros(np.shape(u)[0],u.dtype)
    if vine_par.family == 'ind':
        c = u[:,0]
    elif vine_par.family == 'gaussian':
        x = NormalCDF.forward(NormalCDF(loc,scale), u)
        theta = vine_par.theta
        tmp = (x[:,0] - theta * x[:,1]) / np.sqrt(1-theta**2)
        c = NormalCDF.inverse(NormalCDF(loc,scale), tmp)
    elif vine_par.family == 'student':
        theta1 = vine_par.theta[0]
        theta2 = vine_par.theta[1]
        x = stats.t.ppf(u, theta2, loc, scale)
        tmp = np.sqrt((theta2+1) / (theta2+x[:,1]**2)) * (x[:,0] - theta1 * x[:,1]) / (np.sqrt(1-theta1**2)) #, theta[1]+1)
        c = stats.t.cdf(tmp, theta2+1, loc, scale)
    elif vine_par.family == 'clayton':   
        theta = vine_par.theta
        if theta == 0:
            c = u[:,0]
        else:
            c = np.maximum(u[:,1]**(-1-theta) * (u[:,0]**(-theta) + u[:,1]**(-theta) - 1) ** (-1-1/theta),0)
    elif vine_par.family == 'claytonrot90':   
        theta = vine_par.theta
        if theta == 0:
            c = u[:,0]
        else:
            c = np.maximum((1-u[:,1])**(-1-theta) * (u[:,0]**(-theta) + (1-u[:,1])**(-theta) - 1) ** (-1-1/theta),0)
    
    return c

############################################ COPULA INVERSE CONDITIONED CDF      ############################################

def copulainvccdf(vine_par,u):
    loc = 0
    scale = 1
    
    u[u>=1-1e-7] = 1-1e-7
    u[u<=1e-7] = +1e-7
    c = np.zeros(np.shape(u)[0],u.dtype)
    
    if vine_par.family == 'ind':
        c = u[:,0]
    elif vine_par.family == 'gaussian':
        x = NormalCDF.forward(NormalCDF(loc,scale), u)
        theta = vine_par.theta
        tmp = x[:,0] * np.math.sqrt(1-theta**2) + theta * x[:,1]        
        c = NormalCDF.inverse(NormalCDF(loc,scale), tmp)
    if vine_par.family == 'student':
        theta1 = vine_par.theta[0]
        theta2 = vine_par.theta[1]
        x = stats.t.ppf(u, theta2, loc, scale)
        param = theta2 + 1 
        tmp_inv = stats.t.ppf(u[:,0], param, loc, scale)
        tmp = np.sqrt( ((1-theta1**2) * (theta2 + x[:,1]**2)) / (theta2+1) ) * tmp_inv + theta1 * x[:,1]
        c = stats.t.cdf(tmp, theta2, loc, scale)
    if vine_par.family == 'clayton':
        theta = vine_par.theta
        if theta == 0:
            c = u[:,0]
        else:           
            c = (1 - u[:,1]**(-theta) + (u[:,0] * (u[:,1]**(1+theta)))**(-theta/(1+theta)))**(-1/theta)
    if vine_par.family == 'claytonrot90':
        theta = vine_par.theta
        if theta == 0:
            c = u[:,0]
        else:           
            c = (1 - (1 - u[:,1]) **(-theta) + (u[:,0] * ((1 - u[:,1])**(1+theta)))**(-theta/(1+theta)))**(-1/theta)
    return c

# File: src/DVC_tensorflow/param/copula_fit.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from utils.tensor_op import check_bound3, replace_nan_inf
from param.margin_cost import *

################################# GAUSSIAN FITTING ###############################################

@tf.function(experimental_relax_shapes=True)
def fit_gaussian(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    eps = tf.constant(1e-6,a.dtype)
    
    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace = tf.ones([n_cop],u.dtype)
    err = err_trace + 10*convergence_tol
    
    err = gaussian_cost(u, pos_trace)
    
    m = tf.zeros(tf.shape(a),u.dtype)
    v = tf.zeros(tf.shape(a),u.dtype)
    m_hat = tf.zeros(tf.shape(a),u.dtype)
    v_hat = tf.zeros(tf.shape(a),u.dtype)
    beta_1 = tf.constant(0.9,u.dtype)
    beta_2 = tf.constant(0.999,u.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):
        err_trace = err
        err_trace = tf.reshape(err_trace, [n_cop])
        
        err = gaussian_cost(u, a)
        
        grad = (err - err_trace)/(a-pos_trace)
        grad = replace_nan_inf(grad)

        pos_trace = a
        

        iter1 = tf.cast(iter_err,u.dtype)
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(a))
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(a))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        a_new = check_bound3(a,tf.constant(1-1e-3,u.dtype),tf.constant(0+1e-3,u.dtype))
        a = a_new
        a = tf.reshape(a, tf.shape(a))
        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)


################################# STUDENT FITTING ###############################################

# @tf.function(experimental_relax_shapes=True)
def fit_student(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    eps = 1e-6

    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace_x1y = tf.ones([n_cop],u.dtype)
    err_trace_xy1 = tf.ones([n_cop],u.dtype)
    
    err = err_trace_x1y + 10*convergence_tol
    
    m = tf.zeros(tf.shape(a),u.dtype)

    v = tf.zeros(tf.shape(a),u.dtype)
    m_hat = tf.zeros(tf.shape(a),u.dtype)
    v_hat = tf.zeros(tf.shape(a),u.dtype)
    beta_1 = tf.constant(0.9,u.dtype)
    beta_2 = tf.constant(0.999,u.dtype)
    

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),
                              tf.math.logical_or(
                                  tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_x1y),convergence_tol)),
                                  tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace_xy1),convergence_tol))
                              )
                             ):
    
        x1y = tf.concat([pos_trace[:,0][...,tf.newaxis],a[:,1][...,tf.newaxis]],1)
        xy1 = tf.concat([a[:,0][...,tf.newaxis],pos_trace[:,1][...,tf.newaxis]],1)
        
        err_x1y = student_cost(u, x1y)
        err_xy1 = student_cost(u, xy1)

        err_trace_x1y = err_x1y
        err_trace_xy1 = err_xy1
        err_trace_x1y = tf.reshape(err_trace_x1y, [n_cop])
        err_trace_xy1 = tf.reshape(err_trace_xy1, [n_cop])
        
        err = student_cost(u, a)
        
        err = tf.reshape(err, [n_cop])
        
        grad_x1y = (err - err_trace_x1y)/(a[:,0]-x1y[:,0])
        grad_xy1 = (err - err_trace_xy1)/(a[:,1]-xy1[:,1])

        if tf.shape(tf.shape(grad_x1y)) == 1:
            grad_x1y[...,tf.newaxis]
            grad_xy1[...,tf.newaxis]
            
        grad = tf.concat([grad_x1y[...,tf.newaxis],grad_xy1[...,tf.newaxis]],1)
        grad = replace_nan_inf(grad)

        pos_trace = a
        
        iter1 = tf.cast(iter_err,u.dtype)
        
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(a))

        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(a))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        
        a_new1 = check_bound3(a[:,0][...,tf.newaxis],tf.constant(1,u.dtype),tf.constant(-1,u.dtype))
        a_new2 = check_bound3(a[:,1][...,tf.newaxis],tf.constant(1000,u.dtype),tf.constant(1e-3,u.dtype))
        a_new1 = tf.reshape(a_new1,[n_cop,1])
        a_new2 = tf.reshape(a_new2,[n_cop,1])
        a_new = tf.concat([a_new1,a_new2],1)
        a = a_new
        a = tf.reshape(a, [n_cop,2]) 
        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
        
    return a, err, iter_err, tf.math.logical_or(
                                tf.equal( tf.shape(tf.where(tf.equal(tf.math.less(tf.abs(err-err_trace_x1y),convergence_tol),True)))[0] , n_cop),
                                tf.equal( tf.shape(tf.where(tf.equal(tf.math.less(tf.abs(err-err_trace_xy1),convergence_tol),True)))[0] , n_cop)
                                                )

################################# CLAYTON FITTING ###############################################

@tf.function(experimental_relax_shapes=True)
def fit_clayton(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    eps = 1e-6
    
    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace = tf.ones([n_cop],u.dtype)
    
    err = err_trace + 10*convergence_tol
    
    err = clayton_cost(u, pos_trace)
    err = tf.reshape(err, [n_cop])
    
    m = tf.zeros(tf.shape(a),u.dtype)
    v = tf.zeros(tf.shape(a),u.dtype)
    m_hat = tf.zeros(tf.shape(a),u.dtype)
    v_hat = tf.zeros(tf.shape(a),u.dtype)
    beta_1 = tf.constant(0.9,u.dtype)
    beta_2 = tf.constant(0.999,u.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):

        err_trace = err
        err_trace = tf.reshape(err_trace, [n_cop])
        
        err = clayton_cost(u, a)
        
        err = tf.reshape(err, [n_cop])
        
        grad = (err - err_trace)/(a-pos_trace)
        grad = replace_nan_inf(grad)
        
        pos_trace = a
        
        iter1 = tf.cast(iter_err,u.dtype)
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(a))
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(a))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        a_new = check_bound3(a,tf.constant(20,u.dtype),tf.constant(1e-1,u.dtype))
        a = a_new
        a = tf.reshape(a, tf.shape(a))

        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)

################################# CLAYTON ROT 90 FITTING ###############################################

@tf.function(experimental_relax_shapes=True)
def fit_claytonrot90(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    eps = 1e-6
    
    iter_err = tf.constant(1,dtype=tf.int32)
    err_trace = tf.ones([n_cop],u.dtype)
    
    err = err_trace + 10*convergence_tol
    
    err = claytonrot90_cost(u, pos_trace)
    err = tf.reshape(err, [n_cop])
    
    m = tf.zeros(tf.shape(a),u.dtype)
    v = tf.zeros(tf.shape(a),u.dtype)
    m_hat = tf.zeros(tf.shape(a),u.dtype)
    v_hat = tf.zeros(tf.shape(a),u.dtype)
    beta_1 = tf.constant(0.9,u.dtype)
    beta_2 = tf.constant(0.999,u.dtype)

    while tf.math.logical_and(tf.math.less(iter_err, max_iter),tf.math.reduce_any(tf.math.greater(tf.abs(err-err_trace),convergence_tol))):

        err_trace = err
        err_trace = tf.reshape(err_trace, [n_cop])
        
        err = claytonrot90_cost(u, a)
        
        err = tf.reshape(err, [n_cop])
        
        grad = (err - err_trace)/(a-pos_trace)
        grad = replace_nan_inf(grad)
        
        pos_trace = a
        
        iter1 = tf.cast(iter_err,u.dtype)
        m = beta_1 * m + (1 - beta_1) * grad
        m = tf.reshape(m, tf.shape(a))
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = tf.reshape(v, tf.shape(a))
        
        m_hat = m / (1 - beta_1**iter1) + (1 - beta_1) * grad / (1 - beta_1**iter1)
        v_hat = v / (1 - beta_2**iter1)
        diff = - lr * m_hat / (tf.math.sqrt(v_hat) + eps)

        a = a + diff
        a_new = check_bound3(a,tf.constant(20,u.dtype),tf.constant(1e-1,u.dtype))
        a = a_new
        a = tf.reshape(a, tf.shape(a))
        
        
        iter_err = tf.add(iter_err,tf.constant(1,tf.int32))
    return a, err, iter_err, tf.math.less(tf.abs(err-err_trace),convergence_tol)


# File: src/DVC_tensorflow/param/generate_rvine.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from scipy import stats

from classes.objects import margin_obj
from param.margin_pdf import *
from param.margin_op import *
from param.cond_copula import *
from utils.dataset_op import create_bins
from utils.prob_op import kernel_cdf
from vine_tree.tree_op import parent_var
from grid.grid_op import mk_grid


###################### GENERATE SAMPLES ##############################

def generate_r_samples(cases, r_matrix, ind_vine, nodes, margin_vine, cop_vine, n_bin, binning):
    d = len(r_matrix)
    n = len(r_matrix) -1

    w = np.random.uniform(0,1,(cases,d))
    w = w.astype(np.float32)
    
    tau_bins = []
    tau_corr = []
    for tr in range(0,d-1,1):
        tau_bins1 = []
        tau_corr1 = []
        for col in range(0,d-1-tr,1):
            tau_bins11 = []
            for bb in range(0,n_bin,1):
                tau_bins11.append([])
            tau_bins1.append(tau_bins11)
            tau_corr1.append([])
        tau_bins.append(tau_bins1)
        tau_corr.append(tau_corr1)

    

    v = np.zeros([cases,d,d],w.dtype)
    v_flip = np.zeros([cases,d,d],w.dtype)
    v[:,0,0] = w[:,0]
    
    for i in range(1,d,1):
        v[:,i,i] = w[:,i]

        c = 0
        for k in range(i-1,-1,-1):
            tr = k
            col = i-k-1
            ind_now = ind_vine[k][c]

            ### To fix D-VINE
            # ind_array = np.array(vine.ind_edge_rel[tr])
            # ind_col = np.where(ind_array == col)
            # col = ind_col[0][0]

            if k == 0:
                tr1 = n-k
                col1 = n-i
                ind1 = r_matrix[tr1,col1] #- 1
                ind1 = np.where(nodes == ind1)
                ind1 = ind1[0][0]

                v2 = v[:,k,ind1][...,np.newaxis]
                v1 = v[:,k+1,i][...,np.newaxis]
                vv = np.concatenate((v1,v2),1)
                v[:,k,i] = copulainvccdf(cop_vine[tr][col],vv)
                
                tau, p_value = stats.kendalltau(vv[:,1],v[:,k,i])
                tau_corr[tr][col] = tau
                tau_bins[tr][col] = tau
            else:

                parent, inx1, inx2 = parent_var(k,ind_vine,ind_now)

                if ind_vine[k-1][ind_now[0]][0] != parent: 
                    v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
                else:
                    v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

                v1 = v[:,k+1,i][...,np.newaxis]
                vv = np.concatenate((v1,v2),1)

                ### try to fix d-vine sampling
                col = i-k-1

                if binning == False:
                    v[:,k,i] = copulainvccdf(cop_vine[tr][col],vv)
                    
                    tau, p_value = stats.kendalltau(vv[:,1],v[:,k,i])
                    tau_corr[tr][col] = tau
                    
                else:
                    
                    ind1 = parent
                    
                    if k == 1:
                        ind1 = np.where(nodes == ind1 +1)
                        ind1 = ind1[0][0]
                        bins = create_bins(v[:,k-1,ind1],n_bin)
                        val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                    else:
                        ind_par_now = ind_vine[k-1][ind_now[1]]
                        parent22, inx1, inx2 = parent_var(k-1,ind_vine,ind_par_now)  

                        ind1 = ind1 + k - 1
                        if (ind_vine[k-2][ind_par_now[0]][0] == parent22):
                            bins = create_bins(v[:,k-1,ind1],n_bin)
                            val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                        else:
                            bins = create_bins(v_flip[:,k-1,ind1],n_bin)
                            val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1

                    vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                    for bb in range(0,n_bin,1):
                        mask = np.where(val_to_bin == bb)
                        vv_bin = vv[mask[0],:]
                        
                        ### CDF FORCE UNIFORM
                        vv_bin_new = vv_bin
                        u_1, ex_u = mk_grid(tf.convert_to_tensor(50),vv_bin.dtype)
                        for zz in range(0,2,1):
                            vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],ex_u)
                        vv_bin = vv_bin_new
                        ###
                        
                        v[mask[0],k,i] = copulainvccdf(cop_vine[tr][col][bb],vv_bin)
                        
                        tau, p_value = stats.kendalltau(vv_bin[:,1],v[mask[0],k,i])
                        print('Tau value bin -',bb, '- is: ', tau)
                        tau_bins[tr][col][bb] = tau
#                         tau_binned.append(tau)
                        
                        corr = stats.pearsonr(vv_bin[:,1],v[mask[0],k,i])
                        print('Corr value  UV space, bin(',bb,')',corr[0])
            c += 1
        
        print('-----------')
        
        if i < d -1:
            for ii in range(1,i+1,1):
                for j in range(0,ii,1):
                    tr = j
                    col = ii-j-1

                    ind_now = ind_vine[j][ii-1-j]

                    if j == n-2:
                        ind_sup = ind_vine[j+1][0]
                    else:
                        ind_sup = ind_vine[j+1][i-1-j]

                    if j == 0:
                        tr1 = n-j
                        col1 = n-ii
                        ind1 = r_matrix[tr1,col1] #- 1
                        ind1 = np.where(nodes == ind1)
                        ind1 = ind1[0][0]

                        v2 = v[:,j,ind1][...,np.newaxis]
                    else:
                        parent1, inx1, inx2 = parent_var(j,ind_vine,ind_now)

                        if ind_vine[j-1][ind_now[0]][0] != parent1:
                            v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
                        else:
                            v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

                    v1 = v[:,j,ii][...,np.newaxis]

                    vv = np.concatenate((v1,v2),1)

                    parent, inx1, inx2 = parent_var(j+1,ind_vine,ind_sup)                
                    u_edge = {ind_now[0], ind_now[1]}

                    if (j == 0) | (binning == False):
                        if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                vv = np.concatenate((v2,v1),1)
                                v_flip[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
                            else:
                                v[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
                        else:
                            v[:,j+1,ii] = copulaccdf(cop_vine[tr][col],vv)
                    else:
                        if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                vv = np.concatenate((v2,v1),1)
                                flip_1 = True
                            else:
                                flip_1 = False
                        else:
                            flip_1 = False

                        ind1 = parent1
                        
                        if j == 1:
                            ind1 = np.where(nodes == ind1 +1)
                            ind1 = ind1[0][0]
                            bins = create_bins(v[:,j-1,ind1],n_bin)
                            val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                        else:
                            ind_par_now = ind_vine[j-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(j-1,ind_vine,ind_par_now)  

                            ind1 = ind1 + j - 1
                            if (ind_vine[j-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(v[:,j-1,ind1],n_bin)
                                val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                            else:
                                bins = create_bins(v_flip[:,j-1,ind1],n_bin)
                                val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1
                            
                        vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                        for bb in range(0,n_bin,1):
                            mask = np.where(val_to_bin == bb)
                            vv_bin = vv[mask[0],:]
                            
                            ### CDF FORCE UNIFORM
                            vv_bin_new = vv_bin
                            u_1, ex_u = mk_grid(tf.convert_to_tensor(50),vv_bin.dtype)
                            for zz in range(0,2,1):
                                vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],ex_u)
                            vv_bin = vv_bin_new
                            ###
                        
                            tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])

                            if flip_1 == True:
                                v_flip[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)
                            else:
                                v[mask[0],j+1,ii] = copulaccdf(cop_vine[tr][col][bb],vv_bin)
    
    u = np.reshape(v[:,0,:],np.shape(w))
    u1 = np.zeros(np.shape(u),u.dtype)
    c= 0
    for i in range(d-1,-1,-1):
        ind = r_matrix[i,i]-1
        u1[:,ind] = u[:,c]
        c += 1
    u = u1

    sample = np.zeros((cases,np.shape(u)[1]),w.dtype)
    for i in range(0,np.shape(u)[1],1):
        sample[:,i] = margininv(margin_vine[i],u[:,i])
    return sample, v, v_flip, tau_corr, tau_bins

# File: src/DVC_tensorflow/param/margin_cost.py
import tensorflow as tf
from utils.tensor_op import replace_nan_inf,replace_nan_with, replace_inf_with
from param.margin_pdf import *

############################################ GAUSSIAN COST FUNCTION  ######################################

@tf.function(experimental_relax_shapes=True)
def gaussian_cost(u,theta_par):
    p = gaussian_pdf(u,theta_par)
    p = replace_nan_with(p,tf.constant(1,u.dtype))
    eps = tf.constant(2.220446049250313e-16,u.dtype)
    err = -tf.math.reduce_sum(tf.math.log(p+eps),[0])
    return err

############################################ STUDENT COST FUNCTION  ######################################

# @tf.function(experimental_relax_shapes=True)
def student_cost(u,theta):
    p = student_pdf(u,theta)
    eps = tf.constant(2.220446049250313e-16,u.dtype)
    err = -tf.math.reduce_sum(tf.math.log(p+eps),[0])
    return err

############################################ CLAYTON COST FUNCTION  ######################################

@tf.function(experimental_relax_shapes=True)
def clayton_cost(u,theta_cla1):
    p = clayton_pdf(u,theta_cla1)
    p = replace_nan_with(p,tf.constant(1,u.dtype))
    p = replace_inf_with(p, tf.constant(p.dtype.max,u.dtype))
    eps = tf.constant(2.220446049250313e-16,u.dtype)
    err = -tf.math.reduce_sum(tf.math.log(p+eps),[0])
    return err

############################################ CLAYTON ROT 90 COST FUNCTION  ######################################

@tf.function(experimental_relax_shapes=True)
def claytonrot90_cost(u,theta_cla1):
    p = claytonrot90_pdf(u,theta_cla1)
    p = replace_nan_with(p,tf.constant(1,u.dtype))
    p = replace_inf_with(p, tf.constant(p.dtype.max,u.dtype))
    eps = tf.constant(2.220446049250313e-16,u.dtype)
    err = -tf.math.reduce_sum(tf.math.log(p+eps),[0])
    return err



# File: src/DVC_tensorflow/param/margin_fit.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from scipy import stats

from classes.objects import margin_obj
from param.margin_pdf import *
from param.margin_op import marginpdf


##################### PARAMETRIC MARGIN FITTING   ###############################

# @tf.function
def marginfit(fam,x):   
    theta = []
    if fam == 'norm':
        loc, scale = stats.norm.fit(x)
        theta = [loc,scale]
    elif fam == 'gamma':
        concentration, loc, rate = stats.gamma.fit(x)
        theta = [concentration, rate]
    return theta

def marginfit_all(x):
    if np.any(x < 0):
        families = ["norm"]
    else:
        families = ["gamma"]
    aic = []
    theta = []
    k = 0
    
    for i in families:
        theta1 = marginfit(i,x)
        iscont = True
        
        mar_pp  = margin_obj(i, theta1, iscont)
        
        logp = np.sum(np.log(marginpdf(mar_pp,x) + 1e-30))

        aic1 = 2*tf.cast(tf.size(theta1),x.dtype) - 2*logp
        aic = aic.append(aic1)
        theta.append(theta1)
        k = k+1
    
    ind_min = np.argmin(aic)
    return families[ind_min], theta[ind_min]

# File: src/DVC_tensorflow/param/margin_op.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
from scipy import stats
from utils.bijector import *


############################################ MARGIN PDF      ############################################

def marginpdf(marg,x): 
    logf_tmp = np.zeros(np.shape(x)[0],x.dtype)
    if marg.dist == 'norm': 
        loc = marg.theta[0]
        scale = marg.theta[1]
        norm_dis = tfd.Normal(loc,scale)
        logf_tmp = norm_dis.prob(x)      
    elif marg.dist == 'gamma':
        concentration = marg.theta[0]
        rate = 1/marg.theta[1]
        gamma_dis = tfd.Gamma(concentration,rate)
        logf_tmp = gamma_dis.prob(x)
    return logf_tmp


############################################ MARGIN CDF      ############################################

def margincdf(marg,x):
    Fp_tmp = np.zeros(np.shape(x)[0],x.dtype)

    if marg.dist == 'norm':
        loc = marg.theta[0]
        scale = marg.theta[1]
        Fp_tmp = stats.norm.cdf(x,loc,scale)
    elif marg.dist == 'gamma':
        concentration = marg.theta[0]
        rate = marg.theta[1]
        Fp_tmp = stats.gamma.cdf(x, concentration, 0, rate)
    return Fp_tmp


############################################ MARGIN INV      ############################################

def margininv(marg,x): 
    Fp_tmp = np.zeros(np.shape(x)[0],x.dtype)
    if marg.dist == 'norm':   #"gaussian"
        loc = marg.theta[0]
        scale = marg.theta[1]
        Fp_tmp = NormalCDF.forward(NormalCDF(loc,scale), x)    
    elif marg.dist == 'gamma':  #"gamma"
        concentration = marg.theta[0]
        rate = marg.theta[1]
        loc = 0
        Fp_tmp = stats.gamma.ppf(x, concentration, loc, rate)
    return Fp_tmp

# File: src/DVC_tensorflow/param/margin_pdf.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from scipy.special import gammaln
from scipy.stats import t
from utils.tensor_op import replace_nan_inf,update_tensor2D
import math as m

############################################ GAUSSIAN MARGIN PDF  ######################################

@tf.function(experimental_relax_shapes=True)
def gaussian_pdf(u,theta_par):
    norm_dis = tfd.Normal(loc=tf.constant(0.,u.dtype), scale=tf.constant(1.,u.dtype))
    x = norm_dis.quantile(u)
    p = tf.exp((2*theta_par*x[:,0,:]*x[:,1,:] - (theta_par**2) * (x[:,0,:]**2 + x[:,1,:]**2))/(2*(1-theta_par**2))) / tf.math.sqrt(1-theta_par**2)
    return p

############################################ CLAYTON MARGIN PDF  ######################################

@tf.function(experimental_relax_shapes=True)
def clayton_pdf(u,theta):
    p = (1 + theta) * (u[:,0,:] * u[:,1,:])**(-1-theta) * (u[:,0,:]**(-theta) + u[:,1,:]**(-theta) - 1)**(-1/theta-2)
    if tf.shape(tf.shape(theta))[0] == 0:
        theta = theta[...,tf.newaxis]
    cond = tf.math.equal(theta,0)
    ind = tf.where(tf.equal(cond,True))
    ind = tf.cast(ind,tf.int32)

    for i in ind:
        newval = tf.ones(tf.shape(u)[0],u.dtype)
        p = update_tensor2D(p, i[0] , newval)
    return p

############################################ CLAYTON ROT 90 MARGIN PDF  ######################################

@tf.function(experimental_relax_shapes=True)
def claytonrot90_pdf(u,theta):
    p = (1 + theta) * (u[:,0,:] * (1-u[:,1,:]))**(-1-theta) * (u[:,0,:]**(-theta) + (1-u[:,1,:])**(-theta) - 1)**(-1/theta-2)
    if tf.shape(tf.shape(theta))[0] == 0:
        theta = theta[...,tf.newaxis]
    cond = tf.math.equal(theta,0)
    ind = tf.where(tf.equal(cond,True))
    ind = tf.cast(ind,tf.int32)

    for i in ind:
        newval = tf.ones(tf.shape(u)[0],u.dtype)
        p = update_tensor2D(p, i[0] , newval)
    return p

############################################ STUDENT MARGIN PDF  ######################################

# @tf.function(experimental_relax_shapes=True)
def gammaln1(x):
    return tf.py_function(gammaln, [x], x.dtype)

# @tf.function(experimental_relax_shapes=True)
def tpdf(x,vk):
    term = tf.exp(gammaln1((vk + 1) / 2) - gammaln1(vk/2))
    pi = tf.constant(m.pi,x.dtype)
    y = term / (tf.math.sqrt(vk*pi) * (1 + (x**2) / vk) ** ((vk + 1)/2))
    return y

# @tf.function(experimental_relax_shapes=True)
def student_pdf(u,theta):
    if tf.shape(tf.shape(theta)) == 1:
        theta = theta[tf.newaxis,...]
    
    df = theta[:,1]
    loc = tf.constant(0,u.dtype)
    scale = tf.constant(1,u.dtype)
    pi = tf.constant(m.pi,u.dtype)

    x = tf.py_function(t.ppf, [u, theta[:,1], loc, scale], u.dtype)
    factor1 = gammaln1(theta[:,1]/2+1)
    factor2 = -gammaln1(theta[:,1]/2) - tf.math.log(pi) - tf.math.log(theta[:,1]) - tf.math.log(1-theta[:,0]**2)/2 - tf.math.log(tpdf(x[:,0,:],theta[:,1])) - tf.math.log(tpdf(x[:,1,:],theta[:,1]))

    factor3 = (-(theta[:,1]+2)/2) * tf.math.log(1 + (x[:,0,:]**2 + x[:,1,:]**2 - theta[:,0] * x[:,0,:] * x[:,1,:]) / (theta[:,1]*(1-theta[:,0]**2)))
    p = tf.exp(factor1 + factor2 + factor3)
    p = replace_nan_inf(p)
    return p


# File: src/DVC_tensorflow/plot/.ipynb_checkpoints/plot_vine-checkpoint.py
import matplotlib.pyplot as plt
import numpy as np
from tree.tree_op import edges_index

def plot_vine(typ, vine):
    d = vine.n_cop
    if typ == 'cdf':
        fig, ax = plt.subplots(d, d, sharex='col', sharey='row')
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.suptitle('VINE', weight='bold')
    elif typ == 'pdf':
        fig, ax = plt.subplots(d, d ) #, sharex='col', sharey='row')
        fig.subplots_adjust(hspace=0.7, wspace=0.7)
        fig.suptitle('VINE - COPULA PDF')
        
    n = len(vine.r_matrix) - 1
    
    ### PLOT CDF, PDF, ecc
    if typ == 'cdf':
        tr = 0
        for i in range(n,0,-1):
            ind_ee = edges_index(vine.E,vine.r_matrix,tr)
            c = 0
            for j in range(i-1,-1,-1):
                edg = ind_ee[c]
                ax[i,j].scatter(vine.theta[:,tr,edg[0]],vine.theta[:,tr,edg[1]], s=0.1 ,marker = '.')
                c += 1
            tr += 1
    elif typ == 'pdf':
        tr = 0
        for i in range(n,0,-1):
            c = 0
            for j in range(i-1,-1,-1):
                ax[i,j].imshow(vine.copulas[tr].pd_grid_uv[:,:,c], cmap="jet")
                c += 1
            tr += 1
    
    ## PLOT SUBTITLE
    for i in range(n,-1,-1):
        for j in range(i-1,-1,-1):
            str1 = '(' + str(vine.r_matrix[i,j]) + ',' + str(vine.r_matrix[j,j])
            c = 0
            for ii in range(i+1,n+1,1):
                if c == 0:
                    str1 = str1 + '|' + str (vine.r_matrix[ii,j])
                else:
                    str1 = str1 + ','  + str (vine.r_matrix[ii,j])
                c += 1
            str1 = str1 + ')'
            ax[i, j].title.set_text(str1)
    
    ## PLOT INSIDE SQUARES
    r1 = np.flip(vine.r_matrix)
    for i in range(0,n+1,1):
        for j in range(i,n+1,1):
            str1 = str(r1[i,j])
            ax[i, j].text(0.5, 0.5, str1,
                      fontsize=12, ha='center', weight = 'bold')
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    fig
    return

# File: src/DVC_tensorflow/plot/plot_vine.py
import matplotlib.pyplot as plt
import numpy as np
from tree.tree_op import edges_index

def plot_vine(typ, vine):
    d = vine.n_cop
    if typ == 'cdf':
        fig, ax = plt.subplots(d, d, sharex='col', sharey='row')
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.suptitle('VINE', weight='bold')
    elif typ == 'pdf':
        fig, ax = plt.subplots(d, d ) #, sharex='col', sharey='row')
        fig.subplots_adjust(hspace=0.7, wspace=0.7)
        fig.suptitle('VINE - COPULA PDF')
        
    n = len(vine.r_matrix) - 1
    
    ### PLOT CDF, PDF, ecc
    if typ == 'cdf':
        tr = 0
        for i in range(n,0,-1):
            ind_ee = edges_index(vine.E,vine.r_matrix,tr)
            c = 0
            for j in range(i-1,-1,-1):
                edg = ind_ee[c]
                ax[i,j].scatter(vine.theta[:,tr,edg[0]],vine.theta[:,tr,edg[1]], s=0.1 ,marker = '.')
                c += 1
            tr += 1
    elif typ == 'pdf':
        tr = 0
        for i in range(n,0,-1):
            c = 0
            for j in range(i-1,-1,-1):
                ax[i,j].imshow(vine.copulas[tr].pd_grid_uv[:,:,c], cmap="jet")
                c += 1
            tr += 1
    
    ## PLOT SUBTITLE
    for i in range(n,-1,-1):
        for j in range(i-1,-1,-1):
            str1 = '(' + str(vine.r_matrix[i,j]) + ',' + str(vine.r_matrix[j,j])
            c = 0
            for ii in range(i+1,n+1,1):
                if c == 0:
                    str1 = str1 + '|' + str (vine.r_matrix[ii,j])
                else:
                    str1 = str1 + ','  + str (vine.r_matrix[ii,j])
                c += 1
            str1 = str1 + ')'
            ax[i, j].title.set_text(str1)
    
    ## PLOT INSIDE SQUARES
    r1 = np.flip(vine.r_matrix)
    for i in range(0,n+1,1):
        for j in range(i,n+1,1):
            str1 = str(r1[i,j])
            ax[i, j].text(0.5, 0.5, str1,
                      fontsize=12, ha='center', weight = 'bold')
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    fig
    return

# File: src/DVC_tensorflow/pre_proc/.ipynb_checkpoints/preparation-checkpoint.py
import tensorflow as tf
import numpy as np
import sys
import tensorflow_probability as tfp
tfd = tfp.distributions
from utils.tensor_op import *

def prep_cop(x, vine1, sort_n):
    d = x.shape[1]
    #vine_families = tf.TensorArray(tf.string,size=d*d)
    if sort_n == 'sort':
        corr = tf.TensorArray(x.dtype,size=d*d)
        for i in tf.range(0,d,1):
            for j in tf.range(0,d,1):
                corr1 = tf.py_function(stats.kendalltau, [x[:,i],x[:,j]], x.dtype)
                corr1 = tf.math.abs(corr1)
                corr.write(3*i+j, corr1)
        corr = corr.stack()
        corr = tf.reshape(corr,[3,3])
        ord1 = tf.constant([0],tf.int32)
        for i in tf.range(1,d-3,-1):
            # Difference between 0:d e 0
            #ord1 = tf.reshape(ord1,[1,tf.shape(ord1)[0]])
            print(tf.constant(tf.range(0,d,1),shape=[1,d]),tf.constant(ord1,shape=[1,tf.shape(ord1)[0]]),ord1)
            ss = tf.sets.difference(tf.constant(tf.range(0,d,1),shape=[1,d]),tf.constant(ord1,shape=[1,tf.shape(ord1)[0]]))
            #ss = tf.sets.difference(tf.constant(tf.range(0,d,1),shape=[1,d]),ord1)
            ss = tf.sparse.to_dense(ss)
            ss = tf.transpose(ss)
            
            print(ord1[i-1][...,tf.newaxis])
            # Index of the difference to take in the correlation
            pp1 = tf.tile(ord1[i-1][...,tf.newaxis], [tf.shape(ss)[0]])
            pp1 = pp1[...,tf.newaxis]
            pp2 = tf.concat([pp1,ss],1)
        
            # Index of the difference to take in the correlation
            prova = tf.gather_nd(corr, pp2) 
            ind_max = tf.math.argmax(prova)
            ord1 = tf.concat([ord1,ss[ind_max]],0)
        # Difference between 0:d e ord1
        ss1 = tf.sets.difference(tf.constant(tf.range(0,d,1),shape=[1,d]),tf.constant(ord1,shape=[1,tf.shape(ord1)[0]]))
        ss1 = tf.sparse.to_dense(ss1)
    
        ord1 = tf.concat([ord1,ss1[0]],0) 
    else:
        ord1 = range(0,d,1)
    
    # Change columns corresponding with the order
    x = x[:,ord1]
    
    e = np.empty(x.shape,x.dtype)
    for i in range(0,d,1):   #d
        x_new1 = tf.constant(x[:,i])
        e[:,i] = prep_copula(x_new1,0).numpy()
        vine1.margin[i].ker = e[:,i]
    
    del x, x_new1
    return e
    
# @tf.function
def prep_copula(X_new,tim):
    # Preallocate
    u1 = tfd.Uniform(low=tf.constant(0,dtype=X_new.dtype), high=tf.constant(1,dtype=X_new.dtype))
    samples = u1.sample(tf.shape(X_new)[0])
    samples = tf.expand_dims(samples, 1)
    margin1 = tf.zeros(tf.shape(X_new),X_new.dtype)
    if tf.equal(tf.size(uniquetol(X_new,1e-5)),1):
        margin1 = X_new + samples[:, 0] * 1e-10
    else:
        ad1, idx = tf.unique(X_new)
        ad1 = tf.sort(ad1, axis=0)
        ad1 = ad1[...,tf.newaxis]
        
#         ad_add, id2 = tf.unique(ad1)
#         ad_add = tf.sort(ad_add, axis=0)
#         ad_add = tf.expand_dims(ad_add, 1)

        # Calculate diff vector
        ad = ad1[1:] - ad1[:-1]
#         ad = ad_add[1:] - ad_add[:-1]
        ad = tf.concat([ad, tf.expand_dims(ad[-1, :], 0)], 0)

        # Compute distances
        """
        Di2 = tf.zeros(tf.shape(X_new)[0],X_new.dtype)
        for pro in tf.range(0,tf.shape(ad_add)[0],1,tf.int32):
            ind22 = tf.where(tf.equal(X_new, ad_add[pro]))
            ind22 = tf.cast(ind22, tf.int32) #X_new.dtype
            pp1 = ad[pro]
            pp1 = tf.tile(pp1,[tf.shape(ind22)[0]])
            Di2 = tf.tensor_scatter_nd_update(Di2, ind22, pp1)
        """
        
        batch_size = tf.constant(1,tf.int32)
        if tf.math.less(tf.shape(X_new)[0],500):
            batch_size = tf.constant(1,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],1000):
            batch_size = tf.constant(2,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],4000):
            batch_size = tf.constant(10,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],10000):
            batch_size = tf.constant(20,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],20000):
            batch_size = tf.constant(50,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],100000):
            batch_size = tf.constant(100,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],200000):
            batch_size = tf.constant(200,tf.int32)
        
        #Di2 = tf.zeros(1,X_new.dtype) 
        batch_len = tf.shape(X_new)[0]/batch_size
        batch_len = tf.cast(batch_len,tf.int32)
        Di2 = tf.zeros(tf.shape(X_new)[0],X_new.dtype)
        
        for j in tf.range(0,batch_size,1):
            x_batch = X_new[batch_len*j:batch_len*(j+1)]
            if tf.math.equal(j,batch_size-1):
                x_batch = X_new[batch_len*j:]
                
#             ind22 = tf.where(tf.equal(x_batch, ad_add))
            ind22 = tf.where(tf.equal(x_batch, ad1))
            ind22_1 = ind22[:,1][...,tf.newaxis]
            
            
#             ra = tf.range(0,tf.shape(ad_add)[0],1,tf.int32)
            ra = tf.range(0,tf.shape(ad1)[0],1,tf.int32)
            pro = ind22[:,0]
            pro = pro[...,tf.newaxis]

            pp1 = tf.gather_nd(ad,pro)
            Di2 = tf.tensor_scatter_nd_update(Di2, ind22_1, tf.squeeze(pp1))
            # Add uniform noise proportional to NN distance
            #Di2 = tf.concat([Di2,Dii],0)

        #Di2 = Di2[1:]
        if tim == 0:
            margin1 = X_new + Di2 * samples[:, 0] * 1e-10  #Di1
        if tim == 1:
            margin1 = X_new + Di2 * samples[:, 0]
    return margin1
    


# File: src/DVC_tensorflow/pre_proc/.ipynb_checkpoints/transformation-checkpoint.py
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

# File: src/DVC_tensorflow/pre_proc/define_copulas.py
from vine_tree.tree_op import random_r_matrix_gen, prepare_regular, prepare_vine
from classes.objects import margin_obj, cop_par_obj

def define_copulas(vine_type, method, binning, n_bin, dim):

    ##################### THE FIRST PART IS RELATED TO THE REGULAR VINE ####################################

    if vine_type == 'r-vine':
        
        if method == 'matrix':
            
            ######### REGULAR MATRIX

            ### Example of r-matrix with 5 variables
            r_matrix = np.array([[2, 0, 0, 0, 0],
                                [5, 3, 0, 0, 0],
                                [4, 5, 1, 0, 0],
                                [1, 4, 5, 4, 0],
                                [3, 1, 4, 5, 5]])

            ### Example of r-matrix with 4 variables
    #         r_matrix = np.array([[3, 0, 0, 0],
    #                              [1, 4, 0, 0],
    #                              [2, 1, 2, 0],
    #                              [4, 2, 1, 1]])

            ### Example of r-matrix with 3 variables
    #         r_matrix = np.array([[3, 0, 0],
    #                              [2, 2, 0],
    #                              [1, 1, 1]])

            print(r_matrix)
            
        elif method == 'random':
            
            ##### RANDOM R-MATRIX
            r_matrix, ind_vine, nodes, E = random_r_matrix_gen(dim)
            print(r_matrix)

        ## Function that computes nodex, matrix edges and tree for the given r-matrix
        E, ind_vine, nodes, matrix_edges = prepare_regular(r_matrix)
        print('matrix_edges',matrix_edges)
        
        ### DEFINE MARGINS

        margin_fam1 = ['norm','norm','norm','norm','norm']
        theta_fam1 = [[0,1],[0,1],[0,1],[0,1],[0,1]]

        ### Example of margins with gaussian and gamma distributions
    #     margin_fam1 = ['norm','gamma','norm','gamma','norm']
    #     theta_fam1 = [[0,1],[2,4],[0,1],[2,4],[0,1]]

        is_cont1 = [True,True,True,True,True]

        margin_vine = []
        for i in range(0,len(margin_fam1),1):
            mar_p = margin_obj(margin_fam1[i], theta_fam1[i], is_cont1[i])
            margin_vine.append(mar_p)

        for i in range(0,len(margin_fam1),1):
            print(margin_vine[i].dist, end =' ')
            print(margin_vine[i].theta, end =' ')
        
        ######################## DEFINE COPULAS  ###################
        if not binning:
            
            ### Example with 4 variables
    #         margin_cop1 = [['gaussian','gaussian','gaussian','gaussian'],
    #                        ['gaussian','gaussian','gaussian'],
    #                        ['gaussian','gaussian'],
    #                        ['gaussian']]
            
    #         theta_cop1 = [[0.3, 0.5, 0.7, 0.8],
    #                       [0.5, 0.8, 0.4],
    #                       [0.5, 0.3],
    #                       [0.9]]

            ### Example with 5 variables
    #         margin_cop1 = [['gaussian','gaussian','gaussian','gaussian','gaussian'],
    #                        ['gaussian','gaussian','gaussian','gaussian'],
    #                        ['gaussian','gaussian','gaussian'],
    #                        ['gaussian','gaussian'],
    #                        ['gaussian']]
            
    #         theta_cop1 = [[0.3, 0.5, 0.7, 0.8, -0.8],
    #                       [0.5, 0.8, 0.4, -0.2],
    #                       [0.5, 0.3, -0.7],
    #                       [0.9, 0.6],
    #                       [0.5]]
            
            ### Example with 4 variables and different distributions
            margin_cop1 = [['gaussian','student','clayton','gaussian'],
                        ['student','clayton','gaussian'],
                        ['student','gaussian'],
                        ['clayton']]

            theta_cop1 = [[0.3, [0,0.2], 0.7,  -0.8],
                        [[-0.8,0.2], 4.5,  -0.2],
                        [[0,0.5],  -0.7],
                        [0.9]]
            
            ### Example with 3 variables
    #         margin_cop1 = [['gaussian','gaussian','gaussian'],
    #                        ['gaussian','gaussian'],
    #                        ['gaussian']]

    #         theta_cop1 = [[0.7, 0.8, 0.9],
    #                       [0.6, 0.5],
    #                       [0.7]]

            ### Example with 3 variables
            # margin_cop1 = [['clayton','clayton','clayton'],
            #                ['clayton','clayton'],
            #                ['clayton']]

            # theta_cop1 = [[3.7, 4.8, 5.9],
            #               [6.6, 2.5],
            #               [5.7]]
        else:
            
            ### If binning, you have to define each type of variable for each bin. This is an example with 4 variables and 3 bins, remember 
            ### that the first level is not binned

            margin_cop1 = [['gaussian','gaussian','gaussian','gaussian'],
                        [['gaussian','gaussian','gaussian'],['gaussian','gaussian','gaussian'],['gaussian','gaussian','gaussian']],
                        [['gaussian','gaussian','gaussian'],['gaussian','gaussian','gaussian']],
                        [['gaussian','gaussian','gaussian']]]

            theta_cop1 = [[0.3, 0.5, 0.7, 0.8],
                        [[0.3, 0.4, 0.5],[0.6, 0.7, 0.8],[0.2, 0.3, 0.4]],
                        [[0.3, 0.4, 0.5],[0.3, 0.5, 0.9]],
                        [[0.2, 0.5, 0.9]]] #0.2,0.5,0.9
            
            ### Example with 3 variables
    #         margin_cop1 = [['gaussian','gaussian','gaussian'],
    #                        [['gaussian','gaussian','gaussian'],['gaussian','gaussian','gaussian']],
    #                       [['gaussian','gaussian','gaussian']]]

    #         theta_cop1 = [[0.7, 0.8, 0.9],
    #                       [[0.3, 0.4, 0.5],[0.6, 0.7, 0.8]],
    #                       [[0.2, 0.5, 0.9]]]

            ### Example with 2 variables
    #         margin_cop1 = [['gaussian','gaussian'],
    #                       [['gaussian','gaussian','gaussian']]]

    #         theta_cop1 = [[0.7, 0.8],
    #                       [[0.2, 0.5, 0.9]]]


        d = len(r_matrix)
        cop_vine = []
        for tr in range(0,d-1,1):
            cop_vine1 = []
            for col in range(0,d-1-tr,1):
                if (tr == 0) | (binning == False):
                    cop_p = cop_par_obj(margin_cop1[tr][col],theta_cop1[tr][col])
                    cop_vine1.append(cop_p)
                else:
                    cop_vine11 = []
                    for bb in range(0,n_bin,1):
                        cop_p = cop_par_obj(margin_cop1[tr][col][bb],theta_cop1[tr][col][bb])
                        cop_vine11.append(cop_p)
                    cop_vine1.append(cop_vine11)
            cop_vine.append(cop_vine1)

        for tr in range(0,d-1,1):
            for col in range(0,d-1-tr,1):
                if (tr == 0) | (binning == False):
                    print('edge: {} '.format(matrix_edges[tr][col]), 'cop family: {}'.format(cop_vine[tr][col].family), 'theta: {}'.format(cop_vine[tr][col].theta))
                else:
                    for bb in range(0,n_bin,1):
                        print('edge: {} '.format(matrix_edges[tr][col]),'bin: {} '.format(bb), 'cop family: {}'.format(cop_vine[tr][col][bb].family), 'theta: {}'.format(cop_vine[tr][col][bb].theta))

        ################# IF YOU WANT TO USE C-VINE OR D-VINE    ################################### 
    elif (vine_type == 'c-vine') | (vine_type == 'd-vine'):
        
        
        r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, dim)

        print(r_matrix)

        # binning = False
        # n_bin = 3
        
        ########## DEFINE MARGINS
        
        margin_vine = []
        for i in range(0,dim,1):
            mar_p = margin_obj('norm', [0,1], True)
            margin_vine.append(mar_p)

        for i in range(0,dim,1):
            print(margin_vine[i].dist, end =' ')
            print(margin_vine[i].theta, end =' ')

        ############## DEFINE COPULAS
        # NN = 0
        tr = 0
        cop_vine = []
        for i in range(dim,1,-1):
            cop_vine1 = []
            for j in range(0,i-1,1):
                if (tr == 0) | (binning == False):
                    #### You can decomment the following to set only a specific level as independent

        #             if tr == NN:
        #                 cop_p = cop_par_obj('ind',[])  #
        #                 cop_vine1.append(cop_p)
        #             else:
                    cop_p = cop_par_obj('clayton',4.5)  #
                    cop_vine1.append(cop_p)
                else:
                    cop_vine11 = []
                    for bb in range(0,n_bin,1):
                        cop_p = cop_par_obj('gaussian',0.9)
                        cop_vine11.append(cop_p)
                    cop_vine1.append(cop_vine11)
            cop_vine.append(cop_vine1)
            tr += 1
        
    #     margin_cop1 = [['gaussian','student','clayton','gaussian'],
    #                    ['student','clayton','gaussian'],
    #                    ['student','gaussian'],
    #                    ['clayton']]
            
    #     theta_cop1 = [[0.3, [0,0.2], 0.7,  -0.8],
    #                   [[-0.8,0.2], 4.5,  -0.2],
    #                   [[0,0.5],  -0.7],
    #                   [0.9]]
        
    #     d = len(r_matrix)
    #     cop_vine = []
    #     for tr in range(0,d-1,1):
    #         cop_vine1 = []
    #         for col in range(0,d-1-tr,1):
    #             if (tr == 0) | (binning == False):
    #                 cop_p = cop_par_obj(margin_cop1[tr][col],theta_cop1[tr][col])
    #                 cop_vine1.append(cop_p)
    #             else:
    #                 cop_vine11 = []
    #                 for bb in range(0,n_bin,1):
    #                     cop_p = cop_par_obj(margin_cop1[tr][col][bb],theta_cop1[tr][col][bb])
    #                     cop_vine11.append(cop_p)
    #                 cop_vine1.append(cop_vine11)
    #         cop_vine.append(cop_vine1)

        d = len(r_matrix)
        for tr in range(0,d-1,1):
            for col in range(0,d-1-tr,1):
                if (tr == 0) | (binning == False):
                    print('edge: {} '.format(matrix_edges[tr][col]), 'cop family: {}'.format(cop_vine[tr][col].family), 'theta: {}'.format(cop_vine[tr][col].theta))
                else:
                    for bb in range(0,n_bin,1):
                        print('edge: {} '.format(matrix_edges[tr][col]),'bin: {} '.format(bb), 'cop family: {}'.format(cop_vine[tr][col][bb].family), 'theta: {}'.format(cop_vine[tr][col][bb].theta))
    
    return r_matrix, cop_vine, ind_vine, nodes, matrix_edges, margin_vine

# File: src/DVC_tensorflow/pre_proc/preparation.py
import tensorflow as tf
import numpy as np
import sys
import tensorflow_probability as tfp
tfd = tfp.distributions
from utils.tensor_op import *

def prep_cop(x, vine1, sort_n):
    d = x.shape[1]
    if sort_n == 'sort':
        corr = tf.TensorArray(x.dtype,size=d*d)
        for i in tf.range(0,d,1):
            for j in tf.range(0,d,1):
                corr1 = tf.py_function(stats.kendalltau, [x[:,i],x[:,j]], x.dtype)
                corr1 = tf.math.abs(corr1)
                corr.write(3*i+j, corr1)
        corr = corr.stack()
        corr = tf.reshape(corr,[3,3])
        ord1 = tf.constant([0],tf.int32)
        for i in tf.range(1,d-3,-1):
            # Difference between 0:d e 0
            #ord1 = tf.reshape(ord1,[1,tf.shape(ord1)[0]])
            ss = tf.sets.difference(tf.constant(tf.range(0,d,1),shape=[1,d]),tf.constant(ord1,shape=[1,tf.shape(ord1)[0]]))
            #ss = tf.sets.difference(tf.constant(tf.range(0,d,1),shape=[1,d]),ord1)
            ss = tf.sparse.to_dense(ss)
            ss = tf.transpose(ss)
            
            # Index of the difference to take in the correlation
            pp1 = tf.tile(ord1[i-1][...,tf.newaxis], [tf.shape(ss)[0]])
            pp1 = pp1[...,tf.newaxis]
            pp2 = tf.concat([pp1,ss],1)
        
            # Index of the difference to take in the correlation
            prova = tf.gather_nd(corr, pp2) 
            ind_max = tf.math.argmax(prova)
            ord1 = tf.concat([ord1,ss[ind_max]],0)
        # Difference between 0:d e ord1
        ss1 = tf.sets.difference(tf.constant(tf.range(0,d,1),shape=[1,d]),tf.constant(ord1,shape=[1,tf.shape(ord1)[0]]))
        ss1 = tf.sparse.to_dense(ss1)
    
        ord1 = tf.concat([ord1,ss1[0]],0) 
    else:
        ord1 = range(0,d,1)
    
    # Change columns corresponding with the order
    x = x[:,ord1]
    
    e = np.empty(x.shape,x.dtype)
    for i in range(0,d,1):   #d
        x_new1 = tf.constant(x[:,i])
        e[:,i] = prep_copula(x_new1,0).numpy()
        vine1.margin[i].ker = e[:,i]
    
    # del x_new1
    return e
    
# @tf.function
def prep_copula(X_new,tim):
    # Preallocate
    u1 = tfd.Uniform(low=tf.constant(0,dtype=X_new.dtype), high=tf.constant(1,dtype=X_new.dtype))
    
    samples = u1.sample(tf.shape(X_new)[0])

    samples = tf.expand_dims(samples, 1)
    margin1 = tf.zeros(tf.shape(X_new),X_new.dtype)
    if tf.equal(tf.size(uniquetol(X_new,1e-5)),1):
        margin1 = X_new + samples[:, 0] * 1e-10
    else:
        ad1, idx = tf.unique(X_new)
        ad1 = tf.sort(ad1, axis=0)
        ad1 = ad1[...,tf.newaxis]

        # Calculate diff vector
        ad = ad1[1:] - ad1[:-1]
        ad = tf.concat([ad, tf.expand_dims(ad[-1, :], 0)], 0)

        # Compute distances
        """
        Di2 = tf.zeros(tf.shape(X_new)[0],X_new.dtype)
        for pro in tf.range(0,tf.shape(ad_add)[0],1,tf.int32):
            ind22 = tf.where(tf.equal(X_new, ad_add[pro]))
            ind22 = tf.cast(ind22, tf.int32) #X_new.dtype
            pp1 = ad[pro]
            pp1 = tf.tile(pp1,[tf.shape(ind22)[0]])
            Di2 = tf.tensor_scatter_nd_update(Di2, ind22, pp1)
        """
        
        batch_size = tf.constant(1,tf.int32)
        if tf.math.less(tf.shape(X_new)[0],500):
            batch_size = tf.constant(1,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],1000):
            batch_size = tf.constant(2,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],4000):
            batch_size = tf.constant(10,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],10000):
            batch_size = tf.constant(20,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],20000):
            batch_size = tf.constant(50,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],100000):
            batch_size = tf.constant(100,tf.int32)
        elif tf.math.less(tf.shape(X_new)[0],200000):
            batch_size = tf.constant(200,tf.int32)
        
        #Di2 = tf.zeros(1,X_new.dtype) 
        batch_len = tf.shape(X_new)[0]/batch_size
        batch_len = tf.cast(batch_len,tf.int32)
        Di2 = tf.zeros(tf.shape(X_new)[0],X_new.dtype)
        
        for j in tf.range(0,batch_size,1):
            x_batch = X_new[batch_len*j:batch_len*(j+1)]
            if tf.math.equal(j,batch_size-1):
                x_batch = X_new[batch_len*j:]
                
#             ind22 = tf.where(tf.equal(x_batch, ad_add))
            ind22 = tf.where(tf.equal(x_batch, ad1))
            ind22_1 = ind22[:,1][...,tf.newaxis]
            
            
#             ra = tf.range(0,tf.shape(ad_add)[0],1,tf.int32)
            ra = tf.range(0,tf.shape(ad1)[0],1,tf.int32)
            pro = ind22[:,0]
            pro = pro[...,tf.newaxis]

            pp1 = tf.gather_nd(ad,pro)
            Di2 = tf.tensor_scatter_nd_update(Di2, ind22_1, tf.squeeze(pp1))
            # Add uniform noise proportional to NN distance
            #Di2 = tf.concat([Di2,Dii],0)

        #Di2 = Di2[1:]
        if tim == 0:
            margin1 = X_new + Di2 * samples[:, 0] * 1e-10  #Di1
        if tim == 1:
            margin1 = X_new + Di2 * samples[:, 0]
    return margin1


# File: src/DVC_tensorflow/pre_proc/transformation.py
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

# File: src/DVC_tensorflow/pred/.ipynb_checkpoints/prediction-checkpoint.py
import tensorflow as tf
import numpy as np
from utils.tensor_op import moving_average
from utils.tensor_op import create_points
from utils.tensor_op import replace_nan_inf

################## PREDICT VINE ########################

def predict_vine(x, vine, dim, exp_dim):

    points = create_points(x,dim,exp_dim)
    
    p, p_cop = vine.evaluation(points)
    
    p1 = tf.reshape(p,[x.shape[0],exp_dim])
    p1 = replace_nan_inf(p1)
    
    min_dim = tf.math.reduce_min(x[:,dim])
    max_dim = tf.math.reduce_max(x[:,dim])
    y_vec = tf.linspace(min_dim-2e-16+1e-5,max_dim+2e-16,exp_dim)
    
    y_vec = tf.cast(y_vec,tf.float64)
    mov_p = tf.TensorArray(p1.dtype, size=tf.shape(p1)[0])
    for i in tf.range(0,tf.shape(p1)[0],1,tf.int32): #tf.shape(p1)[1]
#         movag = moving_average(p1[i,:],4)
        movag = smooth(p1[i,:].numpy(),4,'flat')
        mov_p = mov_p.write(i,movag)

    mov_p = mov_p.stack()
    mov_p = mov_p[:,3:] #tf.transpose(mov_p)
    
#     print(mov_p[21,:])
    
    ############### Y MAXIMUM LIKELIHOOD  ##################

    y_diff = y_vec[1:] - y_vec[:-1]
    y_diff = tf.concat([y_diff, tf.expand_dims(y_diff[-1], 0)], 0)

    ind_max1 = tf.math.argmax(mov_p,1)   #mov_p
    ind_max1 = ind_max1[...,tf.newaxis]
    y_ml = tf.gather_nd(y_vec,ind_max1)

    ############### Y EXPECTATION MAXIMIZATION  ##################

    y_diff1 = y_diff[...,tf.newaxis]
    y_diff_tile = tf.tile(y_diff1, [1, tf.shape(p1)[0]])

    q1 = tf.math.reduce_sum(mov_p*tf.transpose(y_diff_tile),1)
    q2 = q1[...,tf.newaxis]
    q1 = tf.tile(q2,[1,tf.shape(p1)[1]])
    q = mov_p/q1
#     print(q[455,:])

    y_tmp = y_vec*y_diff
    y_tmp1 = y_tmp[...,tf.newaxis]
    y_tmp1 = tf.tile(y_tmp1,[1,tf.shape(p1)[0]])

    y_em = tf.math.reduce_sum(q*tf.transpose(y_tmp1),1)
    
    return p, y_ml, y_em

###################  PREDICT RESPONSE   ######################

# @tf.function
def predict_response(p1, y_vec):
    y_vec = tf.cast(y_vec,tf.float64)
    mov_p = tf.TensorArray(p1.dtype, size=tf.shape(p1)[0])
    for i in tf.range(0,tf.shape(p1)[0],1,tf.int32): #tf.shape(p1)[1]
#         movag = moving_average(p1[i,:],4)
        movag = smooth(p1[i,:].numpy(),4,'flat')
        mov_p = mov_p.write(i,movag)

    mov_p = mov_p.stack()
    mov_p = mov_p[:,3:] #tf.transpose(mov_p)
    
    ############### Y MAXIMUM LIKELIHOOD  ##################

    y_diff = y_vec[1:] - y_vec[:-1]
    y_diff = tf.concat([y_diff, tf.expand_dims(y_diff[-1], 0)], 0)

    ind_max1 = tf.math.argmax(mov_p,1)   #mov_p
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

# File: src/DVC_tensorflow/pred/prediction.py
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

# File: src/DVC_tensorflow/sampling/.ipynb_checkpoints/vine_sample-checkpoint.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np

from utils.prob_op import kernel_cdf
from evalu.vine_eval import evaluate_points
from utils.interpolation import interp1d_np
from utils.dataset_op import create_bins, check_bins
from vine_tree.tree_op import parent_var
from pre_proc.preparation import prep_copula
from pre_proc.transformation import Transform
from param.cond_copula import *


#################################### SAMPLING FROM NON-PARAMETRIC COPULA #######################################

################# TRY CON FLAG

def vine_copula_sample(vine,cases):
    d = len(vine.r_matrix)
    n = len(vine.r_matrix) -1
    depth = vine.vine_depth   ### Should I use this and how??

    w = np.random.uniform(1e-3,0.999,(cases,d))   #0,1,(cases,d))
    w = w.astype(vine.data_u.dtype)

    v = np.zeros([cases,d,d],w.dtype)
    v_flip = np.zeros([cases,d,d],w.dtype)
    v[:,0,0] = w[:,0]
    u1 = vine.grid_u.ax1
    u2 = vine.grid_u.ax2

    flip_flag1 = []
    for ii in range(1,d-1,1):
        flip_flag2 = []

        for j in range(0,ii,1):
            flip_flag2.append([])
        flip_flag1.append(flip_flag2)


    for i in range(1,d,1):  #d
        v[:,i,i] = w[:,i]

        c = 0
        for k in range(i-1,-1,-1):
            tr = k
            col = i-k-1
            ind_now = vine.ind_vine[k][c]

            ind_array = np.array(vine.ind_edge_rel[tr])
            ind_col = np.where(ind_array == col)
            col = ind_col[0][0]

            if k == 0:
                tr1 = n-k
                col1 = n-i
                ind1 = vine.r_matrix[tr1,col1] #- 1
                ind1 = np.where(vine.nodes == ind1)
                ind1 = ind1[0][0]

                v2 = v[:,k,ind1][...,np.newaxis]
                v1 = v[:,k+1,i][...,np.newaxis]
                vv = tf.convert_to_tensor(np.concatenate((v2,v1),1))

                cdf_grid = vine.copulas[tr].cdf[:,:,col]
                v[:,k,i] = kerncopccdfinv(vv, cdf_grid, u1,u2)

            else:

                parent, inx1, inx2 = parent_var(k,vine.ind_vine,ind_now)

                if vine.ind_vine[k-1][ind_now[0]][0] != parent: 
                    v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
                else:
                    v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

                v1 = v[:,k+1,i][...,np.newaxis]
    #                 vv = np.concatenate((v1,v2),1)

                ## CHANGED
                vv = np.concatenate((v2,v1),1)

                if vine.binning == False:
                    cdf_grid = vine.copulas[tr].cdf[:,:,col]
                    vv = tf.convert_to_tensor(vv)
                    v[:,k,i] = kerncopccdfinv(vv, cdf_grid, u1,u2)

                else:
                    
                    ind1 = parent
                    
                    if k == 1:
                        ind1 = np.where(vine.nodes == ind1 +1)
                        ind1 = ind1[0][0]
                        bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                        val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                    else:
                        ind_par_now = vine.ind_vine[k-1][ind_now[1]]
                        parent22, inx1, inx2 = parent_var(k-1,vine.ind_vine,ind_par_now)  

                        ind1 = ind1 + k - 1
                        if (vine.ind_vine[k-2][ind_par_now[0]][0] == parent22):
                            bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                            val_to_bin = check_bins(v[:,k-1,ind1],bins)
                        else:
                            bins = create_bins(v_flip[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1
                            val_to_bin = check_bins(v_flip[:,k-1,ind1],bins)
                    
#                     if (vine.ind_vine[k-1][ind_now[0]][0] == parent) | (k == 1): 
#                         if k == 1:
#                             ind1 = np.where(vine.nodes == ind1 +1)
#                             ind1 = ind1[0][0]
#                         else:
#                             ind1 = ind1 + k - 1
#                         bins = create_bins(v[:,k-1,ind1],vine.n_bin)
#                         val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
#                     else:
#                         ind1 = ind1 + k -1
#                         bins = create_bins(v_flip[:,k-1,ind1],vine.n_bin)
#                         val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1
                    
                    
                    vv_all = np.zeros(np.shape(vv)[0],w.dtype)
                    for bb in range(0,vine.n_bin,1):
                        cdf_grid = vine.copulas[tr].cdf[:,:,col,bb]
                        mask = np.where(val_to_bin == bb)
                        vv_bin = vv[mask[0],:]
                        
                        ### CDF FORCE UNIFORM
                        vv_bin_new = vv_bin
                        for zz in range(0,2,1):
                            vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],vine.grid_u.ex)
                        vv_bin = vv_bin_new
                        ###
                        vv_bin = tf.convert_to_tensor(vv[mask[0],:])
                        
                        v[mask[0],k,i] = kerncopccdfinv(vv_bin, cdf_grid, u1,u2)
#                         corr = stats.pearsonr(vv_bin[:,1],v[mask[0],k,i])
#                         print('Corr value  UV space: ',corr[0])
            c += 1


        if i < d-1: 
            cc1 = 0
            for ii in range(1,i+1,1):

                cc2 = 0
                for j in range(0,ii,1):
                    tr = j
                    col = ii-j-1

                    ind_now = vine.ind_vine[j][ii-1-j]

                    if j == n-2:
                        ind_sup = vine.ind_vine[j+1][0]
                    else:
                        ind_sup = vine.ind_vine[j+1][i-1-j]

                    flip_flag = False

                    parent, inx1, inx2 = parent_var(j+1,vine.ind_vine,ind_sup)                
                    u_edge = {ind_now[0], ind_now[1]}
                    if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                flip_flag = True

                    flag = False
                    if i >1:
                        for el in flip_flag1[cc1][cc2]:
                            if el == flip_flag:
                                flag = True
                        if flag == False:
                            flip_flag1[cc1][cc2].append(flip_flag)
                    else:
                        flip_flag1[cc1][cc2].append(flip_flag)


                    if flag == False:

                        ind_array = np.array(vine.ind_edge_rel[tr])
                        ind_col = np.where(ind_array == col)
                        ind_fin = ind_col[0][0]
            #             print('ind_fin',col)
                        if (vine.ind_edge_rel[tr][ind_fin+1] == col) & (vine.flip_flag[tr][ind_fin+1] == flip_flag):
                            col = ind_fin + 1
                        else:
                            col = ind_fin


                        if j == 0:
                            tr1 = n-j
                            col1 = n-ii
                            ind1 = vine.r_matrix[tr1,col1] #- 1
                            ind1 = np.where(vine.nodes == ind1)
                            ind1 = ind1[0][0]

                            v2 = v[:,j,ind1][...,np.newaxis]
                        else:
                            parent1, inx1, inx2 = parent_var(j,vine.ind_vine,ind_now)

#                             if ind_now[0] != parent1:
                            if vine.ind_vine[j-1][ind_now[0]][0] != parent1:
                                v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
                            else:
                                v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

                        v1 = v[:,j,ii][...,np.newaxis]
                        
                        if flip_flag == False:
                            data_u = np.concatenate((v2,v1),1)
                        else:
                            data_u = np.concatenate((v1,v2),1)
                        
#                         data_u = np.concatenate((v1,v2),1)
#                         data_u = np.concatenate((v2,v1),1)

                        if (j==0) | (vine.binning == False):
                            
                            data_u = data_u[...,np.newaxis]
                            trans = Transform(1)

                            ## Transform data
                            data_s = trans.forward_u(data_u)
                            data_x = trans.forward_s(data_s)

                            batch_size = tf.constant(2,tf.int32)
                            pd_grid_uv = vine.copulas[tr].pd_grid_uv[:,:,col]
                            cdf1 = vine.copulas[tr].cdf[:,:,col]

                            ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s[:,:,0],vine.grid_s.min,vine.grid_s.max,cdf1,axis=-2)
        #                         mar_p1, mar_s1 = kernel_cdf(ccdf_data, vine.grid_u.ex)

                            pd_points, ccdf_points = evaluate_points(data_s[:,:,0], batch_size, vine.grid_s, cdf1, pd_grid_uv)    

                            # Update Fp
                            interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_points, vine.grid_u.ex)
        #                         interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
                            if flip_flag == False:
                                v[:,j+1,ii] = interp_cdf_poi
                            else:
                                v_flip[:,j+1,ii] = interp_cdf_poi

                        else: #binning
                            
                            ind1 = parent1
                            
                            if j == 1:
                                ind1 = np.where(vine.nodes == ind1 +1)
                                ind1 = ind1[0][0]
                                bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                                val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                            else:
                                ind_par_now = vine.ind_vine[j-1][ind_now[1]]
                                parent22, inx1, inx2 = parent_var(j-1,vine.ind_vine,ind_par_now)  

                                ind1 = ind1 + j - 1
                                if (vine.ind_vine[j-2][ind_par_now[0]][0] == parent22):
                                    bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                                    val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                                    val_to_bin = check_bins(v[:,j-1,ind1],bins)
                                else:
                                    bins = create_bins(v_flip[:,j-1,ind1],vine.n_bin)
                                    val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1
                                    val_to_bin = check_bins(v_flip[:,j-1,ind1],bins)
                            
#                             if (vine.ind_vine[j-1][ind_now[0]][0] == parent1) | (j == 1): 
#                                 if j == 1:
#                                     ind1 = np.where(vine.nodes == ind1 +1)
#                                     ind1 = ind1[0][0]
#                                 else:
#                                     ind1 = ind1 + j - 1
#                                 bins = create_bins(v[:,j-1,ind1],vine.n_bin)
#                                 val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
#                             else:
#                                 ind1 = ind1 + j -1
#                                 bins = create_bins(v_flip[:,j-1,ind1],vine.n_bin)
#                                 val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1
                            
                            vv_all = np.zeros(np.shape(vv)[0],w.dtype)
                            for bb in range(0,vine.n_bin,1):
                                mask = np.where(val_to_bin == bb)

                                batch_size = tf.constant(1,tf.int32)
                                pd_grid_uv = vine.copulas[tr].pd_grid_uv[:,:,col,bb]
                                cdf1 = vine.copulas[tr].cdf[:,:,col,bb]
                                
                                data_u_bin = data_u[mask[0],:]
                                
                                ### CDF FORCE UNIFORM
                                vv_bin_new = data_u_bin
                                for zz in range(0,2,1):
                                    vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],data_u_bin[:,zz],vine.grid_u.ex)
                                data_u_bin = vv_bin_new[...,np.newaxis]
                                ###
                                
                                trans = Transform(1)

                                ## Transform data
                                data_s_bin = trans.forward_u(data_u_bin)
                                data_x_bin = trans.forward_s(data_s_bin)
                                
#                                 data_s_bin = tf.gather_nd(data_s[:,:,0],mask[0][...,tf.newaxis])

                                ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin[:,:,0],vine.grid_s.min,vine.grid_s.max,cdf1,axis=-2)
        #                             mar_p1, mar_s1 = kernel_cdf(ccdf_data, vine.grid_u.ex)

                                pd_points, ccdf_points = evaluate_points(data_s_bin[:,:,0], batch_size, vine.grid_s, cdf1, pd_grid_uv)    

                                # Update Fp
                                interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_points, vine.grid_u.ex)
        #                             interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
                                if flip_flag == False:
                                    v[mask[0],j+1,ii] = interp_cdf_poi
                                else:
                                    v_flip[mask[0],j+1,ii] = interp_cdf_poi
                    cc2 += 1
                cc1 += 1

    u = np.reshape(v[:,0,:],np.shape(w))
    u1 = np.zeros(np.shape(u),u.dtype)

    u_ax = vine.grid_u.ax1
    u_ax1 = np.tile(u_ax[...,np.newaxis],[1,np.shape(u)[0]]).T
    gr_diff = vine.grid_u.diff1

    c= 0
    for i in range(d-1,-1,-1):
        ind = vine.r_matrix[i,i]-1
        u1[:,ind] = u[:,c]

        ###### COMMENT WHEN COMPUTING INFORMATION OTHERWISE ALL WRONG
    #         u_p1 = u[:,c][...,np.newaxis]
    #         u_upd = np.tile(u_p1,[1,np.shape(u_ax)[0]])

    #         u_diff = np.abs(u_ax1-u_upd)
    #         ind1 = np.argmin(u_diff,1)

    #         diff_val = np.take(gr_diff,ind1)

    #         u1[:,ind] = u[:,c] + diff_val*np.random.uniform(0.,1.,np.shape(diff_val)[0])

        c += 1
    u = u1

    sample1 = np.zeros([cases,d],w.dtype)
    for i in tf.range(0,d,1,tf.int32):
        mar_s1 = vine.Mar_G[i][0]
        mar_p1 = vine.Mar_G[i][1]
        sample1_pro = tfp.math.interp_regular_1d_grid(u[:,i],tf.math.reduce_min(mar_p1),tf.math.reduce_max(mar_p1),mar_s1)
        sample1[:,i] = prep_copula(sample1_pro,0).numpy()
    
    return sample1

# def vine_copula_sample(vine,cases):
#     d = len(vine.r_matrix)
#     n = len(vine.r_matrix) -1
#     depth = vine.vine_depth   ### Should I use this and how??

#     w = np.random.uniform(0,1,(cases,d))
#     w = w.astype(vine.data_u.dtype)

#     v = np.zeros([cases,d,d],w.dtype)
#     v_flip = np.zeros([cases,d,d],w.dtype)
#     v[:,0,0] = w[:,0]
#     u1 = vine.grid_u.ax1
#     u2 = vine.grid_u.ax2

#     flip_flag1 = []
#     for ii in range(1,d-1,1):
#         flip_flag2 = []

#         for j in range(0,ii,1):
#             flip_flag2.append([])
#         flip_flag1.append(flip_flag2)


#     for i in range(1,d,1):  #d
#         v[:,i,i] = w[:,i]

#         c = 0
#         for k in range(i-1,-1,-1):
#             tr = k
#             col = i-k-1
#             ind_now = vine.ind_vine[k][c]

#             ind_array = np.array(vine.ind_edge_rel[tr])
#             ind_col = np.where(ind_array == col)
#     #             print('vine.ind_edge_rel[tr]',vine.ind_edge_rel[tr])
#     #             print('vine.flipflag[tr]',vine.flip_flag[tr])
#     #             print('col',col)
#     #             print('ind_col',ind_col)
#             col = ind_col[0][0]
#     #             print('ind_fin',col)

#             if k == 0:
#                 tr1 = n-k
#                 col1 = n-i
#                 ind1 = vine.r_matrix[tr1,col1] #- 1
#                 ind1 = np.where(vine.nodes == ind1)
#                 ind1 = ind1[0][0]

#                 v2 = v[:,k,ind1][...,np.newaxis]
#                 v1 = v[:,k+1,i][...,np.newaxis]
#                 vv = tf.convert_to_tensor(np.concatenate((v2,v1),1))

#                 cdf_grid = vine.copulas[tr].cdf[:,:,col]
#                 v[:,k,i] = kerncopccdfinv(vv, cdf_grid, u1,u2)

#             else:

#                 parent, inx1, inx2 = parent_var(k,vine.ind_vine,ind_now)

#                 if vine.ind_vine[k-1][ind_now[0]][0] != parent: 
#                     v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
#                 else:
#                     v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

#                 v1 = v[:,k+1,i][...,np.newaxis]
#     #                 vv = np.concatenate((v1,v2),1)

#                 ## CHANGED
#                 vv = np.concatenate((v2,v1),1)

#                 if vine.binning == False:
#                     cdf_grid = vine.copulas[tr].cdf[:,:,col]
#                     vv = tf.convert_to_tensor(vv)
#                     v[:,k,i] = kerncopccdfinv(vv, cdf_grid, u1,u2)

#                 else:
#                     parent11 = vine.r_matrix[n-k+1,n-i] 
#                     ind1 = np.where(vine.nodes == parent11)
#                     ind1 = ind1[0][0]

#                     bins = create_bins(v[:,0,ind1],vine.n_bin)
#                     val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                     vv_all = np.zeros(np.shape(vv)[0],w.dtype)
#                     for bb in range(0,vine.n_bin,1):
#                         cdf_grid = vine.copulas[tr].cdf[:,:,col,bb]
#                         mask = np.where(val_to_bin == bb)
#                         vv_bin = tf.convert_to_tensor(vv[mask[0],:])
#                         v[mask[0],k,i] = kerncopccdfinv(vv_bin, cdf_grid, u1,u2)
#             c += 1


#         if i < d-1: 
#             cc1 = 0
#             for ii in range(1,i+1,1):

#                 cc2 = 0
#                 for j in range(0,ii,1):
#                     tr = j
#                     col = ii-j-1

#                     ind_now = vine.ind_vine[j][ii-1-j]

#                     if j == n-2:
#                         ind_sup = vine.ind_vine[j+1][0]
#                     else:
#                         ind_sup = vine.ind_vine[j+1][i-1-j]

#                     flip_flag = False

#                     parent, inx1, inx2 = parent_var(j+1,vine.ind_vine,ind_sup)                
#                     u_edge = {ind_now[0], ind_now[1]}
#                     if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
#                             if ind_now[0] != parent:
#                                 flip_flag = True

#                     flag = False
#                     if i >1:
#                         for el in flip_flag1[cc1][cc2]:
#                             if el == flip_flag:
#                                 flag = True
#                         if flag == False:
#                             flip_flag1[cc1][cc2].append(flip_flag)
#                     else:
#                         flip_flag1[cc1][cc2].append(flip_flag)


#                     if flag == False:

#                         ind_array = np.array(vine.ind_edge_rel[tr])
#                         ind_col = np.where(ind_array == col)
#             #             print('vine.ind_edge_rel[tr]',vine.ind_edge_rel[tr])
#             #             print('vine.flipflag[tr]',vine.flip_flag[tr])
#             #             print('col',col)
#             #             print('ind_col',ind_col)
#                         ind_fin = ind_col[0][0]
#             #             print('ind_fin',col)
#                         if (vine.ind_edge_rel[tr][ind_fin+1] == col) & (vine.flip_flag[tr][ind_fin+1] == flip_flag):
#                             col = ind_fin + 1
#                         else:
#                             col = ind_fin


#                         if j == 0:
#                             tr1 = n-j
#                             col1 = n-ii
#                             ind1 = vine.r_matrix[tr1,col1] #- 1
#                             ind1 = np.where(vine.nodes == ind1)
#                             ind1 = ind1[0][0]

#                             v2 = v[:,j,ind1][...,np.newaxis]
#                         else:
#                             parent1, inx1, inx2 = parent_var(j,vine.ind_vine,ind_now)

#                             if ind_now[0] != parent1:
#                                 v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
#                             else:
#                                 v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

#                         v1 = v[:,j,ii][...,np.newaxis]

#                         data_u = np.concatenate((v1,v2),1)

#                         data_u = data_u[...,np.newaxis]
#                         trans = Transform(1)

#                         ## Transform data
#                         data_s = trans.forward_u(data_u)
#                         data_x = trans.forward_s(data_s)

#                         if (j==0) | (vine.binning == False):

#                             batch_size = tf.constant(2,tf.int32)
#                             pd_grid_uv = vine.copulas[tr].pd_grid_uv[:,:,col]
#                             cdf1 = vine.copulas[tr].cdf[:,:,col]

#                             ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s[:,:,0],vine.grid_s.min,vine.grid_s.max,cdf1,axis=-2)
#         #                         mar_p1, mar_s1 = kernel_cdf(ccdf_data, vine.grid_u.ex)

#                             pd_points, ccdf_points = evaluate_points(data_s[:,:,0], batch_size, vine.grid_s, cdf1, pd_grid_uv)    

#                             # Update Fp
#                             interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_points, vine.grid_u.ex)
#         #                         interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
#                             if flip_flag == False:
#                                 v[:,j+1,ii] = interp_cdf_poi
#                             else:
#                                 v_flip[:,j+1,ii] = interp_cdf_poi

#                         else: #binning

#                             parent11 = vine.r_matrix[n-j+1,n-ii]  #n-k,n-2-i

#                             ind1 = np.where(vine.nodes == parent11)
#                             ind1 = ind1[0][0]

#                             bins = create_bins(v[:,0,ind1],vine.n_bin)
#                             val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                             vv_all = np.zeros(np.shape(vv)[0],w.dtype)
#                             for bb in range(0,vine.n_bin,1):
#                                 mask = np.where(val_to_bin == bb)

#                                 batch_size = tf.constant(1,tf.int32)
#                                 pd_grid_uv = vine.copulas[tr].pd_grid_uv[:,:,col,bb]
#                                 cdf1 = vine.copulas[tr].cdf[:,:,col,bb]
#                                 data_s_bin = tf.gather_nd(data_s[:,:,0],mask[0][...,tf.newaxis])

#                                 ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin,vine.grid_s.min,vine.grid_s.max,cdf1,axis=-2)
#         #                             mar_p1, mar_s1 = kernel_cdf(ccdf_data, vine.grid_u.ex)

#                                 pd_points, ccdf_points = evaluate_points(data_s_bin, batch_size, vine.grid_s, cdf1, pd_grid_uv)    

#                                 # Update Fp
#                                 interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_points, vine.grid_u.ex)
#         #                             interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
#                                 if flip_flag == False:
#                                     v[mask[0],j+1,ii] = interp_cdf_poi
#                                 else:
#                                     v_flip[mask[0],j+1,ii] = interp_cdf_poi
#                     cc2 += 1
#                 cc1 += 1

#     u = np.reshape(v[:,0,:],np.shape(w))
#     u1 = np.zeros(np.shape(u),u.dtype)

#     u_ax = vine.grid_u.ax1
#     u_ax1 = np.tile(u_ax[...,np.newaxis],[1,np.shape(u)[0]]).T
#     gr_diff = vine.grid_u.diff1

#     c= 0
#     for i in range(d-1,-1,-1):
#         ind = vine.r_matrix[i,i]-1
#         u1[:,ind] = u[:,c]

#         ###### COMMENT WHEN COMPUTING INFORMATION OTHERWISE ALL WRONG
#     #         u_p1 = u[:,c][...,np.newaxis]
#     #         u_upd = np.tile(u_p1,[1,np.shape(u_ax)[0]])

#     #         u_diff = np.abs(u_ax1-u_upd)
#     #         ind1 = np.argmin(u_diff,1)

#     #         diff_val = np.take(gr_diff,ind1)

#     #         u1[:,ind] = u[:,c] + diff_val*np.random.uniform(0.,1.,np.shape(diff_val)[0])

#         c += 1
#     u = u1

#     sample1 = np.zeros([cases,d],w.dtype)
#     for i in tf.range(0,d,1,tf.int32):
#         mar_s1 = vine.Mar_G[i][0]
#         mar_p1 = vine.Mar_G[i][1]
#         sample1_pro = tfp.math.interp_regular_1d_grid(u[:,i],tf.math.reduce_min(mar_p1),tf.math.reduce_max(mar_p1),mar_s1)
#         sample1[:,i] = prep_copula(sample1_pro,0).numpy()
    
#     return sample1


# def vine_copula_sample(vine,cases):
#     d = len(vine.r_matrix)
#     n = len(vine.r_matrix) -1
#     depth = vine.vine_depth   ### Should I use this and how??

#     w = np.random.uniform(0,1,(cases,d))
#     w = w.astype(vine.data_u.dtype)

#     v = np.zeros([cases,d,d],w.dtype)
#     v_flip = np.zeros([cases,d,d],w.dtype)
#     v[:,0,0] = w[:,0]
#     u1 = vine.grid_u.ax1
#     u2 = vine.grid_u.ax2
    
#     for i in range(1,d,1):  #d
#         v[:,i,i] = w[:,i]

#         c = 0
#         for k in range(i-1,-1,-1):
#             tr = k
#             col = i-k-1
#             ind_now = vine.ind_vine[k][c]

#             if k == 0:
#                 tr1 = n-k
#                 col1 = n-i
#                 ind1 = vine.r_matrix[tr1,col1] #- 1
#                 ind1 = np.where(vine.nodes == ind1)
#                 ind1 = ind1[0][0]

#                 v2 = v[:,k,ind1][...,np.newaxis]
#                 v1 = v[:,k+1,i][...,np.newaxis]
#                 vv = tf.convert_to_tensor(np.concatenate((v2,v1),1))
                
#                 cdf_grid = vine.copulas[tr].cdf[:,:,col]
#                 v[:,k,i] = kerncopccdfinv(vv, cdf_grid, u1,u2)

#             else:

#                 parent, inx1, inx2 = parent_var(k,vine.ind_vine,ind_now)

#                 if vine.ind_vine[k-1][ind_now[0]][0] != parent: 
#                     v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
#                 else:
#                     v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

#                 v1 = v[:,k+1,i][...,np.newaxis]
# #                 vv = np.concatenate((v1,v2),1)
#                 vv = np.concatenate((v2,v1),1)

#                 if vine.binning == False:
#                     cdf_grid = vine.copulas[tr].cdf[:,:,col]
#                     vv = tf.convert_to_tensor(vv)
#                     v[:,k,i] = kerncopccdfinv(vv, cdf_grid, u1,u2)

#                 else:
#                     parent11 = vine.r_matrix[n-k+1,n-i] 
#                     ind1 = np.where(vine.nodes == parent11)
#                     ind1 = ind1[0][0]

#                     bins = create_bins(v[:,0,ind1],vine.n_bin)
#                     val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                     vv_all = np.zeros(np.shape(vv)[0],w.dtype)
#                     for bb in range(0,vine.n_bin,1):
#                         cdf_grid = vine.copulas[tr].cdf[:,:,col,bb]
#                         mask = np.where(val_to_bin == bb)
#                         vv_bin = tf.convert_to_tensor(vv[mask[0],:])
#                         v[mask[0],k,i] = kerncopccdfinv(vv_bin, cdf_grid, u1,u2)
#             c += 1
    
#         if i < d-1: 
#             for ii in range(1,i+1,1):

#                 for j in range(0,ii,1):
#                     tr = j
#                     col = ii-j-1

#                     ind_now = vine.ind_vine[j][ii-1-j]

#                     if j == n-2:
#                         ind_sup = vine.ind_vine[j+1][0]
#                     else:
#                         ind_sup = vine.ind_vine[j+1][i-1-j]

#                     if j == 0:
#                         tr1 = n-j
#                         col1 = n-ii
#                         ind1 = vine.r_matrix[tr1,col1] #- 1
#                         ind1 = np.where(vine.nodes == ind1)
#                         ind1 = ind1[0][0]

#                         v2 = v[:,j,ind1][...,np.newaxis]
#                     else:
#                         parent1, inx1, inx2 = parent_var(j,vine.ind_vine,ind_now)

#                         if ind_now[0] != parent1:
#                             v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
#                         else:
#                             v2 = v[:,j,j+ind_now[0]][...,np.newaxis]
                    
#                     v1 = v[:,j,ii][...,np.newaxis]

#                     if j == 0:
#                         data_u = np.concatenate((v2,v1),1)
#                     else:
#                         data_u = np.concatenate((v1,v2),1)
# #                     data_u = np.concatenate((v2,v1),1)

#                     flip_flag = False

#                     parent, inx1, inx2 = parent_var(j+1,vine.ind_vine,ind_sup)                
#                     u_edge = {ind_now[0], ind_now[1]}
#                     if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
#                             if ind_now[0] != parent:
#                                 flip_flag = True

#                     data_u = data_u[...,np.newaxis]
#                     trans = Transform(1)

#                     ## Transform data
#                     data_s = trans.forward_u(data_u)
#                     data_x = trans.forward_s(data_s)

#                     if (j==0) | (vine.binning == False):

#                         batch_size = tf.constant(2,tf.int32)
#                         pd_grid_uv = vine.copulas[tr].pd_grid_uv[:,:,col]
#                         cdf1 = vine.copulas[tr].cdf[:,:,col]

#                         ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s[:,:,0],vine.grid_s.min,vine.grid_s.max,cdf1,axis=-2)
# #                         mar_p1, mar_s1 = kernel_cdf(ccdf_data, vine.grid_u.ex)

#                         pd_points, ccdf_points = evaluate_points(data_s[:,:,0], batch_size, vine.grid_s, cdf1, pd_grid_uv)    

#                         # Update Fp
#                         interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_points, vine.grid_u.ex)
# #                         interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
#                         if flip_flag == False:
#                             v[:,j+1,ii] = interp_cdf_poi
#                         else:
#                             v_flip[:,j+1,ii] = interp_cdf_poi

#                     else: #binning

#                         parent11 = vine.r_matrix[n-j+1,n-ii]  #n-k,n-2-i

#                         ind1 = np.where(vine.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(v[:,0,ind1],vine.n_bin)
#                         val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                         vv_all = np.zeros(np.shape(vv)[0],w.dtype)
#                         for bb in range(0,vine.n_bin,1):
#                             mask = np.where(val_to_bin == bb)

#                             batch_size = tf.constant(1,tf.int32)
#                             pd_grid_uv = vine.copulas[tr].pd_grid_uv[:,:,col,bb]
#                             cdf1 = vine.copulas[tr].cdf[:,:,col,bb]
#                             data_s_bin = tf.gather_nd(data_s[:,:,0],mask[0][...,tf.newaxis])

#                             ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin,vine.grid_s.min,vine.grid_s.max,cdf1,axis=-2)
# #                             mar_p1, mar_s1 = kernel_cdf(ccdf_data, vine.grid_u.ex)

#                             pd_points, ccdf_points = evaluate_points(data_s_bin, batch_size, vine.grid_s, cdf1, pd_grid_uv)    

#                             # Update Fp
#                             interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_points, vine.grid_u.ex)
# #                             interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
#                             if flip_flag == False:
#                                 v[mask[0],j+1,ii] = interp_cdf_poi
#                             else:
#                                 v_flip[mask[0],j+1,ii] = interp_cdf_poi
        
                            
#     u = np.reshape(v[:,0,:],np.shape(w))
#     u1 = np.zeros(np.shape(u),u.dtype)
    
#     u_ax = vine.grid_u.ax1
#     u_ax1 = np.tile(u_ax[...,np.newaxis],[1,np.shape(u)[0]]).T
#     gr_diff = vine.grid_u.diff1
    
#     c= 0
#     for i in range(d-1,-1,-1):
#         ind = vine.r_matrix[i,i]-1
#         u1[:,ind] = u[:,c]
        
#         ###### COMMENT WHEN COMPUTING INFORMATION OTHERWISE ALL WRONG
# #         u_p1 = u[:,c][...,np.newaxis]
# #         u_upd = np.tile(u_p1,[1,np.shape(u_ax)[0]])

# #         u_diff = np.abs(u_ax1-u_upd)
# #         ind1 = np.argmin(u_diff,1)

# #         diff_val = np.take(gr_diff,ind1)

# #         u1[:,ind] = u[:,c] + diff_val*np.random.uniform(0.,1.,np.shape(diff_val)[0])
        
#         c += 1
#     u = u1
    
#     sample1 = np.zeros([cases,d],w.dtype)
#     for i in tf.range(0,d,1,tf.int32):
#         mar_s1 = vine.Mar_G[i][0]
#         mar_p1 = vine.Mar_G[i][1]
#         sample1_pro = tfp.math.interp_regular_1d_grid(u[:,i],tf.math.reduce_min(mar_p1),tf.math.reduce_max(mar_p1),mar_s1)
#         sample1[:,i] = prep_copula(sample1_pro,0).numpy()
    
#     return sample1

# def vine_copula_sample(vine,cases):
#     d = len(vine.r_matrix)
#     n = len(vine.r_matrix) -1
#     depth = vine.vine_depth   ### Should I use this and how??

#     w = np.random.uniform(0,1,(cases,d))
#     w = w.astype(vine.data_u.dtype)

#     v = np.zeros([cases,d,d],w.dtype)
#     v_flip = np.zeros([cases,d,d],w.dtype)
#     v[:,0,0] = w[:,0]
#     u1 = vine.grid_u.ax1
#     u2 = vine.grid_u.ax2
    
#     for i in range(1,d,1):  #d
#         v[:,i,i] = w[:,i]

#         c = 0
#         for k in range(i-1,-1,-1):
#             tr = k
#             col = i-k-1
#             ind_now = vine.ind_vine[k][c]

#             if k == 0:
#                 tr1 = n-k
#                 col1 = n-i
#                 ind1 = vine.r_matrix[tr1,col1] #- 1
#                 ind1 = np.where(vine.nodes == ind1)
#                 ind1 = ind1[0][0]

#                 v2 = v[:,k,ind1][...,np.newaxis]
#                 v1 = v[:,k+1,i][...,np.newaxis]
#                 vv = tf.convert_to_tensor(np.concatenate((v2,v1),1))

#                 cdf_grid = vine.copulas[tr].cdf[:,:,col]
#                 v[:,k,i] = kerncopccdfinv(vv, cdf_grid, u1,u2)

#             else:

#                 parent, inx1, inx2 = parent_var(k,vine.ind_vine,ind_now)

#                 if vine.ind_vine[k-1][ind_now[0]][0] != parent: 
#                     v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
#                 else:
#                     v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

#                 v1 = v[:,k+1,i][...,np.newaxis]
#                 vv = np.concatenate((v1,v2),1)

#                 if vine.binning == False:
#                     cdf_grid = vine.copulas[tr].cdf[:,:,col]
#                     vv = tf.convert_to_tensor(vv)
#                     v[:,k,i] = kerncopccdfinv(vv, cdf_grid, u1,u2)

#                 else:
#                     parent11 = vine.r_matrix[n-k+1,n-i] 
#                     ind1 = np.where(vine.nodes == parent11)
#                     ind1 = ind1[0][0]

#                     bins = create_bins(v[:,0,ind1],vine.n_bin)
#                     val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                     vv_all = np.zeros(np.shape(vv)[0],w.dtype)
#                     for bb in range(0,vine.n_bin,1):
#                         cdf_grid = vine.copulas[tr].cdf[:,:,col,bb]
#                         mask = np.where(val_to_bin == bb)
#                         vv_bin = tf.convert_to_tensor(vv[mask[0],:])
#                         v[mask[0],k,i] = kerncopccdfinv(vv_bin, cdf_grid, u1,u2)
#             c += 1
        
    
#         if i < d-1: 
#             for ii in range(1,i+1,1):

#                 for j in range(0,ii,1):
#                     tr = j
#                     col = ii-j-1

#                     ind_now = vine.ind_vine[j][ii-1-j]

#                     if j == n-2:
#                         ind_sup = vine.ind_vine[j+1][0]
#                     else:
#                         ind_sup = vine.ind_vine[j+1][i-1-j]

#                     if j == 0:
#                         tr1 = n-j
#                         col1 = n-ii
#                         ind1 = vine.r_matrix[tr1,col1] #- 1
#                         ind1 = np.where(vine.nodes == ind1)
#                         ind1 = ind1[0][0]

#                         v2 = v[:,j,ind1][...,np.newaxis]
#                     else:
#                         parent1, inx1, inx2 = parent_var(j,vine.ind_vine,ind_now)

#                         if ind_now[0] != parent1:
#                             v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
#                         else:
#                             v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

#                     v1 = v[:,j,ii][...,np.newaxis]

#                     data_u = np.concatenate((v1,v2),1)

#                     flip_flag = False

#                     parent, inx1, inx2 = parent_var(j+1,vine.ind_vine,ind_sup)                
#                     u_edge = {ind_now[0], ind_now[1]}
#                     if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
#                             if ind_now[0] != parent:
#                                 flip_flag = True

#                     data_u = data_u[...,np.newaxis]
#                     trans = Transform(1)

#                     ## Transform data
#                     data_s = trans.forward_u(data_u)
#                     data_x = trans.forward_s(data_s)

#                     if (j==0) | (vine.binning == False):

#                         batch_size = tf.constant(2,tf.int32)
#                         pd_grid_uv = vine.copulas[tr].pd_grid_uv[:,:,col]
#                         cdf1 = vine.copulas[tr].cdf[:,:,col]

#                         ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s[:,:,0],vine.grid_s.min,vine.grid_s.max,cdf1,axis=-2)
# #                         mar_p1, mar_s1 = kernel_cdf(ccdf_data, vine.grid_u.ex)

#                         pd_points, ccdf_points = evaluate_points(data_s[:,:,0], batch_size, vine.grid_s, cdf1, pd_grid_uv)    

#                         # Update Fp
#                         interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_points, vine.grid_u.ex)
# #                         interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
#                         if flip_flag == False:
#                             v[:,j+1,ii] = interp_cdf_poi
#                         else:
#                             v_flip[:,j+1,ii] = interp_cdf_poi

#                     else: #binning

#                         parent11 = vine.r_matrix[n-j+1,n-ii]  #n-k,n-2-i

#                         ind1 = np.where(vine.nodes == parent11)
#                         ind1 = ind1[0][0]

#                         bins = create_bins(v[:,0,ind1],vine.n_bin)
#                         val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                         vv_all = np.zeros(np.shape(vv)[0],w.dtype)
#                         for bb in range(0,vine.n_bin,1):
#                             mask = np.where(val_to_bin == bb)

#                             batch_size = tf.constant(1,tf.int32)
#                             pd_grid_uv = vine.copulas[tr].pd_grid_uv[:,:,col,bb]
#                             cdf1 = vine.copulas[tr].cdf[:,:,col,bb]
#                             data_s_bin = tf.gather_nd(data_s[:,:,0],mask[0][...,tf.newaxis])

#                             ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin,vine.grid_s.min,vine.grid_s.max,cdf1,axis=-2)
# #                             mar_p1, mar_s1 = kernel_cdf(ccdf_data, vine.grid_u.ex)

#                             pd_points, ccdf_points = evaluate_points(data_s_bin, batch_size, vine.grid_s, cdf1, pd_grid_uv)    

#                             # Update Fp
#                             interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_points, vine.grid_u.ex)
# #                             interp_cdf_poi = interp1d_np(ccdf_points, mar_s1, mar_p1)
#                             if flip_flag == False:
#                                 v[mask[0],j+1,ii] = interp_cdf_poi
#                             else:
#                                 v_flip[mask[0],j+1,ii] = interp_cdf_poi
        
                            
#     u = np.reshape(v[:,0,:],np.shape(w))
#     u1 = np.zeros(np.shape(u),u.dtype)
    
#     u_ax = vine.grid_u.ax1
#     u_ax1 = np.tile(u_ax[...,np.newaxis],[1,np.shape(u)[0]]).T
#     gr_diff = vine.grid_u.diff1
    
#     c= 0
#     for i in range(d-1,-1,-1):
#         ind = vine.r_matrix[i,i]-1
#         u1[:,ind] = u[:,c]
        
#         ###### COMMENT WHEN COMPUTING INFORMATION OTHERWISE ALL WRONG
# #         u_p1 = u[:,c][...,np.newaxis]
# #         u_upd = np.tile(u_p1,[1,np.shape(u_ax)[0]])

# #         u_diff = np.abs(u_ax1-u_upd)
# #         ind1 = np.argmin(u_diff,1)

# #         diff_val = np.take(gr_diff,ind1)

# #         u1[:,ind] = u[:,c] + diff_val*np.random.uniform(0.,1.,np.shape(diff_val)[0])
        
#         c += 1
#     u = u1
    
#     sample1 = np.zeros([cases,d],w.dtype)
#     for i in tf.range(0,d,1,tf.int32):
#         mar_s1 = vine.Mar_G[i][0]
#         mar_p1 = vine.Mar_G[i][1]
#         sample1_pro = tfp.math.interp_regular_1d_grid(u[:,i],tf.math.reduce_min(mar_p1),tf.math.reduce_max(mar_p1),mar_s1)
#         sample1[:,i] = prep_copula(sample1_pro,0).numpy()
    
#     return sample1



############################ INVERSE NON-PARAMETRIC COPULA CDF ####################################

@tf.function(experimental_relax_shapes=True)
def kerncopccdfinv(w, cdf_grid, u1, u2):  
    # Function to sample from the copula
    #u1 = grid_u2.ax1
    #u2 = grid_u2.ax2
    len_w = tf.shape(w)[0]
    len_ax = tf.shape(u1)[0]
    u1_tile = tf.tile(u1,[len_w])
    u1_tile = tf.transpose(tf.reshape(u1_tile,[len_w,len_ax])) #[100,20000]

    w0_tile = tf.tile(w[:,0],[len_ax])
    w0_tile = tf.reshape(w0_tile,[len_ax,len_w]) #[100,20000]
    
    m1 = tf.math.argmin(tf.abs(u1_tile-w0_tile),0)
    
    m1 = m1[...,tf.newaxis]
    g =  tf.gather_nd(cdf_grid, m1)
    g = tf.transpose(g)
    
    len_g = tf.shape(g)[0]

    w1_tile = tf.tile(w[:,1],[len_ax])
    w1_tile = tf.reshape(w1_tile,[len_ax,len_w]) #[100,20000]

    propro = g-w1_tile #tf.abs  tf.math.negative

    mask1 = tf.greater(propro,0)
    mask1 = tf.cast(mask1,dtype=tf.int32)
    ind = tf.math.argmax(mask1)
    U2 = tf.gather(u2,ind)
    return U2



###################### SAMPLING FROM PARAMETRIC VINE COPULAS ################################

def vine_cop_par_sample(vine, cases):
    d = len(vine.r_matrix)
    n = len(vine.r_matrix) -1

    w = np.random.uniform(0,1,(cases,d))
    w = w.astype(np.float32)

    v = np.zeros([cases,d,d],w.dtype)
    v_flip = np.zeros([cases,d,d],w.dtype)
    v[:,0,0] = w[:,0]
    
    for i in range(1,d,1):
        v[:,i,i] = w[:,i]

        c = 0
        for k in range(i-1,-1,-1):
            tr = k
            col = i-k-1
            ind_now = vine.ind_vine[k][c]

            if k == 0:
                tr1 = n-k
                col1 = n-i
                ind1 = vine.r_matrix[tr1,col1] #- 1
                ind1 = np.where(vine.nodes == ind1)
                ind1 = ind1[0][0]

                v2 = v[:,k,ind1][...,np.newaxis]
                v1 = v[:,k+1,i][...,np.newaxis]
                vv = np.concatenate((v1,v2),1)
                v[:,k,i] = copulainvccdf(vine.copulas[tr][col],vv)
            else:

                parent, inx1, inx2 = parent_var(k,vine.ind_vine,ind_now)

                if vine.ind_vine[k-1][ind_now[0]][0] != parent: 
                    v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
                else:
                    v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

                v1 = v[:,k+1,i][...,np.newaxis]
                vv = np.concatenate((v1,v2),1)
                if vine.binning == False:
                    v[:,k,i] = copulainvccdf(vine.copulas[tr][col],vv)
                else:
                    
                    ind1 = parent
                    
                    if k == 1:
                        ind1 = np.where(vine.nodes == ind1 +1)
                        ind1 = ind1[0][0]
                        bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                        val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                    else:
                        ind_par_now = vine.ind_vine[k-1][ind_now[1]]
                        parent22, inx1, inx2 = parent_var(k-1,vine.ind_vine,ind_par_now)  

                        ind1 = ind1 + k - 1
                        if (vine.ind_vine[k-2][ind_par_now[0]][0] == parent22):
                            bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                        else:
                            bins = create_bins(v_flip[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1
                    
#                     if (vine.ind_vine[k-1][ind_now[0]][0] == parent) | (k == 1): 
#                         if k == 1:
#                             ind1 = np.where(vine.nodes == ind1 +1)
#                             ind1 = ind1[0][0]
#                         else:
#                             ind1 = ind1 + k - 1
#                         bins = create_bins(v[:,k-1,ind1],vine.n_bin)
#                         val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
#                     else:
#                         ind1 = ind1 + k - 1
#                         bins = create_bins(v_flip[:,k-1,ind1],vine.n_bin)
#                         val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1
                        
                    vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                    for bb in range(0,vine.n_bin,1):
                        mask = np.where(val_to_bin == bb)
                        vv_bin = vv[mask[0],:]
                        
                        ### CDF FORCE UNIFORM
                        vv_bin_new = vv_bin
                        for zz in range(0,2,1):
                            vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],vine.grid_u.ex)
                        vv_bin = vv_bin_new
                        ###
                        
                        v[mask[0],k,i] = copulainvccdf(vine.copulas[tr][col][bb],vv_bin)
                        
#                         corr = stats.pearsonr(vv_bin[:,1],v[mask[0],k,i])
#                         print('Corr value  UV space, bin(',bb,')',corr[0])
            c += 1
            
        if i < d -1:
            for ii in range(1,i+1,1):
                for j in range(0,ii,1):
                    tr = j
                    col = ii-j-1

                    ind_now = vine.ind_vine[j][ii-1-j]
                    
                    if j == n-2:
                        ind_sup = vine.ind_vine[j+1][0]
                    else:
                        ind_sup = vine.ind_vine[j+1][i-1-j]
                    
                    if j == 0:
                        tr1 = n-j
                        col1 = n-ii
                        ind1 = vine.r_matrix[tr1,col1] #- 1
                        ind1 = np.where(vine.nodes == ind1)
                        ind1 = ind1[0][0]

                        v2 = v[:,j,ind1][...,np.newaxis]
                    else:
                        parent1, inx1, inx2 = parent_var(j,vine.ind_vine,ind_now)
                        
#                         if ind_now[0] != parent1:
                        if vine.ind_vine[j-1][ind_now[0]][0] != parent1:
                            v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
                        else:
                            v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

                    v1 = v[:,j,ii][...,np.newaxis]

                    vv = np.concatenate((v1,v2),1)

                    parent, inx1, inx2 = parent_var(j+1,vine.ind_vine,ind_sup)                
                    u_edge = {ind_now[0], ind_now[1]}

                    if (j == 0) | (vine.binning == False):
                        if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                vv = np.concatenate((v2,v1),1)
                                v_flip[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
                            else:
                                v[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
                        else:
                            v[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
                    else:

                        if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                vv = np.concatenate((v2,v1),1)
                                flip_1 = True
                            else:
                                flip_1 = False
                        else:
                            flip_1 = False

                        ind1 = parent1

                        if j == 1:
                            ind1 = np.where(vine.nodes == ind1 +1)
                            ind1 = ind1[0][0]
                            bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                        else:
                            ind_par_now = vine.ind_vine[j-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(j-1,vine.ind_vine,ind_par_now)  

                            ind1 = ind1 + j - 1
                            if (vine.ind_vine[j-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                                val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                            else:
                                bins = create_bins(v_flip[:,j-1,ind1],vine.n_bin)
                                val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1

#                         if (vine.ind_vine[j-1][ind_now[0]][0] == parent1) | (j == 1):
#                             if j == 1:
#                                 ind1 = np.where(vine.nodes == ind1 +1)
#                                 ind1 = ind1[0][0]
#                             else:
#                                 ind1 = ind1 + j - 1
# #                             ind1 = np.where(nodes == ind1 + 1)
# #                             ind1 = ind1[0][0]
#                             bins = create_bins(v[:,j-1,ind1],vine.n_bin)
#                             val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
#                         else:
#                             ind1 = ind1 + j -1
#                             bins = create_bins(v_flip[:,j-1,ind1],vine.n_bin)
#                             val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1

                        vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                        for bb in range(0,vine.n_bin,1):
                            mask = np.where(val_to_bin == bb)
                            vv_bin = vv[mask[0],:]

                            ### CDF FORCE UNIFORM
                            vv_bin_new = vv_bin
                            for zz in range(0,2,1):
                                vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],vine.grid_u.ex)
                            vv_bin = vv_bin_new
                            ###

                            tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])

                            if flip_1 == True:
                                v_flip[mask[0],j+1,ii] = copulaccdf(vine.copulas[tr][col][bb],vv_bin)
                            else:
                                v[mask[0],j+1,ii] = copulaccdf(vine.copulas[tr][col][bb],vv_bin)
                    
    u = np.reshape(v[:,0,:],np.shape(w))
    u1 = np.zeros(np.shape(u),u.dtype)
    
    u_ax = vine.grid_u.ax1
    u_ax1 = np.tile(u_ax[...,np.newaxis],[1,np.shape(u)[0]]).T
    gr_diff = vine.grid_u.diff1
    
    c= 0
    for i in range(d-1,-1,-1):
        ind = vine.r_matrix[i,i]-1
        u1[:,ind] = u[:,c]

#         u_p1 = u[:,c][...,np.newaxis]
#         u_upd = np.tile(u_p1,[1,np.shape(u_ax)[0]])

#         u_diff = np.abs(u_ax1-u_upd)
#         ind1 = np.argmin(u_diff,1)

#         diff_val = np.take(gr_diff,ind1)

#         u1[:,ind] = u[:,c] + diff_val*np.random.uniform(0.,1.,np.shape(diff_val)[0])
        
        c += 1
    u = u1

    sample1 = np.zeros([cases,d],w.dtype)
    for i in tf.range(0,d,1,tf.int32):
        mar_s1 = vine.Mar_G[i][0]
        mar_p1 = vine.Mar_G[i][1]
        sample1_pro = tfp.math.interp_regular_1d_grid(u[:,i],tf.math.reduce_min(mar_p1),tf.math.reduce_max(mar_p1),mar_s1)
        sample1[:,i] = prep_copula(sample1_pro,0).numpy()
        
    return sample1

# def vine_cop_par_sample(vine, cases):
#     d = len(vine.r_matrix)
#     n = len(vine.r_matrix) -1

#     w = np.random.uniform(0,1,(cases,d))
#     w = w.astype(np.float32)

#     v = np.zeros([cases,d,d],w.dtype)
#     v_flip = np.zeros([cases,d,d],w.dtype)
#     v[:,0,0] = w[:,0]
    
#     for i in range(1,d,1):
#         v[:,i,i] = w[:,i]

#         c = 0
#         for k in range(i-1,-1,-1):
#             tr = k
#             col = i-k-1
#             ind_now = vine.ind_vine[k][c]

#             if k == 0:
#                 tr1 = n-k
#                 col1 = n-i
#                 ind1 = vine.r_matrix[tr1,col1] #- 1
#                 ind1 = np.where(vine.nodes == ind1)
#                 ind1 = ind1[0][0]

#                 v2 = v[:,k,ind1][...,np.newaxis]
#                 v1 = v[:,k+1,i][...,np.newaxis]
#                 vv = np.concatenate((v1,v2),1)
#                 v[:,k,i] = copulainvccdf(vine.copulas[tr][col],vv)
#             else:

#                 parent, inx1, inx2 = parent_var(k,vine.ind_vine,ind_now)

#                 if vine.ind_vine[k-1][ind_now[0]][0] != parent: 
#                     v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
#                 else:
#                     v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

#                 v1 = v[:,k+1,i][...,np.newaxis]
#                 vv = np.concatenate((v1,v2),1)
#                 if vine.binning == False:
#                     v[:,k,i] = copulainvccdf(vine.copulas[tr][col],vv)
#                 else:
#                     parent11 = vine.r_matrix[n-k+1,n-i]  #n-k,n-2-i
# #                     print('indr1',n-k+1)
# #                     print('indr2',n-i)
# #                     print('ind_now',ind_now)
# #                     print('parent',parent11)
# #                     print('nodes',nodes)
#                     ind1 = np.where(vine.nodes == parent11)
#                     ind1 = ind1[0][0]
                    
#                     bins = create_bins(v[:,0,ind1],vine.n_bin)
#                     val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                     vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                     for bb in range(0,vine.n_bin,1):
#                         mask = np.where(val_to_bin == bb)
#                         vv_bin = vv[mask[0],:]
#                         v[mask[0],k,i] = copulainvccdf(vine.copulas[tr][col][bb],vv_bin)
#             c += 1

#         if i < d-1: 
#             for ii in range(1,i+1,1):

#                 for j in range(0,ii,1):
#                     tr = j
#                     col = ii-j-1

#                     ind_now = vine.ind_vine[j][ii-1-j]

#                     if j == n-2:
#                         ind_sup = vine.ind_vine[j+1][0]
#                     else:
#                         ind_sup = vine.ind_vine[j+1][i-1-j]

#                     if j == 0:
#                         tr1 = n-j
#                         col1 = n-ii
#                         ind1 = vine.r_matrix[tr1,col1] #- 1
#                         ind1 = np.where(vine.nodes == ind1)
#                         ind1 = ind1[0][0]

#                         v2 = v[:,j,ind1][...,np.newaxis]
#                     else:
#                         parent1, inx1, inx2 = parent_var(j,vine.ind_vine,ind_now)

#                         if ind_now[0] != parent1:
#                             v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
#                         else:
#                             v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

#                     v1 = v[:,j,ii][...,np.newaxis]

#                     vv = np.concatenate((v1,v2),1)

#                     parent, inx1, inx2 = parent_var(j+1,vine.ind_vine,ind_sup)                
#                     u_edge = {ind_now[0], ind_now[1]}
                    
#                     if (j == 0) | (vine.binning == False):
#                         if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
#                             if ind_now[0] != parent:
#                                 vv = np.concatenate((v2,v1),1)
#                                 v_flip[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
#                             else:
#                                 v[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
#                         else:
#                             v[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
#                     else:
#                         parent11 = vine.r_matrix[n-j+1,n-ii]  #n-k,n-2-i
# #                         print('indr1',n-j+1)
# #                         print('indr2',n-ii)
                        
# # #                         [n-j,n-2-ii]
# #                         print('ind_now',ind_now)
# #                         print('parent',parent)
#                         if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
#                             if ind_now[0] != parent:
#                                 vv = np.concatenate((v2,v1),1)
                                
#                                 ind1 = np.where(vine.nodes == parent11)
#                                 ind1 = ind1[0][0]

#                                 bins = create_bins(v[:,0,ind1],vine.n_bin)
#                                 val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                                 vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                                 for bb in range(0,vine.n_bin,1):
#                                     mask = np.where(val_to_bin == bb)
#                                     vv_bin = vv[mask[0],:]
#                                     v_flip[mask[0],j+1,ii] = copulaccdf(vine.copulas[tr][col][bb],vv_bin)
#                             else:
#                                 ind1 = np.where(vine.nodes == parent11)
#                                 ind1 = ind1[0][0]

#                                 bins = create_bins(v[:,0,ind1],n_bin)
#                                 val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                                 vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                                 for bb in range(0,vine.n_bin,1):
#                                     mask = np.where(val_to_bin == bb)
#                                     vv_bin = vv[mask[0],:]
#                                     v_flip[mask[0],j+1,ii] = copulaccdf(vine.copulas[tr][col][bb],vv_bin)
#                         else:
#                             ind1 = np.where(vine.nodes == parent11)
#                             ind1 = ind1[0][0]

#                             bins = create_bins(v[:,0,ind1],n_bin)
#                             val_to_bin = np.digitize(v[:,0,ind1], bins) -1
#                             vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
#                             for bb in range(0,vine.n_bin,1):
#                                 mask = np.where(val_to_bin == bb)
#                                 vv_bin = vv[mask[0],:]
#                                 v_flip[mask[0],j+1,ii] = copulaccdf(vine.copulas[tr][col][bb],vv_bin)
                    
#     u = np.reshape(v[:,0,:],np.shape(w))
#     u1 = np.zeros(np.shape(u),u.dtype)
    
#     u_ax = vine.grid_u.ax1
#     u_ax1 = np.tile(u_ax[...,np.newaxis],[1,np.shape(u)[0]]).T
#     gr_diff = vine.grid_u.diff1
    
#     c= 0
#     for i in range(d-1,-1,-1):
#         ind = vine.r_matrix[i,i]-1
#         u1[:,ind] = u[:,c]

# #         u_p1 = u[:,c][...,np.newaxis]
# #         u_upd = np.tile(u_p1,[1,np.shape(u_ax)[0]])

# #         u_diff = np.abs(u_ax1-u_upd)
# #         ind1 = np.argmin(u_diff,1)

# #         diff_val = np.take(gr_diff,ind1)

# #         u1[:,ind] = u[:,c] + diff_val*np.random.uniform(0.,1.,np.shape(diff_val)[0])
        
#         c += 1
#     u = u1

#     sample1 = np.zeros([cases,d],w.dtype)
#     for i in tf.range(0,d,1,tf.int32):
#         mar_s1 = vine.Mar_G[i][0]
#         mar_p1 = vine.Mar_G[i][1]
#         sample1_pro = tfp.math.interp_regular_1d_grid(u[:,i],tf.math.reduce_min(mar_p1),tf.math.reduce_max(mar_p1),mar_s1)
#         sample1[:,i] = prep_copula(sample1_pro,0).numpy()
        
#     return sample1

# File: src/DVC_tensorflow/sampling/vine_sample.py
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np

from utils.prob_op import kernel_cdf
from evalu.vine_eval import evaluate_points
from utils.interpolation import interp1d_np
from utils.dataset_op import create_bins, check_bins
from vine_tree.tree_op import parent_var
from pre_proc.preparation import prep_copula
from pre_proc.transformation import Transform
from param.cond_copula import *

#################################### SAMPLING FROM NON-PARAMETRIC COPULA #######################################

############################ INVERSE NON-PARAMETRIC COPULA CDF ####################################

@tf.function(experimental_relax_shapes=True)
def kerncopccdfinv(w, cdf_grid, u1, u2):  
    # Function to sample from the copula
    #u1 = grid_u2.ax1
    #u2 = grid_u2.ax2
    len_w = tf.shape(w)[0]
    len_ax = tf.shape(u1)[0]
    u1_tile = tf.tile(u1,[len_w])
    u1_tile = tf.transpose(tf.reshape(u1_tile,[len_w,len_ax])) #[100,20000]

    w0_tile = tf.tile(w[:,0],[len_ax])
    w0_tile = tf.reshape(w0_tile,[len_ax,len_w]) #[100,20000]
    
    m1 = tf.math.argmin(tf.abs(u1_tile-w0_tile),0)
    
    m1 = m1[...,tf.newaxis]
    g =  tf.gather_nd(cdf_grid, m1)
    g = tf.transpose(g)
    
    len_g = tf.shape(g)[0]

    w1_tile = tf.tile(w[:,1],[len_ax])
    w1_tile = tf.reshape(w1_tile,[len_ax,len_w]) #[100,20000]

    propro = g-w1_tile #tf.abs  tf.math.negative

    mask1 = tf.greater(propro,0)
    mask1 = tf.cast(mask1,dtype=tf.int32)
    ind = tf.math.argmax(mask1)
    U2 = tf.gather(u2,ind)
    return U2

################# TRY CON FLAG

def vine_copula_sample(vine,cases):
    d = len(vine.r_matrix)
    #print(d)
    n = len(vine.r_matrix) -1
    depth = vine.vine_depth   ### Should I use this and how??

    # w = np.random.uniform(1e-3,0.999,(cases,d))   #0,1,(cases,d))

    w = np.random.uniform(0,1,(cases,d))
    mag = np.max(vine.grid_u.ex)#-1e-7
    mig = np.min(vine.grid_u.ex)#+1e-7

    w = (mag-mig)*(w-np.min(w))/(np.max(w)-np.min(w))+mig
    w = w.astype(vine.data_u.dtype)

    v = np.zeros([cases,d,d],w.dtype)
    v_flip = np.zeros([cases,d,d],w.dtype)
    v[:,0,0] = w[:,0]
    u1 = vine.grid_u.ax1
    u2 = vine.grid_u.ax2

    ## Initialize the flag for flipping
    flip_flag1 = []
    for ii in range(1,d-1,1):
        flip_flag2 = []

        for j in range(0,ii,1):
            flip_flag2.append([])
        flip_flag1.append(flip_flag2)

    ## Start from first column [1,1]
    for i in range(1,d,1):  #d
        v[:,i,i] = w[:,i]
        
        c = 0
        for k in range(i-1,-1,-1):
            tr = k
            col = i-k-1
            ind_now = vine.ind_vine[k][c]

            ind_array = np.array(vine.ind_edge_rel[tr])
            ind_col = np.where(ind_array == col)
            col = ind_col[0][0]

            if k == 0:
                tr1 = n-k
                col1 = n-i
                ind1 = vine.r_matrix[tr1,col1] #- 1
                ind1 = np.where(vine.nodes == ind1)
                ind1 = ind1[0][0]

                v2 = v[:,k,ind1][...,np.newaxis]
                v1 = v[:,k+1,i][...,np.newaxis]
                vv = tf.convert_to_tensor(np.concatenate((v2,v1),1))

                cdf_grid = vine.copulas[tr].cdf[:,:,col]
                v[:,k,i] = kerncopccdfinv(vv, cdf_grid, u1,u2)

            else:

                parent, inx1, inx2 = parent_var(k,vine.ind_vine,ind_now)

                if vine.ind_vine[k-1][ind_now[0]][0] != parent: 
                    v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
                else:
                    v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

                v1 = v[:,k+1,i][...,np.newaxis]

                ## CHANGED
                vv = np.concatenate((v2,v1),1)

            if tr > depth:
                ### Independent
                col = i-k-1

                vv = np.flip(vv,axis=1)

                if vine.binning == False:
                    v[:,k,i] = copulainvccdf(vine.copulas[tr][col],vv)

                else:
                    
                    ind1 = parent
                    
                    if k == 1:
                        ind1 = np.where(vine.nodes == ind1 +1)
                        ind1 = ind1[0][0]
                        bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                        val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                    else:
                        ind_par_now = vine.ind_vine[k-1][ind_now[1]]
                        parent22, inx1, inx2 = parent_var(k-1,vine.ind_vine,ind_par_now)  

                        ind1 = ind1 + k - 1
                        if (vine.ind_vine[k-2][ind_par_now[0]][0] == parent22):
                            bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                        else:
                            bins = create_bins(v_flip[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1
                        
                    vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                    for bb in range(0,vine.n_bin,1):
                        mask = np.where(val_to_bin == bb)
                        vv_bin = vv[mask[0],:]
                        
                        ### CDF FORCE UNIFORM
                        vv_bin_new = vv_bin
                        for zz in range(0,2,1):
                            vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],vine.grid_u.ex)
                        vv_bin = vv_bin_new
                        ###
                        
                        v[mask[0],k,i] = copulainvccdf(vine.copulas[tr][col][bb],vv_bin)
            else:

                if vine.binning == False:
                    cdf_grid = vine.copulas[tr].cdf[:,:,col]
                    vv = tf.convert_to_tensor(vv)
                    v[:,k,i] = kerncopccdfinv(vv, cdf_grid, u1,u2)
                    #if k > 9:
                        #print(v[:20,k,i])
                else:
                    
                    ind1 = parent
                    
                    if k == 1:
                        ind1 = np.where(vine.nodes == ind1 +1)
                        ind1 = ind1[0][0]
                        bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                        val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                    else:
                        ind_par_now = vine.ind_vine[k-1][ind_now[1]]
                        parent22, inx1, inx2 = parent_var(k-1,vine.ind_vine,ind_par_now)  

                        ind1 = ind1 + k - 1
                        if (vine.ind_vine[k-2][ind_par_now[0]][0] == parent22):
                            bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                            val_to_bin = check_bins(v[:,k-1,ind1],bins)
                        else:
                            bins = create_bins(v_flip[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1
                            val_to_bin = check_bins(v_flip[:,k-1,ind1],bins)

                    vv_all = np.zeros(np.shape(vv)[0],w.dtype)
                    for bb in range(0,vine.n_bin,1):
                        cdf_grid = vine.copulas[tr].cdf[:,:,col,bb]
                        mask = np.where(val_to_bin == bb)
                        vv_bin = vv[mask[0],:]
                        
                        ### CDF FORCE UNIFORM
                        vv_bin_new = vv_bin
                        for zz in range(0,2,1):
                            vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],vine.grid_u.ex)
                        vv_bin = vv_bin_new
                        ###
                        vv_bin = tf.convert_to_tensor(vv[mask[0],:])
                        
                        v[mask[0],k,i] = kerncopccdfinv(vv_bin, cdf_grid, u1,u2)
#                         corr = stats.pearsonr(vv_bin[:,1],v[mask[0],k,i])
#                         print('Corr value  UV space: ',corr[0])
            c += 1

        if i < d-1: 
            cc1 = 0
            for ii in range(1,i+1,1):

                cc2 = 0
                for j in range(0,ii,1):
                    tr = j
                    col = ii-j-1

                    ind_now = vine.ind_vine[j][ii-1-j]

                    if j == n-2:
                        ind_sup = vine.ind_vine[j+1][0]
                    else:
                        ind_sup = vine.ind_vine[j+1][i-1-j]

                    flip_flag = False

                    parent, inx1, inx2 = parent_var(j+1,vine.ind_vine,ind_sup)                
                    u_edge = {ind_now[0], ind_now[1]}
                    if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                flip_flag = True

                    flag = False
                    if i >1:
                        for el in flip_flag1[cc1][cc2]:
                            if el == flip_flag:
                                flag = True
                        if flag == False:
                            flip_flag1[cc1][cc2].append(flip_flag)
                    else:
                        flip_flag1[cc1][cc2].append(flip_flag)

                    if flag == False:

                        if j == 0:
                            tr1 = n-j
                            col1 = n-ii
                            ind1 = vine.r_matrix[tr1,col1] #- 1
                            ind1 = np.where(vine.nodes == ind1)
                            ind1 = ind1[0][0]

                            v2 = v[:,j,ind1][...,np.newaxis]
                        else:
                            parent1, inx1, inx2 = parent_var(j,vine.ind_vine,ind_now)

#                             if ind_now[0] != parent1:
                            if vine.ind_vine[j-1][ind_now[0]][0] != parent1:
                                v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
                            else:
                                v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

                        v1 = v[:,j,ii][...,np.newaxis]
                        
                        if flip_flag == False:
                            data_u = np.concatenate((v2,v1),1)
                        else:
                            data_u = np.concatenate((v1,v2),1)

                        if j > depth:
                            ## Independent
                            col = ii-j-1
                            # vv = np.concatenate((v1,v2),1)
                            vv = np.flip(data_u,axis=1)

                            parent, inx1, inx2 = parent_var(j+1,vine.ind_vine,ind_sup)                
                            u_edge = {ind_now[0], ind_now[1]}
                            

                            if (j == 0) | (vine.binning == False):
                                
                                if flip_flag == False:
                                    v[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
                                else:
                                    v_flip[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)

                            else:

                                if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                                    if ind_now[0] != parent:
                                        vv = np.concatenate((v2,v1),1)
                                        flip_1 = True
                                    else:
                                        flip_1 = False
                                else:
                                    flip_1 = False

                                ind1 = parent1

                                if j == 1:
                                    ind1 = np.where(vine.nodes == ind1 +1)
                                    ind1 = ind1[0][0]
                                    bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                                    val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                                else:
                                    ind_par_now = vine.ind_vine[j-1][ind_now[1]]
                                    parent22, inx1, inx2 = parent_var(j-1,vine.ind_vine,ind_par_now)  

                                    ind1 = ind1 + j - 1
                                    if (vine.ind_vine[j-2][ind_par_now[0]][0] == parent22):
                                        bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                                        val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                                    else:
                                        bins = create_bins(v_flip[:,j-1,ind1],vine.n_bin)
                                        val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1

                                vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                                for bb in range(0,vine.n_bin,1):
                                    mask = np.where(val_to_bin == bb)
                                    vv_bin = vv[mask[0],:]

                                    ### CDF FORCE UNIFORM
                                    vv_bin_new = vv_bin
                                    for zz in range(0,2,1):
                                        vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],vine.grid_u.ex)
                                    vv_bin = vv_bin_new
                                    ###

                                    tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])

                                    if flip_1 == True:
                                        v_flip[mask[0],j+1,ii] = copulaccdf(vine.copulas[tr][col][bb],vv_bin)
                                    else:
                                        v[mask[0],j+1,ii] = copulaccdf(vine.copulas[tr][col][bb],vv_bin)

                        else:
                            
                            ind_array = np.array(vine.ind_edge_rel[tr])
                            ind_col = np.where(ind_array == col)
                            ind_fin = ind_col[0][0]
                #             print('ind_fin',col)
                            if (vine.ind_edge_rel[tr][ind_fin+1] == col) & (vine.flip_flag[tr][ind_fin+1] == flip_flag):
                                col = ind_fin + 1
                            else:
                                col = ind_fin

                            if (j==0) | (vine.binning == False):
                                
                                data_u = data_u[...,np.newaxis]
                                trans = Transform(1)

                                ## Transform data
                                data_s = trans.forward_u(data_u)
                                data_x = trans.forward_s(data_s)

                                batch_size = tf.constant(2,tf.int32)
                                pd_grid_uv = vine.copulas[tr].pd_grid_uv[:,:,col]
                                cdf1 = vine.copulas[tr].cdf[:,:,col]

                                ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s[:,:,0],vine.grid_s.min,vine.grid_s.max,cdf1,axis=-2)

                                pd_points, ccdf_points = evaluate_points(data_s[:,:,0], batch_size, vine.grid_s, cdf1, pd_grid_uv)    

                                # Update Fp
                                interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_points, vine.grid_u.ex)
                                if flip_flag == False:
                                    v[:,j+1,ii] = interp_cdf_poi
                                else:
                                    v_flip[:,j+1,ii] = interp_cdf_poi

                            else: #binning
                                
                                ind1 = parent1
                                
                                if j == 1:
                                    ind1 = np.where(vine.nodes == ind1 +1)
                                    ind1 = ind1[0][0]
                                    bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                                    val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                                else:
                                    ind_par_now = vine.ind_vine[j-1][ind_now[1]]
                                    parent22, inx1, inx2 = parent_var(j-1,vine.ind_vine,ind_par_now)  

                                    ind1 = ind1 + j - 1
                                    if (vine.ind_vine[j-2][ind_par_now[0]][0] == parent22):
                                        bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                                        val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                                        val_to_bin = check_bins(v[:,j-1,ind1],bins)
                                    else:
                                        bins = create_bins(v_flip[:,j-1,ind1],vine.n_bin)
                                        val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1
                                        val_to_bin = check_bins(v_flip[:,j-1,ind1],bins)
                                
                                vv_all = np.zeros(np.shape(vv)[0],w.dtype)
                                for bb in range(0,vine.n_bin,1):
                                    mask = np.where(val_to_bin == bb)

                                    batch_size = tf.constant(1,tf.int32)
                                    pd_grid_uv = vine.copulas[tr].pd_grid_uv[:,:,col,bb]
                                    cdf1 = vine.copulas[tr].cdf[:,:,col,bb]
                                    
                                    data_u_bin = data_u[mask[0],:]
                                    
                                    ### CDF FORCE UNIFORM
                                    vv_bin_new = data_u_bin
                                    for zz in range(0,2,1):
                                        vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(data_u_bin[:,zz],data_u_bin[:,zz],vine.grid_u.ex)
                                    data_u_bin = vv_bin_new[...,np.newaxis]
                                    ###
                                    
                                    trans = Transform(1)

                                    ## Transform data
                                    data_s_bin = trans.forward_u(data_u_bin)
                                    data_x_bin = trans.forward_s(data_s_bin)
                                    

                                    ccdf_data = tfp.math.batch_interp_regular_nd_grid(data_s_bin[:,:,0],vine.grid_s.min,vine.grid_s.max,cdf1,axis=-2)

                                    pd_points, ccdf_points = evaluate_points(data_s_bin[:,:,0], batch_size, vine.grid_s, cdf1, pd_grid_uv)    

                                    # Update Fp
                                    interp_cdf_poi, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_points, vine.grid_u.ex)
                                    if flip_flag == False:
                                        v[mask[0],j+1,ii] = interp_cdf_poi
                                    else:
                                        v_flip[mask[0],j+1,ii] = interp_cdf_poi
                    cc2 += 1
                cc1 += 1

    u = np.reshape(v[:,0,:],np.shape(w))
    u1 = np.zeros(np.shape(u),u.dtype)

    u_ax = vine.grid_u.ax1
    u_ax1 = np.tile(u_ax[...,np.newaxis],[1,np.shape(u)[0]]).T
    gr_diff = vine.grid_u.diff1

    c= 0
    for i in range(d-1,-1,-1):
        ind = vine.r_matrix[i,i]-1
        u1[:,ind] = u[:,c]

        ###### COMMENT WHEN COMPUTING INFORMATION OTHERWISE ALL WRONG
        u_p1 = u[:,c][...,np.newaxis]
        u_upd = np.tile(u_p1,[1,np.shape(u_ax)[0]])
        u_diff = np.abs(u_ax1-u_upd)
        ind1 = np.argmin(u_diff,1)
        diff_val = np.take(gr_diff,ind1)
        u1[:,ind] = u[:,c] + diff_val*np.random.uniform(0.,1.,np.shape(diff_val)[0])
        
        c += 1

    u = u1

    #print(u)
    sample1 = np.zeros([cases,d],w.dtype)
    #print(sample1.shape)
    #print(d)
    sample_pdf = np.zeros([len(vine.Mar_G[i][0]),d],w.dtype)
    sample_pds = np.zeros([len(vine.Mar_G[i][0]),d],w.dtype)
    u = tf.cast(u,w.dtype)
    for i in tf.range(0,d,1,tf.int32):
        #print(sample_pdf.shape)
        #print(vine.Mar_G[i][1].shape)
        mar_s1 = tf.cast(vine.Mar_G[i][0],w.dtype)
        mar_p1 = tf.cast(vine.Mar_G[i][1],w.dtype)
        sample1_pro = tfp.math.interp_regular_1d_grid(u[:,i],x_ref_min=tf.math.reduce_min(mar_p1),x_ref_max=tf.math.reduce_max(mar_p1),y_ref=mar_s1)
        #print(tf.math.reduce_min(mar_s1),tf.math.reduce_max(mar_s1))     
        sample1[:,i] = prep_copula(sample1_pro,0).numpy()
        sample_pdf[:,i]=mar_p1.numpy() 
        sample_pds[:,i]=mar_s1.numpy() 
        
    
    #print(type(sample1))
    
    return sample1, u, sample_pdf, sample_pds

###################### SAMPLING FROM PARAMETRIC VINE COPULAS ################################

def vine_cop_par_sample(vine, cases):
    d = len(vine.r_matrix)
    n = len(vine.r_matrix) -1

    w = np.random.uniform(0,1,(cases,d))

    mag = np.max(vine.grid_u.ex)-1e-5  ### 1e-5 to constraints the boundary of the cdf
    mig = np.min(vine.grid_u.ex)+1e-5

    w = (mag-mig)*(w-np.min(w))/(np.max(w)-np.min(w))+mig

    w = w.astype(np.float32)

    v = np.zeros([cases,d,d],w.dtype)
    v_flip = np.zeros([cases,d,d],w.dtype)
    v[:,0,0] = w[:,0]
    
    for i in range(1,d,1):
        v[:,i,i] = w[:,i]

        c = 0
        for k in range(i-1,-1,-1):
            tr = k
            col = i-k-1
            ind_now = vine.ind_vine[k][c]

            if k == 0:
                tr1 = n-k
                col1 = n-i
                ind1 = vine.r_matrix[tr1,col1] #- 1
                ind1 = np.where(vine.nodes == ind1)
                ind1 = ind1[0][0]

                v2 = v[:,k,ind1][...,np.newaxis]
                v1 = v[:,k+1,i][...,np.newaxis]
                vv = np.concatenate((v1,v2),1)
                v[:,k,i] = copulainvccdf(vine.copulas[tr][col],vv)
            else:

                parent, inx1, inx2 = parent_var(k,vine.ind_vine,ind_now)

                if vine.ind_vine[k-1][ind_now[0]][0] != parent: 
                    v2 = v_flip[:,k,k+ind_now[0]][...,np.newaxis]
                else:
                    v2 = v[:,k,k+ind_now[0]][...,np.newaxis]

                v1 = v[:,k+1,i][...,np.newaxis]
                vv = np.concatenate((v1,v2),1)
                if vine.binning == False:
                    v[:,k,i] = copulainvccdf(vine.copulas[tr][col],vv)
                else:
                    
                    ind1 = parent
                    
                    if k == 1:
                        ind1 = np.where(vine.nodes == ind1 +1)
                        ind1 = ind1[0][0]
                        bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                        val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                    else:
                        ind_par_now = vine.ind_vine[k-1][ind_now[1]]
                        parent22, inx1, inx2 = parent_var(k-1,vine.ind_vine,ind_par_now)  

                        ind1 = ind1 + k - 1
                        if (vine.ind_vine[k-2][ind_par_now[0]][0] == parent22):
                            bins = create_bins(v[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v[:,k-1,ind1], bins) -1
                        else:
                            bins = create_bins(v_flip[:,k-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v_flip[:,k-1,ind1], bins) -1
                        
                    vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                    for bb in range(0,vine.n_bin,1):
                        mask = np.where(val_to_bin == bb)
                        vv_bin = vv[mask[0],:]
                        
                        ### CDF FORCE UNIFORM
                        vv_bin_new = vv_bin
                        for zz in range(0,2,1):
                            vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],vine.grid_u.ex)
                        vv_bin = vv_bin_new
                        ###
                        
                        v[mask[0],k,i] = copulainvccdf(vine.copulas[tr][col][bb],vv_bin)
                        
            c += 1
            
        if i < d -1:
            for ii in range(1,i+1,1):
                for j in range(0,ii,1):
                    tr = j
                    col = ii-j-1

                    ind_now = vine.ind_vine[j][ii-1-j]
                    
                    if j == n-2:
                        ind_sup = vine.ind_vine[j+1][0]
                    else:
                        ind_sup = vine.ind_vine[j+1][i-1-j]
                    
                    if j == 0:
                        tr1 = n-j
                        col1 = n-ii
                        ind1 = vine.r_matrix[tr1,col1] #- 1
                        ind1 = np.where(vine.nodes == ind1)
                        ind1 = ind1[0][0]

                        v2 = v[:,j,ind1][...,np.newaxis]
                    else:
                        parent1, inx1, inx2 = parent_var(j,vine.ind_vine,ind_now)
                        
                        if vine.ind_vine[j-1][ind_now[0]][0] != parent1:
                            v2 = v_flip[:,j,j+ind_now[0]][...,np.newaxis]
                        else:
                            v2 = v[:,j,j+ind_now[0]][...,np.newaxis]

                    v1 = v[:,j,ii][...,np.newaxis]

                    vv = np.concatenate((v1,v2),1)

                    parent, inx1, inx2 = parent_var(j+1,vine.ind_vine,ind_sup)                
                    u_edge = {ind_now[0], ind_now[1]}

                    if (j == 0) | (vine.binning == False):
                        if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                vv = np.concatenate((v2,v1),1)
                                v_flip[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
                            else:
                                v[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
                        else:
                            v[:,j+1,ii] = copulaccdf(vine.copulas[tr][col],vv)
                    else:

                        if (u_edge.issubset(inx1)) | (u_edge.issubset(inx2)):
                            if ind_now[0] != parent:
                                vv = np.concatenate((v2,v1),1)
                                flip_1 = True
                            else:
                                flip_1 = False
                        else:
                            flip_1 = False

                        ind1 = parent1

                        if j == 1:
                            ind1 = np.where(vine.nodes == ind1 +1)
                            ind1 = ind1[0][0]
                            bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                            val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                        else:
                            ind_par_now = vine.ind_vine[j-1][ind_now[1]]
                            parent22, inx1, inx2 = parent_var(j-1,vine.ind_vine,ind_par_now)  

                            ind1 = ind1 + j - 1
                            if (vine.ind_vine[j-2][ind_par_now[0]][0] == parent22):
                                bins = create_bins(v[:,j-1,ind1],vine.n_bin)
                                val_to_bin = np.digitize(v[:,j-1,ind1], bins) -1
                            else:
                                bins = create_bins(v_flip[:,j-1,ind1],vine.n_bin)
                                val_to_bin = np.digitize(v_flip[:,j-1,ind1], bins) -1

                        vv_all = np.zeros(np.shape(vv)[0],vv.dtype)
                        for bb in range(0,vine.n_bin,1):
                            mask = np.where(val_to_bin == bb)
                            vv_bin = vv[mask[0],:]

                            ### CDF FORCE UNIFORM
                            vv_bin_new = vv_bin
                            for zz in range(0,2,1):
                                vv_bin_new[:,zz], mar_s1, mar_p1 = kernel_cdf(vv_bin[:,zz],vv_bin[:,zz],vine.grid_u.ex)
                            vv_bin = vv_bin_new
                            ###

                            tau, p_value = stats.kendalltau(vv_bin[:,0],vv_bin[:,1])

                            if flip_1 == True:
                                v_flip[mask[0],j+1,ii] = copulaccdf(vine.copulas[tr][col][bb],vv_bin)
                            else:
                                v[mask[0],j+1,ii] = copulaccdf(vine.copulas[tr][col][bb],vv_bin)
                    
    u = np.reshape(v[:,0,:],np.shape(w))
    u1 = np.zeros(np.shape(u),u.dtype)
    
    u_ax = vine.grid_u.ax1
    u_ax1 = np.tile(u_ax[...,np.newaxis],[1,np.shape(u)[0]]).T
    gr_diff = vine.grid_u.diff1
    
    c= 0
    for i in range(d-1,-1,-1):
        ind = vine.r_matrix[i,i]-1
        u1[:,ind] = u[:,c]
        
        c += 1
    u = u1

    sample1 = np.zeros([cases,d],w.dtype)
    for i in tf.range(0,d,1,tf.int32):
        mar_s1 = vine.Mar_G[i][0]
        mar_p1 = vine.Mar_G[i][1]
        sample1_pro = tfp.math.interp_regular_1d_grid(u[:,i],tf.math.reduce_min(mar_p1),tf.math.reduce_max(mar_p1),mar_s1)
        sample1[:,i] = prep_copula(sample1_pro,0).numpy()
    
    return sample1

# File: src/DVC_tensorflow/utils/.ipynb_checkpoints/bijector-checkpoint.py
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class NormalCDF(tfb.Bijector):
    """Bijector that encodes normal CDF and inverse CDF functions.

    We follow the convention that the `inverse` represents the CDF
    and `forward` the inverse CDF (the reason for this convention is
    that inverse CDF methods for sampling are expressed a little more
    tersely this way).

    """

    def __init__(self, loc, scale):
        self.normal_dist = tfd.Normal(loc=loc, scale=scale)
        super(NormalCDF, self).__init__(
            forward_min_event_ndims=0,
            validate_args=False,
            name="NormalCDF")

    def forward(self, y):
        # Inverse CDF of normal distribution.
        return self.normal_dist.quantile(y)

    def inverse(self, x):
        # CDF of normal distribution.
        return self.normal_dist.cdf(x)

    def inverse_log_det_jacobian(self, x):
        # Log PDF of the normal distribution.
        return self.normal_dist.log_prob(x)
    

class GammaCDF(tfb.Bijector):
    """Bijector that encodes normal CDF and inverse CDF functions.

    We follow the convention that the `inverse` represents the CDF
    and `forward` the inverse CDF (the reason for this convention is
    that inverse CDF methods for sampling are expressed a little more
    tersely this way).

    """

    def __init__(self,concentration,rate):
        self.gamma_dist = tfd.Gamma(concentration=concentration, rate=rate)
        super(GammaCDF, self).__init__(
            forward_min_event_ndims=0,
            validate_args=False,
            name="GammaCDF")

    def forward(self, y):
        # Inverse CDF of normal distribution.
        return self.gamma_dist.quantile(y)

    def inverse(self, x):
        # CDF of normal distribution.
        return self.gamma_dist.cdf(x)

    def inverse_log_det_jacobian(self, x):
        # Log PDF of the normal distribution.
        return self.gamma_dist.log_prob(x)

# File: src/DVC_tensorflow/utils/.ipynb_checkpoints/dataset_op-checkpoint.py
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold

def kfold(data, n_splits):    
    train_ind = tf.TensorArray(tf.int32,size=n_splits)
    test_ind = tf.TensorArray(tf.int32,size=n_splits)
    k = 0
    kf = KFold(n_splits=n_splits, shuffle=True) # Instantiate the cross validator
    for train_index, test_index in kf.split(data): #[:,:,0]
        train_ind = train_ind.write(k,train_index)
        test_ind = test_ind.write(k,test_index)
        k = k+1

    train_ind = train_ind.stack()
    test_ind = test_ind.stack()
    return train_ind, test_ind

@tf.function(experimental_relax_shapes=True)
def data_split(data,ind):
    # DIVIDE DATA IN TRAINING AND TEST SET
    if tf.math.equal(tf.shape(tf.shape(data)),2):
        data = data[...,tf.newaxis]
    
    n_splits = tf.shape(ind)[0]
    data_new = tf.TensorArray(data.dtype,size=n_splits)
    for j in tf.range(0,n_splits,1,tf.int32):
        data_new1 = tf.gather_nd(data,ind[j,:][...,tf.newaxis])
        data_new = data_new.write(j,data_new1)

    data_new = data_new.stack()
    data_new = tf.transpose(data_new,perm=[1,2,3,0])
    return data_new


############### BIN FUNCTIONS

def create_bins(data, n_bin):
    len_bin = tf.shape(data)[0]/n_bin
    len_bin = tf.cast(len_bin,tf.int32)
    data = np.sort(data)
    bins = [data[0]-1e-15]
    for i in range(1,n_bin,1):
        bins.append(data[len_bin*i])
    bins.append(data[-1]+1e-15)
    return bins

def check_bins(data,bins):
    n_bin = np.shape(bins)[0] -1
    len_bin = np.shape(data)[0]/n_bin
    len_bin = tf.cast(len_bin,tf.int32)
    val_to_bin = np.digitize(data, bins) - 1
    
    ind_sort = np.argsort(data)
    
    val_to_bin2 = val_to_bin
    for bb in range(0,n_bin,1):
        val_to_bin2[ind_sort[bb*len_bin:(bb+1)*len_bin]] = bb*np.ones(len_bin,data.dtype)

    return val_to_bin2

# File: src/DVC_tensorflow/utils/.ipynb_checkpoints/interpolation-checkpoint.py
import tensorflow as tf
import numpy as np

###################### NEAREST INTERPOLATION #####################

@tf.function(experimental_relax_shapes=True)
def nearestInterp2d(sample_s, pro_s1, pro_s2, pd_grid_uv):
    # Nearest neighbor interpolation on the grid
    len_sample = tf.shape(sample_s[:,0])[0]
    len_grid = tf.shape(pro_s1)[0]
    #pro_s1_tile = tf.tile(pro_s1,tf.constant(len_sample,dtype=tf.int32,shape=[1]))
    pro_s1_tile = tf.tile(pro_s1,[len_sample])
    pro_s1_tile = tf.transpose(tf.reshape(pro_s1_tile,[len_sample,len_grid])) #[100,20000]
    #pro_s2_tile = tf.tile(pro_s2,tf.constant(len_sample,dtype=tf.int32,shape=[1]))
    pro_s2_tile = tf.tile(pro_s2,[len_sample])
    pro_s2_tile = tf.transpose(tf.reshape(pro_s2_tile,[len_sample,len_grid])) #[100,20000]
    
    sample_s1_tile = tf.tile(sample_s[:,0],[len_grid])
    sample_s1_tile = tf.reshape(sample_s1_tile,[len_grid,len_sample]) #[100,20000]
    sample_s2_tile = tf.tile(sample_s[:,1],[len_grid])
    sample_s2_tile = tf.reshape(sample_s2_tile,[len_grid,len_sample]) #[100,20000]
    
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

# File: src/DVC_tensorflow/utils/.ipynb_checkpoints/prob_op-checkpoint.py
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
def kernel_cdf_batch(data, y, ex, batch_size):
    
    margin_s = tf.sort(data)
    margin_s, idx = tf.unique(margin_s)
    
    exc = tf.shape(data)[0] - tf.shape(margin_s)[0]
    margin_s_exc = tf.concat([margin_s,tf.zeros(exc,data.dtype)],0)
    
    kka = tf.TensorArray(tf.int32,size=batch_size)

    batch_len = tf.shape(margin_s_exc)[0]/batch_size
    batch_len = tf.cast(batch_len,tf.int32)

    for i in tf.range(0,batch_size,1):
#         kka1 = op_cdf(data,margin_s_exc[batch_len*i:batch_len*(i+1)])
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
def kernel_cdf(data, y, ex):
#     kka = tf.TensorArray(tf.int32,size=tf.shape(data)[0])
#     for i in tf.range(0,tf.shape(data)[0],1,tf.int32):  #tf.shape(data)[0]
#         kka = kka.write(i,tf.shape(tf.where(tf.math.less_equal(data,data[i])))[0])
    
#     kka = kka.stack()
#     u = kka/(tf.shape(data)[0]+1)
    
#     nn = tf.argsort(u)
#     margin_p = tf.gather_nd(u,nn[...,tf.newaxis])
#     margin_s = tf.gather_nd(data,nn[...,tf.newaxis])
    
#     uu,nn = tf.unique(margin_p)
#     margin_p = tf.gather_nd(margin_p,nn[...,tf.newaxis])
#     margin_s = tf.gather_nd(margin_s,nn[...,tf.newaxis])
    
#     margin_p = tf.cast(margin_p,data.dtype)
    
#     margin_s = tf.sort(data)
#     kka = tf.range(1,tf.shape(data)[0]+1,1,dtype=tf.int32)
#     margin_p = kka/(tf.shape(data)[0]+1)
#     margin_p = tf.cast(margin_p,data.dtype)
    
    
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
#     interp_cdf = check_bound_and_nan(interp_cdf,tf.math.reduce_max(margin_s),tf.math.reduce_min(margin_s))
    
    return interp_cdf, margin_s, margin_p

# @tf.function
# def kernel_cdf(data, ex):
#     # Compute cdf of the data
#     margin_s = tf.sort(data)
#     kka = tf.range(1,tf.shape(data)[0]+1,1,dtype=tf.int32)
#     margin_p = kka/(tf.shape(data)[0]+1)
#     margin_p = tf.cast(margin_p,data.dtype)
#     margin_p = check_bound(margin_p,ex)
#     #pp = tf.math.cumsum(kka)/(len(ccc)+1)
#     return margin_p, margin_s

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
def fixed_point(xx,N,I,a2):
        # Ir represents function t-zeta*gamma^[l](t)
        pi = tf.cast(m.pi,a2.dtype)
        l = tf.constant(7, dtype=a2.dtype)
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
def dct1d(data):
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
def idct1d(data):
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
def histc1(X, bins):
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

# File: src/DVC_tensorflow/utils/.ipynb_checkpoints/tensor_op-checkpoint.py
import tensorflow as tf

#################### CHECK BOUNDARIES OF TENSOR ########################

@tf.function(experimental_relax_shapes=True)
def check_bound(data,mesh):
    # Clips tensor value to its minimum and maximum
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
    ind_max = tf.where(tf.math.greater(data,max_m))
    ind_min = tf.where(tf.math.less(data,min_m))
#     upd_min = tf.tile([min_m],[ tf.shape(ind_min)[0]])
#     upd_max = tf.tile([max_m],[ tf.shape(ind_max)[0]])
    
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

# File: src/DVC_tensorflow/utils/bijector.py
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class NormalCDF(tfb.Bijector):
    """Bijector that encodes normal CDF and inverse CDF functions.

    We follow the convention that the `inverse` represents the CDF
    and `forward` the inverse CDF (the reason for this convention is
    that inverse CDF methods for sampling are expressed a little more
    tersely this way).

    """

    def __init__(self, loc, scale):
        self.normal_dist = tfd.Normal(loc=loc, scale=scale)
        super(NormalCDF, self).__init__(
            forward_min_event_ndims=0,
            validate_args=False,
            name="NormalCDF")

    def forward(self, y):
        # Inverse CDF of normal distribution.
        return self.normal_dist.quantile(y)

    def inverse(self, x):
        # CDF of normal distribution.
        return self.normal_dist.cdf(x)

    def inverse_log_det_jacobian(self, x):
        # Log PDF of the normal distribution.
        return self.normal_dist.log_prob(x)
    

class GammaCDF(tfb.Bijector):
    """Bijector that encodes normal CDF and inverse CDF functions.

    We follow the convention that the `inverse` represents the CDF
    and `forward` the inverse CDF (the reason for this convention is
    that inverse CDF methods for sampling are expressed a little more
    tersely this way).

    """

    def __init__(self,concentration,rate):
        self.gamma_dist = tfd.Gamma(concentration=concentration, rate=rate)
        super(GammaCDF, self).__init__(
            forward_min_event_ndims=0,
            validate_args=False,
            name="GammaCDF")

    def forward(self, y):
        # Inverse CDF of normal distribution.
        return self.gamma_dist.quantile(y)

    def inverse(self, x):
        # CDF of normal distribution.
        return self.gamma_dist.cdf(x)

    def inverse_log_det_jacobian(self, x):
        # Log PDF of the normal distribution.
        return self.gamma_dist.log_prob(x)

# File: src/DVC_tensorflow/utils/dataset_op.py
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold

def kfold(data, n_splits):    
    train_ind = tf.TensorArray(tf.int32,size=n_splits)
    test_ind = tf.TensorArray(tf.int32,size=n_splits)
    k = 0
    kf = KFold(n_splits=n_splits, shuffle=True) # Instantiate the cross validator
    for train_index, test_index in kf.split(data): #[:,:,0]
        train_ind = train_ind.write(k,train_index)
        test_ind = test_ind.write(k,test_index)
        k = k+1

    train_ind = train_ind.stack()
    test_ind = test_ind.stack()
    return train_ind, test_ind

@tf.function(experimental_relax_shapes=True)
def data_split(data,ind):
    # DIVIDE DATA IN TRAINING AND TEST SET
    if tf.math.equal(tf.shape(tf.shape(data)),2):
        data = data[...,tf.newaxis]
    
    n_splits = tf.shape(ind)[0]
    data_new = tf.TensorArray(data.dtype,size=n_splits)
    for j in tf.range(0,n_splits,1,tf.int32):
        data_new1 = tf.gather_nd(data,ind[j,:][...,tf.newaxis])
        data_new = data_new.write(j,data_new1)

    data_new = data_new.stack()
    data_new = tf.transpose(data_new,perm=[1,2,3,0])
    return data_new


############### BIN FUNCTIONS

def create_bins(data, n_bin):
    len_bin = tf.shape(data)[0]/n_bin
    len_bin = tf.cast(len_bin,tf.int32)
    data = np.sort(data)
    bins = [data[0]-1e-15]
    for i in range(1,n_bin,1):
        bins.append(data[len_bin*i])
    bins.append(data[-1]+1e-15)
    return bins

def check_bins(data,bins):
    n_bin = np.shape(bins)[0] -1
    len_bin = np.shape(data)[0]/n_bin
    len_bin = tf.cast(len_bin,tf.int32)
    val_to_bin = np.digitize(data, bins) - 1
    
    ind_sort = np.argsort(data)
    
    val_to_bin2 = val_to_bin
    for bb in range(0,n_bin,1):
        val_to_bin2[ind_sort[bb*len_bin:(bb+1)*len_bin]] = bb*np.ones(len_bin,data.dtype)

    return val_to_bin2

# File: src/DVC_tensorflow/utils/interpolation.py
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


# File: src/DVC_tensorflow/utils/prob_op.py
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

# File: src/DVC_tensorflow/utils/tensor_op.py
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

# File: src/DVC_tensorflow/vine_tree/.ipynb_checkpoints/tree_op-checkpoint.py
import math as m
import numpy as np
import random
from scipy.stats import kendalltau

###################### CHECK OPTIMAL TREE ##########################

### Prim algorithm - Section 23 Minium Spanning Three - Algorithm book

# def optimal_tree(data):
#     random.seed(9001)
#     V = set(range(0,data.shape[1])) #{0,1,2,3,4,5}
#     Q = set()
#     edges = []
#     weights = []
#     u = random.randint(0,data.shape[1]-1)
#     Q.add(u)
#     V.remove(u)
#     while V:
#         max_v = -m.inf
#         for i in Q:
#             for j in V:
#                 tau, p_value = kendalltau(data[:,i], data[:,j])
#                 if abs(tau) > max_v:
#                     max_v = abs(tau)
#                     u = i
#                     v = j
#         Q.add(v)
#         V.remove(v)
#         if v>u:
#             edges.append([v,u])
#         else:
#             edges.append([u,v])
#         weights.append(max_v)
#     return edges,weights

# def optimal_tree(data, data_flip, ind_vine, tr):
#     random.seed(9001)
#     V = set(range(0,data.shape[1]-tr)) #{0,1,2,3,4,5}
#     Q = set()
#     edges = []
#     weights = []
#     u = random.randint(0,data.shape[1]-1-tr)
#     Q.add(u)
#     V.remove(u)
# #     print('Q',Q)
# #     print('V',V)
#     while V:
#         max_v = -m.inf
#         for i in Q:
# #             print('u',i)
#             for j in V:
# #                 print('v',j)
#                 if tr == 0:
#                     tau, p_value = kendalltau(data[:,i], data[:,j])
#                     if abs(tau) > max_v:
#                         max_v = abs(tau)
#                         u = i
#                         v = j
#                 else: 
#                     par, inx1, inx2 = parent_var(tr,ind_vine,[i,j])
# #                     print('par',par)
# #                     print('ind_vine prev 1',ind_vine[tr-1][i])
#                     if par != None:
#                         if par != ind_vine[tr-1][i][0]:
#                             tau, p_value = kendalltau(data_flip[:,i], data[:,j])
#                             if abs(tau) > max_v:
#                                 max_v = abs(tau)
#                                 u = i
#                                 v = j
#                         else:
#                             tau, p_value = kendalltau(data[:,i], data[:,j])
#                             if abs(tau) > max_v:
#                                 max_v = abs(tau)
#                                 u = i
#                                 v = j
#         Q.add(v)
#         V.remove(v)
# #         print('---------')
# #         if v>u:
# #             edges.append([v,u])
# #         else:
#         edges.append([u,v])
#         weights.append(max_v)
#     return edges,weights

def optimal_tree(data, data_flip, ind_vine, tr, rand):
    random.seed(9001)
    V = set(range(0,data.shape[1]-tr)) #{0,1,2,3,4,5}
    Q = set()
    edges = []
    weights = []
    u = random.randint(0,data.shape[1]-1-tr)
    Q.add(u)
    V.remove(u)
#     print('Q',Q)
#     print('V',V)
#     c = 0
    while V:
        max_v = -m.inf
        for i in Q:
#             print('u',i)
            for j in V:
#                 print('v',j)
                if tr == 0:
                    if rand == False:
                        tau, p_value = kendalltau(data[:,i], data[:,j])
                    else:
                        tau = np.random.uniform(-1.,1.,1)
                    if abs(tau) > max_v:
                        max_v = abs(tau)
                        u = i
                        v = j
                else: 
                    par, inx1, inx2 = parent_var(tr,ind_vine,[i,j])
#                     print('par',par)
#                     print('ind_vine prev 1',ind_vine[tr-1][i])
                    if par != None:
                        if par != ind_vine[tr-1][i][0]:
                            if rand == False:
                                tau, p_value = kendalltau(data_flip[:,i], data[:,j])
                            else:
                                tau = np.random.uniform(-1.,1.,1)
                            if abs(tau) > max_v:
                                max_v = abs(tau)
                                u = i
                                v = j
                        else:
                            if rand == False:
                                tau, p_value = kendalltau(data[:,i], data[:,j])
                            else:
                                tau = np.random.uniform(-1.,1.,1)
                            if abs(tau) > max_v:
                                max_v = abs(tau)
                                u = i
                                v = j
        Q.add(v)
        V.remove(v)
#         print('---------')
#         if v>u:
#             edges.append([v,u])
#         else:

#         if c == 0:
#             edges.append([v,u])
#         else:
        edges.append([u,v])
        weights.append(max_v)
#         c += 1
    return edges,weights

###################### BUILD LIST OF EDGES #########################

def build_edges(tree_index):
    n_ind = tree_index.shape[0]
    
    e_0 = {tree_index[n_ind-2,n_ind-2],tree_index[n_ind-1,n_ind-2]}
    E_1 = [e_0]
    E = [E_1]
    
#     print('e_0',e_0)
    
    for i in range(1,n_ind-1,1):
        tmp = []
        E.append(tmp)

    n = n_ind-1
    ind_e = 0
    for i in range(n-2,-1,-1):  #-1
#         print('Column',i)
        e_1 = {tree_index[i,i],tree_index[n,i]}
        E[0].append(e_1)
        e_new = e_1
#         print('e_1',e_1)

        for k in range(1,n_ind-i-1,1):
#             print('k',k)
            u = set()
            for uu in range(n-k,n,1):
#                 print('uu',uu)
                u.add(tree_index[uu,i])
#                 print('u_elem',tree_index[uu,i])
            u.add(tree_index[n,i])
            
#             print('u',u)
            flag = False
            for hh in range(0,len(E[k-1]),1):
#                 print('hh',hh)
#                 print('prova_edge',E[k-1][hh])
                if isedge(E[k-1][hh],u):
                    flag = True
                    e_new1 = [e_new,E[k-1][hh]]
                    E[k].append(e_new1)
#                     print('e_new1',e_new1)
            if not flag:
                raise Exception('The matrix is not a regular vine')
            e_new = e_new1
#         print('--------------------')
    return E

# def build_edges(tree_index):
#     n_ind = tree_index.shape[0]
    
#     e_0 = {tree_index[n_ind-2,n_ind-2],tree_index[n_ind-1,n_ind-2]}
#     E_1 = [e_0]
#     E = [E_1]
    
#     for i in range(1,n_ind-1,1):
#         tmp = []
#         E.append(tmp)

#     n = n_ind-1
#     ind_e = 0
#     for i in range(n-2,-1,-1):  #-1
# #         print('Column',i)
#         e_1 = {tree_index[i,i],tree_index[n,i]}
#         E[0].append(e_1)
#         e_new = e_1

#         for k in range(1,n_ind-i-1,1):

#             u = set()
#             for uu in range(n-k,n,1):
#                 u.add(tree_index[uu,i])
#             u.add(tree_index[n,i])

#             for hh in range(0,len(E[k-1]),1):
#                 if isedge(E[k-1][hh],u):
#                     e_new1 = [e_new,E[k-1][hh]]
#                     E[k].append(e_new1)
#             e_new = e_new1
# #         print('--------------------')
#     return E

#################### RETURN LIST ON INDEX #######################

def edges_index(E,r_matrix,tr):
    edges_ind = []
    n = r_matrix.shape[0]-1
    if tr == 0:
        for i in range(n-1,-1,-1):
            ind = [r_matrix[n,i]-1,r_matrix[i,i]-1]
            edges_ind.append(ind)
    else:
        for ii in range(0,len(E[tr]),1):
            edge = E[tr][ii]
            for yy in range(0,len(E[tr-1]),1):
                if edge[0] == E[tr-1][yy]:
                    ind0 = yy
                if edge[1] == E[tr-1][yy]:
                    ind1 = yy
            ind = [ind1,ind0]
            edges_ind.append(ind)
    return edges_ind

# def edges_index(E,r_matrix,tr):
#     edges_ind = []
#     n = r_matrix.shape[0]-1
#     if tr == 0:
#         for i in range(n-1,-1,-1):
#             ind = [r_matrix[n,i]-1,r_matrix[i,i]-1]
#             edges_ind.append(ind)
#     else:
#         for ii in range(0,len(E[tr]),1):
#             edge = E[tr][ii]
#             for yy in range(0,len(E[tr-1]),1):
#                 if edge[0] == E[tr-1][yy]:
#                     ind0 = yy
#                 if edge[1] == E[tr-1][yy]:
#                     ind1 = yy
#             ind = [ind0,ind1]
#             edges_ind.append(ind)
#     return edges_ind

# def edges_index(E,tr):
#     edges_ind = []
#     for ii in range(0,len(E[tr]),1):
#         edge = E[tr][ii]
#         if type(edge) is list:
#             for yy in range(0,len(E[tr-1]),1):
#                 if edge[0] == E[tr-1][yy]:
#                     ind0 = yy
#                 if edge[1] == E[tr-1][yy]:
#                     ind1 = yy
#             ind = [ind0,ind1]
#         else:
#             ind = []
#             for x in edge:
#                 ind.append(x -1)
#         edges_ind.append(ind)
#     return edges_ind


################### CHECK IF IS AN EDGE #######################

def isedge(edge,u):
    if type(edge) is list:
        return (isedge(edge[0],u)) & (isedge(edge[1],u))
    else:
        return edge.issubset(u)


############################################ PREPARE C-VINE AND D-VINE MATRIX ################################

def prepare_vine(vine_type, dim):
    if vine_type == 'c-vine':
        r_matrix = np.tril(np.tile(np.array(range(dim,0,-1)),(dim,1)).T)
        
        ### EDGES FOR THE CODES
        ind_vine = []
        for i in range(0,len(r_matrix)-1,1):
            ind_vine1 = []
            for j in range(1,len(r_matrix)-i,1):
                ind_vine1.append([0,j])
            ind_vine.append(ind_vine1)
            
    if vine_type == 'd-vine':
        r_matrix = np.zeros((dim,dim),np.int32)
        for i in range(0,dim,1):
            r_matrix[i,i] = dim-i #-1
        for j in range(0,dim-1,1):
            c = 1 #0
            for i in range(j+1,dim,1):
                r_matrix[i,j] = c
                c += 1
                
        ### EDGES FOR THE CODES        
        ind_vine = []
        for i in range(0,len(r_matrix)-1,1):
            ind_vine1 = []
            for j in range(0,len(r_matrix)-i-1,1):
                ind_vine1.append([j,j+1])
            ind_vine.append(ind_vine1)
        
    ### NODES

    d = len(r_matrix)
    n = d-1
    nodes = np.zeros(d,np.int32)
    for i in range(0,d,1):
        nodes[i]=r_matrix[i,i]

    nodes = np.flip(nodes)
    print('nodes:')
    print(nodes)
    
    ### EDGES of the vine
    matrix_edges = []
    c = 0
    for i in range(n,0,-1):
    #     print('level',c)
        edge1 = []
        for j in range(i-1,-1,-1):
            str1 = '(' + str(r_matrix[i,j]) + ',' + str(r_matrix[j,j])
            c = 0
            for ii in range(i+1,n+1,1):
                if c == 0:
                    str1 = str1 + '|' + str (r_matrix[ii,j])
                else:
                    str1 = str1 + ','  + str (r_matrix[ii,j])
                c += 1
            str1 = str1 + ')'
            edge1.append(str1)
        c += 1
        matrix_edges.append(edge1)
    
    print('edges:')
    for i in range(0,len(matrix_edges),1):
        print(matrix_edges[i])
    return r_matrix, ind_vine, nodes, matrix_edges
    
################  PREPARE REGULAR MATRIX ######################

def prepare_regular(r_matrix):
    E = build_edges(r_matrix)
    
    ### EDGES FOR THE CODES
    
    ind_vine = []
    for i in range(0,len(E),1):
        ind_ee = edges_index(E,r_matrix,i)
        ind_vine.append(ind_ee)
        
    ### NODES

    d = len(r_matrix)
    n = d-1
    nodes = np.zeros(d,np.int32)
    for i in range(0,d,1):
        nodes[i]=r_matrix[i,i]

    nodes = np.flip(nodes)
    print('nodes:')
    print(nodes)
    
    ### EDGES of the vine
    matrix_edges = []
    c = 0
    for i in range(n,0,-1):
    #     print('level',c)
        edge1 = []
        for j in range(i-1,-1,-1):
            str1 = '(' + str(r_matrix[i,j]) + ',' + str(r_matrix[j,j])
            c = 0
            for ii in range(i+1,n+1,1):
                if c == 0:
                    str1 = str1 + '|' + str (r_matrix[ii,j])
                else:
                    str1 = str1 + ','  + str (r_matrix[ii,j])
                c += 1
            str1 = str1 + ')'
            edge1.append(str1)
        c += 1
        matrix_edges.append(edge1)
    
    print('edges:')
    for i in range(0,len(matrix_edges),1):
        print(matrix_edges[i])
    return E, ind_vine, nodes, matrix_edges


###################### GET PARENT VARIABLE ##############################

def parent_var(k,ind_vine,edge):
    u = set()
    u.add(ind_vine[k-1][edge[0]][0])
    u.add(ind_vine[k-1][edge[0]][1])

    u1 = set()
    u1.add(ind_vine[k-1][edge[1]][0])
    u1.add(ind_vine[k-1][edge[1]][1])
    
    parent = None
    inter = u.intersection(u1)
    for elem in inter:
        parent = elem
    return parent, u, u1


######################## CHECK WHEN TO FLIP - R-MATRIX ORDER ###################################

def flip_check_all(ind_vine,tr, binning, n_bin):
    if tr < len(ind_vine)-1:
        ind_ee1 = ind_vine[tr+1]

        u_set = []
        parent = []
        for edges in ind_ee1:
            parent1, inx1, inx2 = parent_var(tr+1,ind_vine,edges)
            u_union = inx1.union(inx2)

            u_set.append(u_union)
            parent.append(parent1)
    else:
        ind_ee1 = [0,1]
        u_set = [{0,1}]
        parent = [0]

    parent_all = []
    edges_now = ind_vine[tr]
    ind_edge_rel1 = []
    flip_flag1 = []

    for j in range(0,len(edges_now),1):
        edge = edges_now[j]
        uu_now = {edge[0],edge[1]}
        parent_now = []
        parent_now_set = set()
        for jj in range(0,len(u_set),1):
            uu = u_set[jj]
            if uu_now.issubset(uu):
                if not {parent[jj]}.issubset(parent_now_set):
                    parent_now.append(parent[jj])
                    parent_now_set.add(parent[jj])
                
        # Check if they are all equal
        if len(set(parent_now)) <= 1:
            parent_now = [parent_now[0]]

        parent_all.append(parent_now)
        for par in parent_now:

#             if binning == False:

            if edge[0] != par:

                flip_flag1.append(True)
            else:
                flip_flag1.append(False)
#             else: #binning
#                 flip_flag_bin = []
#                 for bb in range(0,n_bin,1):

#                     if edge[0] != par:
#                         flip_flag_bin.append(True)
#                     else:
#                         flip_flag_bin.append(False)
#                 flip_flag1.append(flip_flag_bin)
            ind_edge_rel1.append(j)
    return flip_flag1, ind_edge_rel1, parent_all


################################## PREPARE R-MATRIX OPTIMAL AND RANDOM ###################

def prepare_optimal(d, ind_vine):
    E = []
    uu_uni = []
    par = []
    diff = []
    for tr in range(0,d-1,1):
        E.append([])
        uu_uni.append([])
        par.append([])
        diff.append([])
        
    for ii in range(0,len(ind_vine[0]),1):
        E[0].append({ind_vine[0][ii][0]+1,ind_vine[0][ii][1]+1})  
    
    u_union = set()
    for tr in range(1,d-1,1):
        for ii in range(0,len(ind_vine[tr]),1):
            ind1 = ind_vine[tr][ii]
            E[tr].append([E[tr-1][ind1[0]],E[tr-1][ind1[1]]])
            if tr ==1:
                u_union = E[tr-1][ind1[0]].union(E[tr-1][ind1[1]])
                parent = E[tr-1][ind1[0]].intersection(E[tr-1][ind1[1]])
                diff1 = u_union - parent
            else:
                u_union = uu_uni[tr-1][ind1[0]].union(uu_uni[tr-1][ind1[1]])
                parent = uu_uni[tr-1][ind1[0]].intersection(uu_uni[tr-1][ind1[1]])
                diff1 = u_union - parent
            uu_uni[tr].append(u_union)
            par[tr].append(parent)
            diff[tr].append(diff1)
    
    rr = np.zeros((d,d),np.int32)
    n = len(rr)-1
    
    for tr in range(d-2,-1,-1): #0
#         print('tr',tr)
        ind_list = set()
        for j in range(0,n-tr,1):
#             print('j',j)
            edge = []
            if tr > 0:
                for ii in range(0,len(diff[tr]),1):
                    edge1 = []
                    for elem in diff[tr][ii]:
                        edge1.append(elem)
                    edge.append(edge1)
            else:
                for ii in range(0,len(E[tr]),1):
                    edge1 = []
#                     print('aa',E[tr][ii])
                    for elem in E[tr][ii]:
                        edge1.append(elem)
                    edge.append(edge1)
#             print(edge)
#             print('diff',diff[tr])
#             print('ind_list',ind_list)

            if tr == d-2:
                rr[j,j] = edge[ii][0]
                rr[n-tr,j] = edge[ii][1]

            if (tr > 0) & (tr < d-2):            
                for ii in range(0,len(diff[tr]),1):
#                     print('ii',ii)
                    if {ii}.issubset(ind_list) == False:
                        a1 = edge[ii][0]
                        a2 = edge[ii][1]
#                         print('a1',a1)
                        if j == d-2-tr:
                            rr[j,j] = a1
                        if (rr[j,j] == a1):
                            ind1 = ii
                            ind2 = 1
                            ind_list.add(ind1)
                        elif (rr[j,j] == a2):
                            ind1 = ii
                            ind2 = 0
                            ind_list.add(ind1)

                rr[n-tr,j] = edge[ind1][ind2]
            else:
                for ii in range(0,len(E[tr]),1):
                    if {ii}.issubset(ind_list) == False:
                        a1 = edge[ii][0]
                        a2 = edge[ii][1]

                        if j == d-2-tr:
                            rr[j,j] = a1

                        if (rr[j,j] == a1):
                            ind1 = ii
                            ind2 = 1
                            ind_list.add(ind1)
                        elif (rr[j,j] == a2):
                            ind1 = ii
                            ind2 = 0
                            ind_list.add(ind1)
#                 print('ind1',ind1)
#                 print('ind2',ind2)
                rr[n-tr,j] = edge[ind1][ind2]

#             print(rr)
#             print('----------')

    nodes = np.zeros(d,np.int32)
    V = set(range(1,d+1))
    for i in range(0,d,1):
        nodes[i]=rr[i,i]
        u_nod = {nodes[i]}
        if u_nod.issubset(V):
            V.remove(nodes[i])
    nodes = np.flip(nodes)

    for elem in V:
        ind = np.where(nodes == 0)
        nodes[nodes == 0] = elem
        rr[n-ind[0],n-ind[0]] = elem
    
    
    return rr, E, nodes #uu_uni, par, diff, E

# def prepare_optimal(d, ind_vine, nodes):
#     E = []
#     uu_uni = []
#     par = []
#     diff = []
#     for tr in range(0,d,1):
#         E.append([])
#         uu_uni.append([])
#         par.append([])
#         diff.append([])
        
#     for ii in range(0,len(ind_vine[0]),1):
#         E[0].append({ind_vine[0][ii][0]+1,ind_vine[0][ii][1]+1})  
    
#     u_union = set()
#     for tr in range(1,d-1,1):
#         for ii in range(0,len(ind_vine[tr]),1):
#             ind1 = ind_vine[tr][ii]
#             E[tr].append([E[tr-1][ind1[0]],E[tr-1][ind1[1]]])
#             if tr ==1:
#                 u_union = E[tr-1][ind1[0]].union(E[tr-1][ind1[1]])
#                 parent = E[tr-1][ind1[0]].intersection(E[tr-1][ind1[1]])
#                 diff1 = u_union - parent
#             else:
#                 u_union = uu_uni[tr-1][ind1[0]].union(uu_uni[tr-1][ind1[1]])
#                 parent = uu_uni[tr-1][ind1[0]].intersection(uu_uni[tr-1][ind1[1]])
#                 diff1 = u_union - parent
#             uu_uni[tr].append(u_union)
#             par[tr].append(parent)
#             diff[tr].append(diff1)
            
#     rr = np.zeros((d,d),np.int32)

#     n = len(rr)-1

#     for i in range(0,d,1):
#         rr[i,i] = nodes[n-i]

#     c = 0
#     for tr in range(d-2,-1,-1): #0
# #         print('tr',tr)
#         ind_list = set()
#         for j in range(0,n-tr,1):
# #             print('j',j)
#             edge = []
#             if tr > 0:
#                 for ii in range(0,len(diff[tr]),1):
#                     edge1 = []
#                     for elem in diff[tr][ii]:
#                         edge1.append(elem)
#                     edge.append(edge1)
#             else:
#                 for ii in range(0,len(E[tr]),1):
#                     edge1 = []
# #                     print('aa',E[tr][ii])
#                     for elem in E[tr][ii]:
#                         edge1.append(elem)
#                     edge.append(edge1)
# #             print(edge)

#             if tr > 0:
#                 for ii in range(0,len(diff[tr]),1):
#                     if {ii}.issubset(ind_list) == False:
#                         a1 = edge[ii][0]
#                         a2 = edge[ii][1]
#                         if (rr[j,j] == a1):
#                             ind1 = ii
#                             ind2 = 1
#                             ind_list.add(ind1)
#                         elif (rr[j,j] == a2):
#                             ind1 = ii
#                             ind2 = 0
#                             ind_list.add(ind1)
#             else:
#                 for ii in range(0,len(E[tr]),1):
#                     if {ii}.issubset(ind_list) == False:
#                         a1 = edge[ii][0]
#                         a2 = edge[ii][1]
#                         if (rr[j,j] == a1):
#                             ind1 = ii
#                             ind2 = 1
#                             ind_list.add(ind1)
#                         elif (rr[j,j] == a2):
#                             ind1 = ii
#                             ind2 = 0
#                             ind_list.add(ind1)
#             rr[n-tr,j] = edge[ind1][ind2]
#     return rr, E


####################################### RANDOM R-MATRIX  #########################################################

def random_tree(vine_depth, ind_vine, tr):
    random.seed(9001)
    V = set(range(0,vine_depth-tr)) #{0,1,2,3,4,5}
    Q = set()
    edges = []
    weights = []
    u = random.randint(0,vine_depth-1-tr)
    Q.add(u)
    V.remove(u)
#     print('Q',Q)
#     print('V',V)
#     c = 0
    while V:
        max_v = -m.inf
        for i in Q:
#             print('u',i)
            for j in V:
#                 print('v',j)
                if tr == 0:
                    tau = np.random.uniform(-1.,1.,1)
                    if abs(tau) > max_v:
                        max_v = abs(tau)
                        u = i
                        v = j
                else: 
                    par, inx1, inx2 = parent_var(tr,ind_vine,[i,j])
#                     print('par',par)
#                     print('ind_vine prev 1',ind_vine[tr-1][i])
                    if par != None:
                        if par != ind_vine[tr-1][i][0]:
                            tau = np.random.uniform(-1.,1.,1)
                            if abs(tau) > max_v:
                                max_v = abs(tau)
                                u = i
                                v = j
                        else:
                            tau = np.random.uniform(-1.,1.,1)
                            if abs(tau) > max_v:
                                max_v = abs(tau)
                                u = i
                                v = j
        Q.add(v)
        V.remove(v)
#         print('---------')
#         if v>u:
#             edges.append([v,u])
#         else:

#         if c == 0:
#             edges.append([v,u])
#         else:
        edges.append([u,v])
        weights.append(max_v)
#         c += 1
    return edges,weights


def random_r_matrix_gen(dim):
    ind_vine = []
    for i in range(0,dim-1,1):
        ind_vine.append([])

    for tr in range(0,dim-1,1):
        ind_ee, weights = random_tree(dim,ind_vine,tr)
        ind_vine[tr] = ind_ee

    r_matrix, nodes, E = prepare_optimal(dim,ind_vine)
    return r_matrix, ind_vine, nodes, E

# File: src/DVC_tensorflow/vine_tree/tree_op.py
import math as m
import numpy as np
import random
from scipy.stats import kendalltau

###################### CHECK OPTIMAL TREE ##########################

### Prim algorithm - Section 23 Minium Spanning Three - Algorithm book

def optimal_tree(data, data_flip, ind_vine, tr, rand):
    random.seed(9001)
    V = set(range(0,data.shape[1]-tr)) #{0,1,2,3,4,5}
    Q = set()
    edges = []
    weights = []
    u = random.randint(0,data.shape[1]-1-tr)
    Q.add(u)
    V.remove(u)
    while V:
        max_v = -m.inf
        for i in Q:
            for j in V:
                if tr == 0:
                    if rand == False:
                        tau, p_value = kendalltau(data[:,i], data[:,j])
                    else:
                        tau = np.random.uniform(-1.,1.,1)
                    if abs(tau) > max_v:
                        max_v = abs(tau)
                        u = i
                        v = j
                else: 
                    par, inx1, inx2 = parent_var(tr,ind_vine,[i,j])
                    if par != None:
                        if par != ind_vine[tr-1][i][0]:
                            if rand == False:
                                tau, p_value = kendalltau(data_flip[:,i], data[:,j])
                            else:
                                tau = np.random.uniform(-1.,1.,1)
                            if abs(tau) > max_v:
                                max_v = abs(tau)
                                u = i
                                v = j
                        else:
                            if rand == False:
                                tau, p_value = kendalltau(data[:,i], data[:,j])
                            else:
                                tau = np.random.uniform(-1.,1.,1)
                            if abs(tau) > max_v:
                                max_v = abs(tau)
                                u = i
                                v = j
        Q.add(v)
        V.remove(v)
        edges.append([u,v])
        weights.append(max_v)
    return edges,weights

###################### BUILD LIST OF EDGES #########################

def build_edges(tree_index):
    n_ind = tree_index.shape[0]
    
    e_0 = {tree_index[n_ind-2,n_ind-2],tree_index[n_ind-1,n_ind-2]}
    E_1 = [e_0]
    E = [E_1]
    
    u_union = [[e_0]] 
    
    for i in range(1,n_ind-1,1):
        E.append([])
        u_union.append([])  

    n = n_ind-1
    ind_e = 0
    for i in range(n-2,-1,-1): 
        e_1 = {tree_index[i,i],tree_index[n,i]}
        
        E[0].append(e_1)
        u_union[0].append(e_1) 
        
        e_new = e_1
        u_new = e_1

        for k in range(1,n_ind-i-1,1):
            u = set()
            for uu in range(n-k,n,1):
                u.add(tree_index[uu,i])
            u.add(tree_index[n,i])
            flag = False
            for hh in range(0,len(E[k-1]),1):
                if u_union[k-1][hh].issubset(u):
                    
                    flag = True
                    e_new1 = [e_new,E[k-1][hh]]
                    E[k].append(e_new1)
                    
                    u_union1 = u_new.union(u_union[k-1][hh])
                    
                    u_union[k].append(u_union1)
                    
            if not flag:
                raise Exception('The matrix is not a regular vine')
            e_new = e_new1
            u_new = u_union1
    return E


#################### RETURN LIST ON INDEX #######################

def edges_index(E,r_matrix,tr):
    edges_ind = []
    n = r_matrix.shape[0]-1
    if tr == 0:
        for i in range(n-1,-1,-1):
            ind = [r_matrix[n,i]-1,r_matrix[i,i]-1]
            edges_ind.append(ind)
    else:
        for ii in range(0,len(E[tr]),1):
            edge = E[tr][ii]
            for yy in range(0,len(E[tr-1]),1):
                if edge[0] == E[tr-1][yy]:
                    ind0 = yy
                if edge[1] == E[tr-1][yy]:
                    ind1 = yy
            ind = [ind1,ind0]
            edges_ind.append(ind)
    return edges_ind

################### CHECK IF IS AN EDGE #######################

def isedge(edge,u):
    if type(edge) is list:
        return (isedge(edge[0],u)) & (isedge(edge[1],u))
    else:
        return edge.issubset(u)


############################################ PREPARE C-VINE AND D-VINE MATRIX ################################

def prepare_vine(vine_type, dim):
    if vine_type == 'c-vine':
        r_matrix = np.tril(np.tile(np.array(range(dim,0,-1)),(dim,1)).T)
        
        ### EDGES FOR THE CODES
        ind_vine = []
        for i in range(0,len(r_matrix)-1,1):
            ind_vine1 = []
            for j in range(1,len(r_matrix)-i,1):
                ind_vine1.append([0,j])
            ind_vine.append(ind_vine1)
            
    if vine_type == 'd-vine':
        r_matrix = np.zeros((dim,dim),np.int32)
        for i in range(0,dim,1):
            r_matrix[i,i] = dim-i #-1
        for j in range(0,dim-1,1):
            c = 1 #0
            for i in range(j+1,dim,1):
                r_matrix[i,j] = c
                c += 1
                
        ### EDGES FOR THE CODES        
        ind_vine = []
        for i in range(0,len(r_matrix)-1,1):
            ind_vine1 = []
            for j in range(0,len(r_matrix)-i-1,1):
                ind_vine1.append([j,j+1])
            ind_vine.append(ind_vine1)
        
    ### NODES

    d = len(r_matrix)
    n = d-1
    nodes = np.zeros(d,np.int32)
    for i in range(0,d,1):
        nodes[i]=r_matrix[i,i]

    nodes = np.flip(nodes)
    print('nodes:')
    print(nodes)
    
    ### EDGES of the vine
    matrix_edges = []
    c = 0
    for i in range(n,0,-1):
    #     print('level',c)
        edge1 = []
        for j in range(i-1,-1,-1):
            str1 = '(' + str(r_matrix[i,j]) + ',' + str(r_matrix[j,j])
            c = 0
            for ii in range(i+1,n+1,1):
                if c == 0:
                    str1 = str1 + '|' + str (r_matrix[ii,j])
                else:
                    str1 = str1 + ','  + str (r_matrix[ii,j])
                c += 1
            str1 = str1 + ')'
            edge1.append(str1)
        c += 1
        matrix_edges.append(edge1)
    
    print('edges:')
    for i in range(0,len(matrix_edges),1):
        print(matrix_edges[i])
    return r_matrix, ind_vine, nodes, matrix_edges
    
################  PREPARE REGULAR MATRIX ######################

def prepare_regular(r_matrix):
    E = build_edges(r_matrix)
    
    ### EDGES FOR THE CODES
    
    ind_vine = []
    for i in range(0,len(E),1):
        ind_ee = edges_index(E,r_matrix,i)
        ind_vine.append(ind_ee)
        
    ### NODES

    d = len(r_matrix)
    n = d-1
    nodes = np.zeros(d,np.int32)
    for i in range(0,d,1):
        nodes[i]=r_matrix[i,i]

    nodes = np.flip(nodes)
    print('nodes:')
    print(nodes)
    
    ### EDGES of the vine
    matrix_edges = []
    c = 0
    for i in range(n,0,-1):
    #     print('level',c)
        edge1 = []
        for j in range(i-1,-1,-1):
            str1 = '(' + str(r_matrix[i,j]) + ',' + str(r_matrix[j,j])
            c = 0
            for ii in range(i+1,n+1,1):
                if c == 0:
                    str1 = str1 + '|' + str (r_matrix[ii,j])
                else:
                    str1 = str1 + ','  + str (r_matrix[ii,j])
                c += 1
            str1 = str1 + ')'
            edge1.append(str1)
        c += 1
        matrix_edges.append(edge1)
    
    print('edges:')
    for i in range(0,len(matrix_edges),1):
        print(matrix_edges[i])
    return E, ind_vine, nodes, matrix_edges


###################### GET PARENT VARIABLE ##############################

def parent_var(k,ind_vine,edge):
    u = set()
    u.add(ind_vine[k-1][edge[0]][0])
    u.add(ind_vine[k-1][edge[0]][1])

    u1 = set()
    u1.add(ind_vine[k-1][edge[1]][0])
    u1.add(ind_vine[k-1][edge[1]][1])
    
    parent = None
    inter = u.intersection(u1)
    for elem in inter:
        parent = elem
    return parent, u, u1


######################## CHECK WHEN TO FLIP - R-MATRIX ORDER ###################################

def flip_check_all(ind_vine,tr, binning, n_bin):
    if tr < len(ind_vine)-1:
        ind_ee1 = ind_vine[tr+1]

        u_set = []
        parent = []
        for edges in ind_ee1:
            parent1, inx1, inx2 = parent_var(tr+1,ind_vine,edges)
            u_union = inx1.union(inx2)

            u_set.append(u_union)
            parent.append(parent1)
    else:
        ind_ee1 = [0,1]
        u_set = [{0,1}]
        parent = [0]

    parent_all = []
    edges_now = ind_vine[tr]
    ind_edge_rel1 = []
    flip_flag1 = []

    for j in range(0,len(edges_now),1):
        edge = edges_now[j]
        uu_now = {edge[0],edge[1]}
        parent_now = []
        parent_now_set = set()
        for jj in range(0,len(u_set),1):
            uu = u_set[jj]
            if uu_now.issubset(uu):
                if not {parent[jj]}.issubset(parent_now_set):
                    parent_now.append(parent[jj])
                    parent_now_set.add(parent[jj])
                
        # Check if they are all equal
        if len(set(parent_now)) <= 1:
            parent_now = [parent_now[0]]

        parent_all.append(parent_now)
        for par in parent_now:

            if edge[0] != par:

                flip_flag1.append(True)
            else:
                flip_flag1.append(False)
            ind_edge_rel1.append(j)
    return flip_flag1, ind_edge_rel1, parent_all


################################## PREPARE R-MATRIX OPTIMAL AND RANDOM ###################

def prepare_optimal(d, ind_vine):
    E = []
    uu_uni = []
    par = []
    diff = []
    for tr in range(0,d-1,1):
        E.append([])
        uu_uni.append([])
        par.append([])
        diff.append([])
        
    for ii in range(0,len(ind_vine[0]),1):
        E[0].append({ind_vine[0][ii][0]+1,ind_vine[0][ii][1]+1})  
    
    u_union = set()
    for tr in range(1,d-1,1):
        for ii in range(0,len(ind_vine[tr]),1):
            ind1 = ind_vine[tr][ii]
            E[tr].append([E[tr-1][ind1[0]],E[tr-1][ind1[1]]])
            if tr ==1:
                u_union = E[tr-1][ind1[0]].union(E[tr-1][ind1[1]])
                parent = E[tr-1][ind1[0]].intersection(E[tr-1][ind1[1]])
                diff1 = u_union - parent
            else:
                u_union = uu_uni[tr-1][ind1[0]].union(uu_uni[tr-1][ind1[1]])
                parent = uu_uni[tr-1][ind1[0]].intersection(uu_uni[tr-1][ind1[1]])
                diff1 = u_union - parent
            uu_uni[tr].append(u_union)
            par[tr].append(parent)
            diff[tr].append(diff1)
    
    rr = np.zeros((d,d),np.int32)
    n = len(rr)-1
    
    for tr in range(d-2,-1,-1): #0
#         print('tr',tr)
        ind_list = set()
        for j in range(0,n-tr,1):
#             print('j',j)
            edge = []
            if tr > 0:
                for ii in range(0,len(diff[tr]),1):
                    edge1 = []
                    for elem in diff[tr][ii]:
                        edge1.append(elem)
                    edge.append(edge1)
            else:
                for ii in range(0,len(E[tr]),1):
                    edge1 = []
#                     print('aa',E[tr][ii])
                    for elem in E[tr][ii]:
                        edge1.append(elem)
                    edge.append(edge1)
#             print(edge)
#             print('diff',diff[tr])
#             print('ind_list',ind_list)

            if tr == d-2:
                rr[j,j] = edge[ii][0]
                rr[n-tr,j] = edge[ii][1]

            if (tr > 0) & (tr < d-2):            
                for ii in range(0,len(diff[tr]),1):
#                     print('ii',ii)
                    if {ii}.issubset(ind_list) == False:
                        a1 = edge[ii][0]
                        a2 = edge[ii][1]
#                         print('a1',a1)
                        if j == d-2-tr:
                            rr[j,j] = a1
                        if (rr[j,j] == a1):
                            ind1 = ii
                            ind2 = 1
                            ind_list.add(ind1)
                        elif (rr[j,j] == a2):
                            ind1 = ii
                            ind2 = 0
                            ind_list.add(ind1)

                rr[n-tr,j] = edge[ind1][ind2]
            else:
                for ii in range(0,len(E[tr]),1):
                    if {ii}.issubset(ind_list) == False:
                        a1 = edge[ii][0]
                        a2 = edge[ii][1]

                        if j == d-2-tr:
                            rr[j,j] = a1

                        if (rr[j,j] == a1):
                            ind1 = ii
                            ind2 = 1
                            ind_list.add(ind1)
                        elif (rr[j,j] == a2):
                            ind1 = ii
                            ind2 = 0
                            ind_list.add(ind1)
#                 print('ind1',ind1)
#                 print('ind2',ind2)
                rr[n-tr,j] = edge[ind1][ind2]

#             print(rr)
#             print('----------')

    nodes = np.zeros(d,np.int32)
    V = set(range(1,d+1))
    for i in range(0,d,1):
        nodes[i]=rr[i,i]
        u_nod = {nodes[i]}
        if u_nod.issubset(V):
            V.remove(nodes[i])
    nodes = np.flip(nodes)

    for elem in V:
        ind = np.where(nodes == 0)
        nodes[nodes == 0] = elem
        rr[n-ind[0],n-ind[0]] = elem
    
    
    return rr, E, nodes 

####################################### RANDOM R-MATRIX  #########################################################

def random_tree(vine_depth, ind_vine, tr):
    random.seed(9001)
    V = set(range(0,vine_depth-tr)) #{0,1,2,3,4,5}
    Q = set()
    edges = []
    weights = []
    u = random.randint(0,vine_depth-1-tr)
    Q.add(u)
    V.remove(u)
#     print('Q',Q)
#     print('V',V)
#     c = 0
    while V:
        max_v = -m.inf
        for i in Q:
#             print('u',i)
            for j in V:
#                 print('v',j)
                if tr == 0:
                    tau = np.random.uniform(-1.,1.,1)
                    if abs(tau) > max_v:
                        max_v = abs(tau)
                        u = i
                        v = j
                else: 
                    par, inx1, inx2 = parent_var(tr,ind_vine,[i,j])
#                     print('par',par)
#                     print('ind_vine prev 1',ind_vine[tr-1][i])
                    if par != None:
                        if par != ind_vine[tr-1][i][0]:
                            tau = np.random.uniform(-1.,1.,1)
                            if abs(tau) > max_v:
                                max_v = abs(tau)
                                u = i
                                v = j
                        else:
                            tau = np.random.uniform(-1.,1.,1)
                            if abs(tau) > max_v:
                                max_v = abs(tau)
                                u = i
                                v = j
        Q.add(v)
        V.remove(v)
#         print('---------')
#         if v>u:
#             edges.append([v,u])
#         else:

#         if c == 0:
#             edges.append([v,u])
#         else:
        edges.append([u,v])
        weights.append(max_v)
#         c += 1
    return edges,weights


def random_r_matrix_gen(dim):
    ind_vine = []
    for i in range(0,dim-1,1):
        ind_vine.append([])

    for tr in range(0,dim-1,1):
        ind_ee, weights = random_tree(dim,ind_vine,tr)
        ind_vine[tr] = ind_ee

    r_matrix, nodes, E = prepare_optimal(dim,ind_vine)
    return r_matrix, ind_vine, nodes, E

