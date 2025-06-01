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
        
        # Get optimization method from gen_dict if available
        optimization_method = gen_dict.get('optimization_method', 'tau')
        
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
                        ind_ee, weights = optimal_tree(self.theta[:,tr,:],self.theta_flip[:,tr,:],self.ind_vine,tr,random,optimization_method)
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

                        ind_ee, weights = optimal_tree(self.theta[:,tr,:],self.theta_flip[:,tr,:],self.ind_vine,tr,random,optimization_method)
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
        
        # Get optimization method from gen_dict if available
        optimization_method = gen_dict.get('optimization_method', 'tau')
        
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
            
            data_u = np.empty([self.theta.shape[0],2,n_cop],x.dtype)  
            
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
            
            if self.parallel == True:
                n_cop1 = tf.constant(n_cop,tf.int32)

                grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex}
                data_dict = {'data_s':self.data_s, 'data_x':self.data_x}
                par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,100], 'lr':[0.1, 0.01], 
                                'conv_tol': [0.000001,0.0000001], 'opt_method': self.opt_method}

                opt = optimization(grid_dict, data_dict, par_dict)
                opt_bw = opt            
            elif self.parallel == False:
                opt_bw = tf.TensorArray(x.dtype,size=n_cop)
            
                for i in range(0,n_cop,1):
        #            print('col:',i)

                    n_cop1 = tf.constant(1,tf.int32)

                    grid_dict = {'grid_u':self.grid_u, 'grid_s':self.grid_s, 'grid_x':self.grid_x.ex[:,:,i]}
                    data_dict = {'data_s':self.data_s[:,:,i], 'data_x':self.data_x[:,:,i]}
                    par_dict = {'n_cop':n_cop1, 'batch':tf.constant(2,tf.int32), 'max_iter': [70,100], 'lr':[0.1, 0.01], 
                                'conv_tol': [0.000001,0.0000001], 'opt_method': self.opt_method}
    
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