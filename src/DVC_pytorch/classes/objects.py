import torch
import torch.distributions as dist
import numpy as np
from utils.tensor_op import *
from utils.prob_op import *
from utils.interpolation import *
from pre_proc.transformation import *
from grid.grid_op import *
from grid.grid_class import *
from vine_tree.tree_op import *
from utils.dataset_op import create_bins
from scipy import stats
from time import perf_counter
from grid.grid_class import grid_obj
from grid.grid_op import create_grids
from pre_proc.transformation import Transform
from pre_proc.preparation import prep_cop
from utils.prob_op import kernel_cdf, kendalltau
from vine_tree.tree_op import prepare_vine, optimal_tree

# BASIC OBJECT CLASSES

class copula_obj(object):
    """Copula object."""
    
    def __init__(self, opt_bw):
        """Create a copula object.
        Args:
            opt_bw: Optimal fitted bandwidth.
        """
        self.opt_bw = opt_bw
        self.pd_grid_uv = None
        self.cdf = None

class cop_par_obj(object):
    """Parametric copula object."""
    
    def __init__(self, family, theta):
        """Create a parametric copula object.
        Args:
            family: Type of the copula family.
            theta: Parameter(s) of the copula.
        """
        self.family = family
        self.theta = theta

class margin_obj(object):
    """Marginal object."""
    
    def __init__(self, dist, theta, is_cont):
        """Create a marginal object.
        Args:
            dist: Distribution type of the marginal.
            theta: Parameters of the marginal.
            is_cont: Whether the marginal is continuous.
        """
        self.dist = dist
        self.theta = theta
        self.is_cont = is_cont
        self.ker = None

class vine_obj_bin(object):
    """Vine object with binning support."""
    
    def __init__(self, vine_family, families, vine_depth, margin, knots, method, *args):
        """Create a vine object.
        Args:
            vine_family: Type of vine (r-vine, c-vine, d-vine).
            families: Copula families.
            vine_depth: Depth of the vine.
            margin: Margins of the copula.
            knots: Number of knots for grid.
            method: Method for vine construction.
            *args: Additional arguments (e.g., r_matrix for r-vine).
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
        for i in range(0, self.n_cop-1, 1):
            self.ind_vine.append([])
        
        if self.vine_family == 'r-vine':
            if self.method == 'matrix':
                self.r_matrix = args[0]
                self.E, self.ind_vine, self.nodes, self.matrix_edges = prepare_regular(self.r_matrix)
            elif self.method == 'random':
                self.r_matrix, _, _, _ = random_r_matrix_gen(self.n_cop)
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
    
    def select_batch_size_cdf(self, x):
        """Select appropriate batch size for CDF computation based on data size"""
        batch_size = 1
        data_size = x.shape[0]
        
        if data_size > 5000:
            batch_size = 10
        elif data_size > 10000:
            batch_size = 100
        elif data_size > 50000:
            batch_size = 200
        elif data_size > 100000:
            batch_size = 500
        elif data_size > 200000:
            batch_size = 1000
        elif data_size > 500000:
            batch_size = 2000
            
        return torch.tensor(batch_size, dtype=torch.int32)
    
    def select_batch_size(self, data):
        """Select appropriate batch size based on data size"""
        batch_size = 1
        data_size = data.shape[0]
        
        if data_size > 2000:
            batch_size = 5
        elif data_size > 10000:
            batch_size = 10
        elif data_size > 50000:
            batch_size = 20
        elif data_size > 100000:
            batch_size = 50
        elif data_size > 200000:
            batch_size = 100
        elif data_size > 500000:
            batch_size = 200
            
        return torch.tensor(batch_size, dtype=torch.int32)
    
    def fit(self, x, gen_dict, npc_dict, par_dict, bin_dict):
        """
        Fit vine copula to data
        
        Args:
            x: Input data tensor
            gen_dict: General parameters dictionary
            npc_dict: Non-parametric copula parameters
            par_dict: Parametric copula parameters
            bin_dict: Binning parameters
            
        Returns:
            Fitted vine copula object
        """
        # Initialization
        self.cdf_data = False
        self.fitted = gen_dict['fitted']
        
        # Set device
        device = x.device if torch.is_tensor(x) else torch.device('cpu')
        dtype = x.dtype if torch.is_tensor(x) else torch.float32
        
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=dtype, device=device)
        
        self.binning = gen_dict['binning']
        self.parallel = gen_dict['parallel']
        self.param = gen_dict['param']
        self.vine_depth = gen_dict['vine_depth'] - 1
        
        # Get data dimension
        d = x.shape[1]
        
        if not self.param:
            self.opt_method = npc_dict['opt_method']
            self.batch_paral = npc_dict['batch_paral']
        else:
            param_families = par_dict['param_families']
            
        if self.binning:
            self.n_bin = bin_dict['n_bin']
        
        # Create grids if not already created
        if self.grid_u is None:
            self.grid_u, self.grid_s, self.grid_x = create_grids(self.knots, device=device, dtype=dtype)
        
        # Create margins and compute CDF
        self.Mar_G = []
        q = x.clone()
        
        for i in range(x.shape[1]):
            if self.margin[i].is_cont:
                if hasattr(self.margin[i], 'ker') and self.margin[i].ker is not None:
                    ccc = torch.tensor(self.margin[i].ker, device=device, dtype=dtype)
                    interp_cdf, mar_s1, mar_p1 = kernel_cdf(ccc, ccc, self.grid_u.ex)
                else:
                    interp_cdf, mar_s1, mar_p1 = kernel_cdf(q[:, i], q[:, i], self.grid_u.ex)
                self.Mar_G.append([mar_s1, mar_p1])
            else:
                raise NotImplementedError("Discrete margins not yet implemented")
        
        # Initialize theta arrays for conditional CDFs (h-functions)
        n_samples = x.shape[0]
        self.theta = torch.zeros((n_samples, d, d), dtype=dtype, device=device)
        self.theta_flip = torch.zeros((n_samples, d, d), dtype=dtype, device=device)
        
        # Transform data to uniform space and set first layer of theta
        for i in range(d):
            if hasattr(self.margin[i], 'ker') and self.margin[i].ker is not None:
                ccc = torch.tensor(self.margin[i].ker, device=device, dtype=dtype)
                interp_cdf, _, _ = kernel_cdf(ccc, ccc, self.grid_u.ex)
                self.theta[:, 0, i] = interp_cdf
            else:
                interp_cdf, _, _ = kernel_cdf(x[:, i], q[:, i], self.grid_u.ex)
                self.theta[:, 0, i] = interp_cdf
        
        # Only proceed with fitting if not already fitted
        if self.fitted == False:
            # Prepare data for copula fitting
            self.x = x
            
            # Update vine structure based on data dimension if necessary
            if self.vine_family == 'r-vine':
                if self.method == 'optimal':
                    self.nodes = torch.arange(1, d + 1)
                    self.r_matrix = torch.zeros((d, d), dtype=torch.long)
                    self.ind_vine = []
                    # Initialize structure
                    for i in range(d):
                        self.r_matrix[i, i] = i + 1
                        if i < d - 1:
                            self.ind_vine.append([])
                elif self.method == 'matrix':
                    # Use provided R-matrix
                    pass
                elif self.method == 'random':
                    # Generate random R-matrix
                    from vine_tree.tree_op import random_r_matrix_gen
                    self.r_matrix, _, _, _ = random_r_matrix_gen(d)
                    self.E, self.ind_vine, self.nodes, self.matrix_edges = prepare_regular(self.r_matrix)
            elif self.vine_family == 'c-vine':
                self.r_matrix, self.ind_vine, self.nodes, _ = prepare_vine('c-vine', d)
            elif self.vine_family == 'd-vine':
                self.r_matrix, self.ind_vine, self.nodes, _ = prepare_vine('d-vine', d)
            
            # Initialize copula storage
            self.copulas = []
        
        # Always initialize these for correlation tracking
        self.correlations = []
        self.correlations_bins = []
        self.flip_flag = []
        self.ind_edge_rel = []
        
        # Fit copulas tree by tree
        for tr in range(min(d - 1, self.vine_depth + 1)):
            print(f'Fitting tree {tr}...')
            
            # Number of copulas in this tree level
            n_cop = d - 1 - tr
            
            # For fitted == True case, set vine family and method
            if self.fitted == True:
                self.vine_family = 'r-vine'
                self.method = 'matrix'
            
            # Handle different vine structures
            if self.vine_family == 'r-vine':
                if self.method == 'matrix' or self.method == 'random':
                    # For matrix method, ensure ind_vine is properly initialized
                    if tr < len(self.ind_vine):
                        edges_now = self.ind_vine[tr]
                    else:
                        edges_now = []
                elif self.method == 'optimal':
                    # Build tree structure dynamically
                    if tr == 0:
                        # First tree: optimal edges based on Kendall's tau
                        from vine_tree.tree_op import optimal_tree
                        edges, weights = optimal_tree(self.theta[:, tr, :], None, [], tr, rand=False)
                        self.ind_vine[tr] = edges
                        edges_now = edges
                        
                        # Update R-matrix
                        n = len(self.r_matrix) - 1
                        for j in range(len(edges)):
                            edg = edges[len(edges)-1-j]
                            self.r_matrix[n, j] = edg[0] + 1
                            self.r_matrix[j, j] = edg[1] + 1
                        
                        # Update nodes
                        self.nodes = torch.zeros(d, dtype=torch.long)
                        V = set(range(1, d+1))
                        for i in range(d):
                            self.nodes[i] = self.r_matrix[i, i]
                            u_nod = {int(self.nodes[i])}
                            if u_nod.issubset(V):
                                V.remove(int(self.nodes[i]))
                        self.nodes = torch.flip(self.nodes, [0])
                        
                        for elem in V:
                            ind = torch.where(self.nodes == 0)[0]
                            if len(ind) > 0:
                                self.nodes[ind[0]] = elem
                                self.r_matrix[n-ind[0], n-ind[0]] = elem
                    else:
                        # Higher trees: use h-functions from previous tree
                        from vine_tree.tree_op import optimal_tree
                        edges, weights = optimal_tree(self.theta[:, tr, :], 
                                                      self.theta_flip[:, tr, :], 
                                                      self.ind_vine, tr, rand=False)
                        self.ind_vine[tr] = edges
                        edges_now = edges
            else:
                edges_now = self.ind_vine[tr] if tr < len(self.ind_vine) else []
            
            if len(edges_now) == 0:
                continue
            
            # Prepare data for this tree level
            self.data_u = torch.zeros((n_samples, 2, len(edges_now)), dtype=dtype, device=device)
            
            for j in range(len(edges_now)):
                edge = edges_now[j]
                
                if tr == 0:
                    # First tree: use original data
                    self.data_u[:, :, j] = torch.stack([self.theta[:, tr, edge[0]], 
                                                         self.theta[:, tr, edge[1]]], dim=1)
                else:
                    # Higher trees: check if we need flipped values
                    from vine_tree.tree_op import parent_var
                    parent, inx1, inx2 = parent_var(tr, self.ind_vine, edge)
                    
                    if self.ind_vine[tr-1][edge[0]][0] != parent:
                        self.data_u[:, :, j] = torch.stack([self.theta_flip[:, tr, edge[0]], 
                                                            self.theta[:, tr, edge[1]]], dim=1)
                    else:
                        self.data_u[:, :, j] = torch.stack([self.theta[:, tr, edge[0]], 
                                                            self.theta[:, tr, edge[1]]], dim=1)
            
            # Transform data for this tree
            trans = Transform(len(edges_now))
            self.data_s = trans.forward_u(self.data_u)
            self.data_x = trans.forward_s(self.data_s)
            grid_x = trans.forward_s(self.grid_s.ex)
            
            # Only fit if not already fitted
            if self.fitted == False:
                
                # Check if this tree is beyond vine_depth (use independence copulas)
                if tr > self.vine_depth:
                    # Independence copulas for trees beyond vine_depth
                    if self.parallel:
                        # Parallel parametric fitting for independence
                        families = ["ind"]
                        from optim.vine_fit import parametric_fit
                        aic, theta_par, logp = parametric_fit(self.data_u, families, len(edges_now))
                        
                        par_copulas = []
                        tau_values = []
                        for i in range(len(edges_now)):
                            tau, p_value = kendalltau(self.data_u[:, 0, i], self.data_u[:, 1, i])
                            tau_values.append(tau)
                            
                            cop_p = cop_par_obj('ind', [])
                            par_copulas.append(cop_p)
                        
                        self.copulas.append(par_copulas)
                        self.correlations.append(tau_values)
                    else:
                        # Non-parallel independence copulas
                        if self.binning:
                            # Binning logic for independence copulas
                            # TODO: Implement binning for independence copulas
                            pass
                        else:
                            par_copulas = []
                            tau_values = []
                            for j in range(len(edges_now)):
                                tau, p_value = kendalltau(self.data_u[:, 0, j], self.data_u[:, 1, j])
                                tau_values.append(tau)
                                cop_p = cop_par_obj('ind', [])
                                par_copulas.append(cop_p)
                            
                            self.copulas.append(par_copulas)
                            self.correlations.append(tau_values)
                
                else:  # tr <= self.vine_depth
                    # Regular copula fitting
                    if tr == 0 or self.binning == False:
                        # No binning for first tree or when binning is disabled
                        
                        if self.parallel == False:
                            if self.param == True:
                                # Non-parallel parametric fitting
                                par_copulas = []
                                tau_values = []
                                
                                for j in range(len(edges_now)):
                                    tau, p_value = kendalltau(self.data_u[:, 0, j], self.data_u[:, 1, j])
                                    tau_values.append(tau)
                                    
                                    from optim.vine_fit import parametric_fit
                                    aic, theta_par, logp = parametric_fit(
                                        self.data_u[:, :, j].unsqueeze(-1), param_families, 1)
                                    
                                    ind_fam = np.argmin(aic)
                                    family = param_families[ind_fam]
                                    theta_est = theta_par[0][ind_fam]
                                    
                                    cop_p = cop_par_obj(family, theta_est)
                                    par_copulas.append(cop_p)
                                
                                self.copulas.append(par_copulas)
                                self.correlations.append(tau_values)
                            else:
                                # Non-parallel non-parametric fitting
                                from optim.bandwidth import bandwidth_mul
                                from optim.vine_fit import optimization
                                
                                # Compute bandwidth
                                bw = bandwidth_mul(self.data_x, 2, len(edges_now))
                                
                                # Optimize bandwidth multipliers
                                batch_size = self.select_batch_size(self.data_s)
                                
                                opt_bw_array = []
                                tau_values = []
                                
                                for i in range(len(edges_now)):
                                    tau, p_value = kendalltau(self.data_u[:, 0, i], self.data_u[:, 1, i])
                                    tau_values.append(tau)
                                    
                                    grid_dict = {'grid_u': self.grid_u, 'grid_s': self.grid_s, 
                                               'grid_x': grid_x[:, :, i]}
                                    data_dict = {'data_s': self.data_s[:, :, i], 
                                               'data_x': self.data_x[:, :, i]}
                                    par_dict = {'n_cop': 1, 'batch': batch_size, 
                                               'max_iter': [70, 100], 'lr': [0.1, 0.03],
                                               'conv_tol': [1e-5, 5e-5], 'opt_method': self.opt_method}
                                    
                                    opt = optimization(grid_dict, data_dict, par_dict)
                                    opt_bw_array.append(opt)
                                
                                opt_bw = torch.stack(opt_bw_array)
                                bw_opt = opt_bw.squeeze() * bw
                                
                                # Check constraints on bandwidth
                                bw_opt = torch.clamp(bw_opt, min=1e-2, max=2.0)
                                
                                copula = copula_obj(bw_opt.numpy())
                                self.copulas.append(copula)
                                self.correlations.append(tau_values)
                        
                        else:  # self.parallel == True
                            if self.param == True:
                                # Parallel parametric fitting
                                from optim.vine_fit import parametric_fit
                                aic, theta_par, logp = parametric_fit(self.data_u, param_families, len(edges_now))
                                
                                par_copulas = []
                                tau_values = []
                                
                                for i in range(len(edges_now)):
                                    tau, p_value = kendalltau(self.data_u[:, 0, i], self.data_u[:, 1, i])
                                    tau_values.append(tau)
                                    
                                    ind_fam = np.argmin(aic[i])
                                    family = param_families[ind_fam]
                                    theta_est = theta_par[i][ind_fam]
                                    
                                    cop_p = cop_par_obj(family, theta_est)
                                    par_copulas.append(cop_p)
                                
                                self.copulas.append(par_copulas)
                                self.correlations.append(tau_values)
                            else:
                                # Parallel non-parametric fitting
                                # TODO: Implement parallel non-parametric fitting
                                pass
                    
                    else:  # self.binning == True and tr > 0
                        # Binning logic for higher trees
                        # TODO: Implement full binning logic
                        pass
            
            # Determine flip flags and edge relationships
            from vine_tree.tree_op import flip_check_all
            flip_flag1, ind_edge_rel1, parent_all = flip_check_all(self.ind_vine, tr, 
                                                                   self.binning, self.n_bin)
            
            self.flip_flag.append(flip_flag1)
            self.ind_edge_rel.append(ind_edge_rel1)
            
            # Update theta values (h-functions) for next tree
            if tr < d - 2 and len(edges_now) > 0:  # Don't need h-functions for last tree
                
                n_eval = len(ind_edge_rel1)
                
                if self.fitted == False and tr <= self.vine_depth:
                    # Compute h-functions from fitted copulas
                    if self.param:
                        # Parametric h-functions
                        from param.cond_copula import copulaccdf_torch
                        
                        for j in range(n_eval):
                            ind_edge = ind_edge_rel1[j]
                            if ind_edge < len(self.copulas[tr]):
                                cop_p = self.copulas[tr][ind_edge]
                                edge = edges_now[ind_edge]
                                
                                # Get the data for this edge
                                if tr == 0:
                                    vv = torch.stack([self.theta[:, tr, edge[0]], 
                                                     self.theta[:, tr, edge[1]]], dim=1)
                                else:
                                    parent, inx1, inx2 = parent_var(tr, self.ind_vine, edge)
                                    if self.ind_vine[tr-1][edge[0]][0] != parent:
                                        vv = torch.stack([self.theta_flip[:, tr, edge[0]], 
                                                         self.theta[:, tr, edge[1]]], dim=1)
                                    else:
                                        vv = torch.stack([self.theta[:, tr, edge[0]], 
                                                         self.theta[:, tr, edge[1]]], dim=1)
                                
                                # Apply flipping if needed
                                if flip_flag1[j]:
                                    vv = torch.flip(vv, dims=[1])
                                    h_val = copulaccdf_torch(cop_p, vv)
                                    self.theta_flip[:, tr + 1, ind_edge] = h_val.squeeze()
                                else:
                                    h_val = copulaccdf_torch(cop_p, vv)
                                    self.theta[:, tr + 1, ind_edge] = h_val.squeeze()
                    else:
                        # Non-parametric h-functions
                        # TODO: Implement non-parametric h-function computation
                        pass
                
                elif tr > self.vine_depth:
                    # Independence copulas - h-function is just the first component
                    for j in range(len(edges_now)):
                        edge = edges_now[j]
                        if flip_flag1[j]:
                            self.theta_flip[:, tr + 1, j] = self.data_u[:, 0, j]
                        else:
                            self.theta[:, tr + 1, j] = self.data_u[:, 0, j]
        
        # Store final R-matrix for optimal R-vine
        if self.vine_family == 'r-vine' and self.method == 'optimal' and self.fitted == False:
            from vine_tree.tree_op import prepare_optimal
            self.r_matrix, self.E, self.nodes = prepare_optimal(d, self.ind_vine)
        
        print("Vine copula fitting completed!")
        return self
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample from fitted vine copula
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Samples in original data space
        """
        if not hasattr(self, 'copulas') or self.copulas is None:
            raise ValueError("Vine must be fitted before sampling")
            
        # Import the corrected sampler with index remapping fix
        from sampling.vine_sampler import VineSampler
        sampler = VineSampler(self)
        samples, _ = sampler.sample(n_samples)
        
        return samples
    
    def evaluation(self, points):
        """
        Evaluate vine copula log-likelihood on new points
        
        Args:
            points: Points to evaluate at
            
        Returns:
            p: Total probability density
            p_copula: Copula probability density  
            plog: Log probability density
        """
        if not hasattr(self, 'copulas') or self.copulas is None or len(self.copulas) == 0:
            raise ValueError("Model must be fitted before evaluation")
        
        device = points.device if torch.is_tensor(points) else torch.device('cpu')
        dtype = points.dtype if torch.is_tensor(points) else torch.float32
        
        if not torch.is_tensor(points):
            points = torch.tensor(points, dtype=dtype, device=device)
        
        n_points = points.shape[0]
        d = len(self.margin) if hasattr(self, 'margin') else self.n_cop
        
        # Initialize arrays for h-functions
        self.Fp = torch.zeros((n_points, d, d), dtype=dtype, device=device)
        self.Fp_flip = torch.zeros((n_points, d, d), dtype=dtype, device=device)
        
        # Initialize log densities
        logf = torch.zeros((n_points, d, d), dtype=dtype, device=device)
        self.logf_flip = torch.zeros_like(points)
        
        # First layer: marginal transformations and densities
        for i in range(d):
            # Transform to uniform using empirical CDF
            interp_cdf_poi, _, _ = kernel_cdf(self.x[:, i], points[:, i], self.grid_u.ex)
            self.Fp[:, 0, i] = interp_cdf_poi
            
            # Compute marginal densities
            den1, mden1 = kernel_pdf2(self.x[:, i])
            
            # Interpolate density at points
            from utils.interpolation import interp1d_np
            inter = interp1d_np(points[:, i], mden1, den1)
            
            # Log marginal density - add numerical stability
            inter = torch.where(inter <= 0, torch.tensor(1e-10, dtype=inter.dtype, device=inter.device), inter)
            logf[:, 0, i] = torch.log(inter)
        
        self.logf = logf
        logf_marginal = self.logf.clone()
        
        # Transform points for evaluation
        self.points_u = self.Fp[:, 0, :].clone()
        
        # Traverse the vine structure
        for tr in range(min(d - 1, self.vine_depth + 1)):
            
            # Get edges for this tree
            if tr < len(self.ind_vine):
                edges_now = self.ind_vine[tr]
                n_eval = len(self.ind_edge_rel[tr]) if tr < len(self.ind_edge_rel) else len(edges_now)
                
                # Prepare data for this tree level
                vv_u = torch.zeros((n_points, 2, n_eval), dtype=dtype, device=device)
                
                for j in range(n_eval):
                    if tr < len(self.ind_edge_rel) and j < len(self.ind_edge_rel[tr]):
                        edge = edges_now[self.ind_edge_rel[tr][j]]
                    else:
                        edge = edges_now[j] if j < len(edges_now) else [0, 1]
                    
                    if tr == 0:
                        # First tree: use marginal CDFs
                        vv_u[:, :, j] = torch.stack([self.Fp[:, tr, edge[0]], 
                                                     self.Fp[:, tr, edge[1]]], dim=1)
                    else:
                        # Higher trees: use h-functions
                        from vine_tree.tree_op import parent_var
                        parent, inx1, inx2 = parent_var(tr, self.ind_vine, edge)
                        
                        if self.ind_vine[tr-1][edge[0]][0] != parent:
                            vv_u[:, :, j] = torch.stack([self.Fp_flip[:, tr, edge[0]], 
                                                        self.Fp[:, tr, edge[1]]], dim=1)
                        else:
                            vv_u[:, :, j] = torch.stack([self.Fp[:, tr, edge[0]], 
                                                        self.Fp[:, tr, edge[1]]], dim=1)
                    
                    # Apply flipping if needed
                    if not self.param and tr < len(self.flip_flag) and j < len(self.flip_flag[tr]) and self.flip_flag[tr][j]:
                        vv_u[:, :, j] = torch.flip(vv_u[:, :, j], dims=[1])
                
                # Evaluate copulas for this tree
                for j in range(n_eval):
                    if tr < len(self.ind_edge_rel) and j < len(self.ind_edge_rel[tr]):
                        ind_edge = self.ind_edge_rel[tr][j]
                    else:
                        ind_edge = j
                    
                    if self.param:
                        # Parametric copula evaluation
                        if tr < len(self.copulas) and ind_edge < len(self.copulas[tr]):
                            cop_p = self.copulas[tr][ind_edge]
                            
                            # Compute PDF
                            from param.cond_copula import copulapdf
                            vv = vv_u[:, :, j].unsqueeze(-1)
                            pd_points = torch.squeeze(torch.tensor(
                                copulapdf(cop_p, vv.cpu().numpy()), 
                                device=device, dtype=dtype))
                            
                            # Update log density
                            pd_points = torch.where(pd_points <= 0, torch.tensor(1e-10, dtype=pd_points.dtype, device=pd_points.device), pd_points)
                            self.logf[:, ind_edge, tr + 1] = torch.log(pd_points)
                            
                            # Compute h-functions for next tree
                            if tr < d - 2:
                                from param.cond_copula import copulaccdf
                                if tr < len(self.flip_flag) and j < len(self.flip_flag[tr]) and self.flip_flag[tr][j]:
                                    h_val = torch.squeeze(torch.tensor(
                                        copulaccdf(cop_p, vv_u[:, :, j].cpu().numpy()),
                                        device=device, dtype=dtype))
                                    h_val = torch.clamp(h_val, 1e-7, 1 - 1e-7)
                                    self.Fp_flip[:, tr + 1, ind_edge] = h_val
                                else:
                                    vv_flip = torch.flip(vv_u[:, :, j], dims=[1])
                                    h_val = torch.squeeze(torch.tensor(
                                        copulaccdf(cop_p, vv_flip.cpu().numpy()),
                                        device=device, dtype=dtype))
                                    h_val = torch.clamp(h_val, 1e-7, 1 - 1e-7)
                                    self.Fp[:, tr + 1, ind_edge] = h_val
                    else:
                        # Non-parametric copula evaluation  
                        # Transform to s-space
                        trans = Transform(1)
                        vv_s = trans.forward_u(vv_u[:, :, j:j+1])
                        
                        # Get copula object
                        copula = self.copulas[tr]
                        
                        # Evaluate PDF on grid (this would need the full grid evaluation)
                        # For now, use nearest neighbor interpolation
                        if hasattr(copula, 'pd_grid_uv') and copula.pd_grid_uv is not None:
                            from utils.interpolation import nearestInterp2d
                            s_ax1, s_ax2 = self.grid_s.axis()
                            pd_points = nearestInterp2d(vv_s[:, :, 0], s_ax1, s_ax2, 
                                                      copula.pd_grid_uv[:, :, ind_edge])
                        else:
                            # Fallback to independence
                            pd_points = torch.ones(n_points, dtype=dtype, device=device)
                        
                        # Update log density
                        pd_points = torch.where(pd_points <= 0, torch.tensor(1e-10, dtype=pd_points.dtype, device=pd_points.device), pd_points)
                        self.logf[:, ind_edge, tr + 1] = torch.log(pd_points)
                        
                        # For h-functions, would need CDF evaluation
                        # For now, use uniform
                        if tr < d - 2:
                            if tr < len(self.flip_flag) and j < len(self.flip_flag[tr]) and self.flip_flag[tr][j]:
                                self.Fp_flip[:, tr + 1, ind_edge] = vv_u[:, 1, j]
                            else:
                                self.Fp[:, tr + 1, ind_edge] = vv_u[:, 0, j]
        
        # Sum log densities across all trees
        log_marginal = torch.sum(self.logf[:, :, 0], dim=1)
        log_copula = torch.sum(torch.sum(self.logf[:, :, 1:], dim=2), dim=1)
        log_total = log_marginal + log_copula
        
        # Compute probability densities
        p_marginal = torch.exp(log_marginal)
        p_copula = torch.exp(log_copula)
        p = torch.exp(log_total)
        
        return p, p_copula, log_total 