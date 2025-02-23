# src/classes/objects.py
import torch

class margin_obj:
    """
    Object to hold margin distribution information.
    """
    def __init__(self, dist: str, theta, is_cont: bool):
        self.dist = dist
        self.theta = theta
        self.is_cont = is_cont
        self.ker = None

class cop_par_obj:
    """
    Object to hold copula parameters.
    """
    def __init__(self, family: str, theta):
        self.family = family
        self.theta = theta

class vine_obj_bin:
    """
    Vine object (binary version) that stores vine structure, margins, fitted copulas,
    and intermediate cumulative distributions (theta).
    """
    def __init__(self, vine_family: str, families, vine_depth: int, margin, knots: int, method: str, r_matrix=None):
        self.vine_family = vine_family
        self.families = families
        self.n_cop = vine_depth
        self.margin = margin
        self.knots = knots
        self.method = method
        self.r_matrix = r_matrix
        # For simplicity, we use trivial vine structure.
        self.ind_vine = []
        for i in range(vine_depth - 1):
            self.ind_vine.append([(0, i+1)])
        self.nodes = torch.arange(1, vine_depth+1)
        self.matrix_edges = []
        self.copulas = []  # List of copula fits for each tree level.
        self.theta = None  # Will be set during fitting.
        self.theta_flip = None

    def fit(self, x: torch.Tensor, gen_dict: dict, npc_dict: dict, par_dict: dict, bin_dict: dict):
        """
        Fit the vine to data x.
        1. Compute margin CDFs for each variable.
        2. For each tree level tr, for each edge, fit a copula (parametric or nonparametric)
           and update the cumulative distribution (theta) for the next tree level.
        """
        device = x.device
        N, d = x.shape
        # Initialize theta tensor: shape [N, d, d]
        self.theta = torch.zeros((N, d, d), dtype=x.dtype, device=device)
        self.theta_flip = torch.zeros_like(self.theta)
        # Compute margin CDFs:
        for i in range(d):
            loc, scale = self.margin[i].theta
            dist = torch.distributions.Normal(loc, scale)
            self.theta[:, 0, i] = dist.cdf(x[:, i])
        # For each tree level (tr from 0 to d-2):
        for tr in range(d - 1):
            n_edges = d - 1 - tr
            data_u = torch.zeros((N, 2, n_edges), dtype=x.dtype, device=device)
            edges = []
            # Here, we assume a trivial vine: edge j is (tr, tr+j+1)
            for j in range(n_edges):
                data_u[:, 0, j] = self.theta[:, tr, tr]
                data_u[:, 1, j] = self.theta[:, tr, tr+j+1]
                edges.append((tr, tr+j+1))
            if not gen_dict['param']:
                # Nonparametric fitting using local likelihood:
                bw = __import__('param.local_lik', fromlist=['local_likelihood_fit']).local_likelihood_fit(data_u, n_edges)
                self.copulas.append({'type': 'nonparam', 'bw': bw})
                # Create a grid for evaluation:
                K = self.knots
                grid_vals = torch.linspace(-3.2, 3.2, K, device=device, dtype=x.dtype)
                grid = torch.zeros((K, 2, n_edges), device=device, dtype=x.dtype)
                for j in range(n_edges):
                    grid[:, 0, j] = grid_vals
                    grid[:, 1, j] = grid_vals
                # Evaluate kernel density over grid and compute approximate CDF by cumulative sum:
                for j in range(n_edges):
                    ker_sum = __import__('param.local_lik', fromlist=['loclik_batch_eval']).loclik_batch_eval(
                        bw, data_u[:, :, j].unsqueeze(-1), grid[:, :, j].unsqueeze(-1), 1, batch_size=10
                    )
                    ker_sum = ker_sum.squeeze(-1)
                    cdf_grid = torch.cumsum(ker_sum, dim=0)
                    self.theta[:, tr+1, edges[j][1]] = cdf_grid[-1].expand(N)
            else:
                # Parametric fitting:
                families = par_dict['param_families']
                aic_list, theta_list, logp_list = __import__('param.parametric_fit', fromlist=['parametric_fit']).parametric_fit(data_u, families, n_edges)
                best_fams = []
                best_thetas = []
                for j in range(n_edges):
                    best_aic = np.inf
                    bestF = None
                    bestTh = None
                    for idx, fam in enumerate(families):
                        aic_val = aic_list[idx][j]
                        if aic_val < best_aic:
                            best_aic = aic_val
                            bestF = fam
                            bestTh = theta_list[idx][j]
                    best_fams.append(bestF)
                    best_thetas.append(bestTh)
                    c_vals = torch.zeros(N, device=device, dtype=x.dtype)
                    for m in range(N):
                        uv = data_u[m, :, j].unsqueeze(0)
                        c_vals[m] = __import__('param.cond_copula', fromlist=['copulaccdf']).copulaccdf(bestF, bestTh, uv)
                    self.theta[:, tr+1, edges[j][1]] = c_vals
                self.copulas.append({'type': 'param', 'families': best_fams, 'theta': best_thetas})
        print("[vine_obj_bin.fit] Completed fitting.")


    def evaluation(self, points: torch.Tensor):
        """
        Evaluate the joint density at new points.
        Compute the product of margin densities and add the copula density contributions
        from each vine tree level.
        """
        N, d = points.shape
        device = points.device
        p = torch.ones(N, device=device, dtype=points.dtype)
        logf = torch.zeros(N, device=device, dtype=points.dtype)
        margin_cdf = torch.zeros(N, d, device=device, dtype=points.dtype)
        for i in range(d):
            loc, scale = self.margin[i].theta
            dist = torch.distributions.Normal(loc, scale)
            pdf_i = torch.exp(dist.log_prob(points[:, i]))
            cdf_i = dist.cdf(points[:, i])
            p *= pdf_i
            logf += torch.log(pdf_i + 1e-16)
            margin_cdf[:, i] = cdf_i
        # For each vine tree level, add copula contributions.
        # In our trivial vine, for each level tr (0 <= tr <= d-2), we assume one edge per level.
        for tr in range(0, d - 1):
            # In our trivial vine, edge index is: (tr, tr+1)
            edge_idx = tr + 1
            uv = torch.stack([margin_cdf[:, tr], margin_cdf[:, edge_idx]], dim=1)
            # Assume a parametric branch was used at tree level tr.
            cop_info = self.copulas[tr]
            if cop_info['type'] == 'param':
                fam = cop_info['families'][0]  # In our trivial vine, one edge per tree.
                th = cop_info['theta'][0]
                cop_density = __import__('param.cond_copula', fromlist=['copulapdf']).copulapdf(fam, th, uv)
                logf += torch.log(cop_density + 1e-16)
            else:
                # For nonparam branch, use a default of 0 contribution.
                logf += 0.0
        p_cop = torch.exp(logf)
        return p, p_cop, logf


    def sample(self, shape: tuple):
        """
        Generate samples from the fitted vine.
        For a full vine, this involves sequential inversion.
        Here we use a simple approach:
          - Sample independent uniforms,
          - Transform via inverse margins.
        """
        n_samples = shape[0]
        d = self.n_cop
        device = self.margin[0].ker.device if self.margin[0].ker is not None else torch.device('cpu')
        u = torch.rand(n_samples, d, device=device)
        samples = torch.zeros_like(u)
        for i in range(d):
            loc, scale = self.margin[i].theta
            dist = torch.distributions.Normal(loc, scale)
            samples[:, i] = dist.icdf(u[:, i])
        return samples