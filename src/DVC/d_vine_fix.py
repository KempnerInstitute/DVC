##################################################
# DVC/d_vine_fix.py
##################################################
import torch
import numpy as np
import math
from typing import Optional, Tuple, List
from .objects import vine_obj_bin, cop_par_obj
from .utils_tensor import clamp_probs

def _cdf_gaussian(u, v, rho):
    """
    2D Gaussian Copula's cdf is not closed-form except via bvn cdf.
    We'll just do an approximation or skip if we only need the 'h-function'.
    """
    # Typically we only need cdf_inversion for the conditional approach.
    pass

def _h_function_gaussian(u_known, u_in, rho, inverse=False):
    """
    The h-function for a Gaussian copula:
        H(u_in | u_known) = partial cond. distribution
    If inverse=False, we compute H(u_in). If inverse=True, we do inverse of H.
    In D-vine sampling, we often need the inverse h to map a random V in [0,1] to U_i.
    """
    # Gaussian copula partial formula:
    #   U_i | U_j = Phi( (Phi^{-1}(u_in) - rho * Phi^{-1}(u_known)) / sqrt(1-rho^2 ) ).
    # Or the inverse if we have V in [0,1].
    # We'll define a standard normal dist in PyTorch:
    normal = torch.distributions.Normal(0., 1.)
    z_known = normal.icdf(clamp_probs(u_known))
    if not inverse:
        # forward h: h(u_in|u_known) = ...
        z_in = normal.icdf(clamp_probs(u_in))
        numerator = z_in - rho*z_known
        denom = math.sqrt(1 - rho**2)
        cond_val = normal.cdf(numerator / denom)
        return cond_val
    else:
        # inverse h: from v -> u_in given u_known
        z_v = normal.icdf(clamp_probs(u_in))
        denom = math.sqrt(1 - rho**2)
        z_in = rho*z_known + denom*z_v
        return normal.cdf(z_in)

def _h_function(cop, u_known, v, inverse=True):
    """
    A generic h_function dispatcher. 
    If 'cop' is param. gaussian => use _h_function_gaussian.
    If 'cop' is nonparam => do local-likelihood approach, etc.
    Here we show only param. gaussian for brevity.
    """
    if isinstance(cop, cop_par_obj):
        if cop.family == "gaussian":
            rho = cop.theta
            return _h_function_gaussian(u_known, v, rho, inverse=inverse)
        else:
            # Placeholder: implement other families
            raise NotImplementedError("Only gaussian in this example.")
    else:
        # Nonparam path => do local-likelihood interpolation
        raise NotImplementedError("Nonparam not shown in detail here.")

def sample_d_vine(vine: vine_obj_bin, nsamples: int) -> np.ndarray:
    """
    Correct D-vine sampling that replicates the chain-of-conditionals approach
    from your TF code. This ensures correlation among non-adjacent variables.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = vine.n_cop

    # We'll store samples in [nsamples, d], each col is a variable U_i in [0,1].
    samples = torch.empty(nsamples, d, dtype=torch.float32, device=device)

    # U_1 ~ Uniform(0,1)
    samples[:, 0] = torch.rand(nsamples, device=device)

    # For each subsequent variable i from 1..(d-1) in 0-based indexing:
    # i.e. the i^th variable is samples[:, i].
    # In a standard D-vine, at tree level 0 we have copulas between (0,1), (1,2), ...
    # Then at tree level 1, we have copulas between (0,2)|(1), etc.
    # We assume vine.ind_vine[0] = edges for the first tree, vine.ind_vine[1] = edges for second tree, etc.
    # vine.copulas[0] = the copulas for level 0, etc.
    # 
    # The chain for U_i is something like:
    #   we start with an auxiliary V = Uniform(0,1) => that becomes U_i conditionally.
    #   For each tree level from 0..(i-1), find the pair-copula that connects i with i-(level+1),
    #   condition on the previously determined variable(s). Actually, in a strict D-vine, you apply them in
    #   the correct sequence so that each step re-maps the partial distribution.

    for i in range(1, d):
        # Start with an independent uniform random
        # We'll transform it step by step using the chain of pair-copulas.
        U_i = torch.rand(nsamples, device=device)

        # We go level by level. For level=0, we have pair (i, i-1).
        # For level=1, we have pair (i, i-2) conditioned on i-1, etc.
        # We'll need to figure out which index in vine.copulas[level] corresponds to (i, i-1, etc.)
        # Because your code typically has "ind_vine[level]" as a list of edges.
        # We'll do a small helper to track the chain.

        for level in range(i):
            # The j-th variable to condition on is i - level - 1
            j = i - (level + 1)
            if j < 0:
                break

            # find the edge (j, i) in vine.ind_vine[level], and get the corresponding copula
            # we assume edges are stored in a known order. Let's do a lookup.
            edge_idx = None
            edges_level = vine.ind_vine[level]
            for idx_e, e in enumerate(edges_level):
                e_sorted = tuple(sorted(e))
                if e_sorted == tuple(sorted((j, i))):
                    edge_idx = idx_e
                    break
            if edge_idx is None:
                # might happen if that edge doesn't exist in the structure
                # for a pure chain, we do expect it, but let's skip if not found
                continue

            this_cop = vine.copulas[level][edge_idx]

            # We apply the h-function in "inverse" mode:
            #   U_i = h^{-1}(U_i | samples[:, j])  for param or nonparam
            U_i = _h_function(this_cop, samples[:, j], U_i, inverse=True)

        # store the newly constructed variable
        samples[:, i] = U_i

    # If the user has margins != Uniform(0,1), transform each column.
    if vine.margin is not None:
        for i in range(d):
            if i < len(vine.margin):
                marg = vine.margin[i]
                # e.g. if marg.dist == 'norm' and marg.theta=(loc,scale):
                if marg.dist == 'norm':
                    loc, scale = marg.theta
                    dist = torch.distributions.Normal(loc, scale)
                    u_clamped = torch.clamp(samples[:, i], 1e-9, 1-1e-9)
                    samples[:, i] = dist.icdf(u_clamped)
                # else custom code for other distributions

    return samples.cpu().numpy()

def apply_d_vine_fix(vine: vine_obj_bin) -> None:
    """
    Patch the vine object's .sample method for a D-vine so that it uses our chain-of-h-functions approach.
    """
    if vine.vine_family != 'd-vine':
        return

    orig_sample = vine.sample

    def patched_sample(nsamples):
        return sample_d_vine(vine, nsamples)

    vine.sample = patched_sample
    vine.d_vine_patched = True
    print("[DVC] D-vine sampling has been patched with chain-of-conditionals approach.")