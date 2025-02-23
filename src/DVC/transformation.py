###############################################
# src/DVC/transformation.py
###############################################

import torch
import math
from torch.distributions import Normal

class Transform:
    """
    Transform class for mapping:
      - uniform u -> normal s space (via inverse CDF, clipped to [-3.2,3.2])
      - optional PCA or SVD-based rotation on the 's' space => 'x' space

    Typically:
      1) forward_u(obj_u):  apply Normal icdf
      2) forward_s(obj_s):  if self.coeff is None, do an SVD or PCA to get 'coeff', then transform
    """

    def __init__(self,
                 n_cop: int,
                 use_pca: bool = False):
        """
        Args:
          n_cop: number of bivariate edges or dimension (some contexts)
          use_pca: if True, we actually do a PCA transform in forward_s
        """
        self.n_cop = n_cop
        self.use_pca = use_pca

        # If we do PCA or SVD, we store them here
        self.coeff = None   # shape [2,2,...] if we do 2D, or (d,d) for general
        self.mu = None      # shape [*,2], the mean, or [d] for general dimension

    def forward_u(self, obj_u: torch.Tensor):
        """
        Maps from uniform in [0,1] => normal in [-3.2,3.2], by icdf & clamp.

        Args:
          obj_u: shape [*, D], typically in [0,1]
        Returns:
          obj_s: shape same, in ~[-3.2,3.2]
        """
        eps = 1e-7
        clipped = torch.clamp(obj_u, eps, 1.0 - eps)
        s = Normal(0.,1.).icdf(clipped)  # shape same
        # clamp to [-3.2, 3.2]
        s = torch.clamp(s, -3.2, 3.2)
        return s

    def forward_s(self, obj_s: torch.Tensor):
        """
        Optionally apply a PCA or SVD-based transform to 'obj_s'.

        Steps:
          1) If use_pca=False => return obj_s as-is.
          2) Else => if self.coeff is None:
               - compute mu => the mean of obj_s
               - center => obj_s - mu
               - SVD => s,u,coeff => store 'coeff' for rotation
             Then transform => (obj_s - mu) @ coeff

        The shape is expected to be [N,2] or [N,2,k], so we do multiple columns if needed.
        For simpler usage, if dimension>2, you'd do an alternative approach.
        """
        if not self.use_pca:
            # do nothing
            return obj_s

        # we assume a 2D shape or [N,2,...]. We'll handle [N,2,k].
        shape_ = obj_s.shape
        # e.g. [N,2] or [N,2,k]
        if obj_s.dim() == 2:
            # shape [N,2]
            if self.coeff is None:
                # compute mu => [2]
                self.mu = obj_s.mean(dim=0)
                # center data
                centered = obj_s - self.mu
                # do SVD => shape [N,2], we want [2,2]
                # for 2D => s, u, v => we get (u*s)@v^T => data
                # We'll do a torch.linalg.svd
                # note: 'centered' => shape [N,2]
                u_, s_, v_ = torch.linalg.svd(centered, full_matrices=False)
                # v_ => shape [2,2], we'll use that as the rotation
                # we want to ensure a consistent sign => typical approach
                # we'll just store v_ as is
                self.coeff = v_
            # now transform
            centered = obj_s - self.mu
            obj_x = centered @ self.coeff
            return obj_x

        elif obj_s.dim() == 3:
            # shape [N,2,k]. We'll do a loop over k or replicate logic.
            N, two, k = shape_
            # if self.coeff is None, we build it for each 'k' or handle them separately
            # for demonstration, we'll do a single transform for each slice => store an array
            if self.coeff is None:
                # we store a list of v_ => one per k
                self.mu = []
                self.coeff = []
                for i in range(k):
                    slice_i = obj_s[:,:,i]  # shape [N,2]
                    mean_i = slice_i.mean(dim=0)
                    centered_i = slice_i - mean_i
                    u_, s_, v_ = torch.linalg.svd(centered_i, full_matrices=False)
                    self.mu.append(mean_i)
                    self.coeff.append(v_)
            # now transform each slice
            out_slices = []
            for i in range(k):
                slice_i = obj_s[:,:,i]
                mean_i = self.mu[i]
                v_ = self.coeff[i]
                centered_i = slice_i - mean_i
                obj_x_i = centered_i @ v_
                out_slices.append(obj_x_i.unsqueeze(2))
            obj_x = torch.cat(out_slices, dim=2)  # shape [N,2,k]
            return obj_x
        else:
            # fallback => no transform
            return obj_s