"""
Optional VDC (Vine-Diffusion-Copula) backend for DVC.

Provides a wrapper around the ``vdc`` package so that DVC can use
diffusion-based nonparametric pair-copula estimation instead of (or
alongside) classical parametric families.

Installation
------------
Install an external ``vdc`` package in the active environment before using this
optional backend. For example::

    pip install -e /path/to/vine_diffusion_copula

Usage within DVC
----------------
>>> from dvc_package.baselines.vdc_backend import VDCBackend, vdc_available
>>> if vdc_available():
...     backend = VDCBackend.from_checkpoint("model.pt", device="cuda")
...     vine = backend.fit(U_train, vine_type="dvine")
...     logp = backend.logpdf(U_test)
...     samples = backend.simulate(1000)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger("DVC.vdc_backend")

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

_VDC_AVAILABLE: Optional[bool] = None


def vdc_available() -> bool:
    """Return True if the ``vdc`` package can be imported."""
    global _VDC_AVAILABLE
    if _VDC_AVAILABLE is None:
        try:
            import vdc  # noqa: F401
            _VDC_AVAILABLE = True
        except ImportError:
            _VDC_AVAILABLE = False
    return _VDC_AVAILABLE


def _require_vdc():
    if not vdc_available():
        raise ImportError(
            "The 'vdc' (vine-diffusion-copula) package is not installed. "
            "Install it in the active environment before using VDCBackend."
        )


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class VDCBackend:
    """Thin wrapper around ``vdc.VineCopulaModel`` for use inside DVC.

    This class manages the lifecycle of a pre-trained diffusion model and
    a fitted ``VineCopulaModel``, exposing the same logical interface
    (fit, logpdf, simulate, rosenblatt) that DVC's own vine pipeline uses.

    Parameters
    ----------
    model : torch.nn.Module
        A trained diffusion model (``GridUNet``, ``CopulaDenoiser``, or
        ``CopulaCNNEnhanced``).
    diffusion : optional
        A ``CopulaAwareDiffusion`` instance for iterative sampling.
        If ``None``, single-pass inference is used (faster but less accurate).
    vine_type : str
        One of ``'dvine'``, ``'cvine'``, ``'rvine'``.
    m : int
        Grid resolution for density estimation.
    device : str
        PyTorch device string.
    diffusion_steps : int
        Number of DDIM steps for iterative density estimation.
    projection_iters : int
        IPFP iterations for copula constraint enforcement.
    """

    def __init__(
        self,
        model,
        diffusion=None,
        vine_type: str = "dvine",
        m: int = 64,
        device: str = "cuda",
        diffusion_steps: int = 50,
        projection_iters: int = 50,
    ):
        _require_vdc()
        import torch
        from vdc.vine.api import VineCopulaModel

        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.m = m
        self.vine_type = vine_type
        self.diffusion_steps = diffusion_steps
        self.projection_iters = projection_iters

        # Ensure model is on the correct device and in eval mode
        self.model = self.model.to(device)
        self.model.eval()

        # Fitted vine - populated after calling .fit()
        self._vine: Optional[VineCopulaModel] = None

    # -----------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "cuda",
        vine_type: str = "dvine",
        m: int = 64,
        use_diffusion: bool = False,
        diffusion_steps: int = 50,
    ) -> "VDCBackend":
        """Load a pre-trained VDC model from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to a ``.pt`` checkpoint saved by ``vdc`` training.
        device : str
            Target device.
        vine_type : str
            Vine structure type (``'dvine'``, ``'cvine'``, ``'rvine'``).
        m : int
            Grid resolution.
        use_diffusion : bool
            If True, also instantiate the ``CopulaAwareDiffusion`` process
            for iterative (higher-quality) density estimation.
        diffusion_steps : int
            Number of DDIM steps (only relevant when ``use_diffusion=True``).

        Returns
        -------
        VDCBackend
        """
        _require_vdc()
        import torch
        from vdc.models.unet_grid import GridUNet

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Reconstruct model from saved config
        model_cfg = ckpt.get("config", {}).get("model", {})
        model = GridUNet(
            m=model_cfg.get("m", m),
            in_channels=model_cfg.get("in_channels", 1),
            base_channels=model_cfg.get("base_channels", 64),
            channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 3, 4])),
            num_res_blocks=model_cfg.get("num_res_blocks", 2),
        )
        model.load_state_dict(ckpt["model_state_dict"])

        diffusion = None
        if use_diffusion:
            from vdc.models.copula_diffusion import CopulaAwareDiffusion

            diff_cfg = ckpt.get("config", {}).get("diffusion", {})
            diffusion = CopulaAwareDiffusion(
                timesteps=diff_cfg.get("timesteps", 1000),
                beta_schedule=diff_cfg.get("noise_schedule", "cosine"),
            ).to(device)

        return cls(
            model=model,
            diffusion=diffusion,
            vine_type=vine_type,
            m=m,
            device=device,
            diffusion_steps=diffusion_steps,
        )

    # -----------------------------------------------------------------
    # Core API (mirrors DVC vine pipeline)
    # -----------------------------------------------------------------

    def fit(
        self,
        U: np.ndarray,
        order: Optional[list] = None,
        truncation_level: Optional[int] = None,
        verbose: bool = True,
    ) -> "VDCBackend":
        """Fit a vine copula to pseudo-observations *U*.

        Parameters
        ----------
        U : ndarray of shape (n, d)
            Pseudo-observations in [0, 1]^d.
        order : list of int, optional
            Variable ordering for D-vine / C-vine.
        truncation_level : int, optional
            Truncate vine at this tree level.
        verbose : bool
            Print progress.

        Returns
        -------
        self
        """
        from vdc.vine.api import VineCopulaModel

        vine = VineCopulaModel(
            vine_type=self.vine_type,
            order=order,
            truncation_level=truncation_level,
            m=self.m,
            device=self.device,
            diffusion_steps=self.diffusion_steps,
            projection_iters=self.projection_iters,
        )
        vine.fit(U, self.model, diffusion=self.diffusion, verbose=verbose)
        self._vine = vine
        logger.info("VDC vine fitted (%s, d=%d)", self.vine_type, U.shape[1])
        return self

    def logpdf(self, U: np.ndarray) -> np.ndarray:
        """Evaluate log-density at pseudo-observations *U*."""
        if self._vine is None:
            raise RuntimeError("Call .fit() before .logpdf()")
        return self._vine.logpdf(U)

    def pdf(self, U: np.ndarray) -> np.ndarray:
        """Evaluate density at pseudo-observations *U*."""
        if self._vine is None:
            raise RuntimeError("Call .fit() before .pdf()")
        return self._vine.pdf(U)

    def simulate(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate *n* samples from the fitted vine copula."""
        if self._vine is None:
            raise RuntimeError("Call .fit() before .simulate()")
        return self._vine.simulate(n, seed=seed)

    def rosenblatt(self, U: np.ndarray) -> np.ndarray:
        """Forward Rosenblatt transform (copula -> independent uniforms)."""
        if self._vine is None:
            raise RuntimeError("Call .fit() before .rosenblatt()")
        return self._vine.rosenblatt(U)

    def nll(self, U: np.ndarray) -> float:
        """Mean negative log-likelihood on *U*."""
        logp = self.logpdf(U)
        return float(-np.mean(logp[np.isfinite(logp)]))

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save_vine(self, filepath: Union[str, Path]) -> None:
        """Save the fitted vine (not the diffusion model weights)."""
        if self._vine is None:
            raise RuntimeError("Nothing to save - call .fit() first")
        self._vine.save(filepath)

    def load_vine(self, filepath: Union[str, Path]) -> None:
        """Load a previously saved vine."""
        from vdc.vine.api import VineCopulaModel

        self._vine = VineCopulaModel.load(filepath)

    # -----------------------------------------------------------------
    # Integration with DVC experiment framework
    # -----------------------------------------------------------------

    def evaluate_against_dvc(
        self,
        U_train: np.ndarray,
        U_test: np.ndarray,
    ) -> dict:
        """Fit on *U_train*, evaluate NLL on *U_test*.

        Returns a dict with keys ``'train_nll'``, ``'test_nll'``,
        ``'vine_type'``, and ``'n_dimensions'``.
        """
        self.fit(U_train, verbose=False)
        train_nll = self.nll(U_train)
        test_nll = self.nll(U_test)
        return {
            "train_nll": train_nll,
            "test_nll": test_nll,
            "vine_type": self.vine_type,
            "n_dimensions": U_train.shape[1],
            "backend": "vdc_diffusion",
        }
