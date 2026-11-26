"""Binary three-neuron odds contrasts at fixed two-neuron margins."""
import numpy as np
from scipy.optimize import minimize, root


BITS = ((np.arange(8)[:, None] >> np.arange(3)) & 1).astype(int)
PARITY = np.where(BITS.sum(1) % 2, 1., -1.)


def at_eta(prior, eta):
    """Change the three-cell contrast while preserving all observed pairwise margins."""
    prior = np.asarray(prior, dtype=float)
    eta = np.broadcast_to(eta, prior.shape[:-1])
    left = -np.min(prior[..., PARITY > 0], axis=-1)
    right = np.min(prior[..., PARITY < 0], axis=-1)
    for _ in range(43):
        middle = 0.5 * (left + right)
        trial = prior + middle[..., None] * PARITY
        contrast = np.log(trial) @ PARITY
        below = contrast < eta
        left = np.where(below, middle, left)
        right = np.where(below, right, middle)
    return prior + (0.5 * (left + right))[..., None] * PARITY


def parameters(base,counts,design,sd):
    def objective(coeff):
        values=at_eta(base,design@coeff)
        inv=1./values
        derivative=inv*PARITY/inv.sum(axis=1,keepdims=True)
        penalty=coeff/(sd**2)
        value=-np.sum(counts*np.log(values))+.5*np.dot(coeff,penalty)
        grad=-(design.T@np.sum(counts*derivative,axis=1))+penalty
        return value,grad
    fit=minimize(objective,np.zeros(len(sd)),method="BFGS",jac=True,options={"gtol":2.e-5})
    if np.isfinite(fit.fun) and np.max(np.abs(fit.jac)) > 2.e-4:
        # For large counts the absolute log likelihood can lose the final BFGS
        # improvement to rounding. Refine its score equations at the same fit.
        refined = root(lambda coeff: objective(coeff)[1], fit.x)
        value, grad = objective(refined.x)
        if np.isfinite(value) and np.max(np.abs(grad)) <= 2.e-4 and value <= fit.fun + 1.e-12 * (1. + abs(fit.fun)):
            return refined.x
    if np.max(np.abs(fit.jac))>2.e-4 or not np.isfinite(fit.fun):
        raise ValueError(f"Unresolved coefficient fit: {fit.message}, {fit.jac}")
    return fit.x
