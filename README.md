# Distributed Vine Copula Library in PyTorch

This repository implements a full vine copula library in PyTorch. It provides functionality for:

- Nonparametric vine copula density estimation (with kernel‐based local likelihood optimization)
- Parametric copula fitting (Gaussian, Student, Clayton, Clayton‑rotated)
- Vine structure construction (for C‑vine, D‑vine, and R‑vine)
- Sampling from the fitted vine model
- Evaluating multivariate densities and decomposing entropy
- Predicting conditional densities (using maximum likelihood and expectation estimates)

The code is organized into several modules under the `src/` folder:
- **`classes/`**: Contains main object definitions (e.g. `vine_obj_bin`, `margin_obj`).
- **`info/`**: Mutual information/entropy estimation.
- **`param/`**: Functions for parametric copula fitting, local likelihood fitting, and generating a random vine.
- **`pred/`**: Prediction routines.
- **`pre_proc/`**: Preprocessing and vine structure definitions.
- **`sampling/`**: Sampling from the vine.
- **`utils/`**: Helper modules for tensor operations, interpolation, bandwidth selection, NAdam optimization, and tree operations.

## Installation and Setup

### Requirements

- Python 3.8 or later

