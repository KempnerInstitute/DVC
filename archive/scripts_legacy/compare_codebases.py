##########################################
# compare_vine_families.py
##########################################
import numpy as np
import torch

# (A) Simulate 5D Gaussian data with a chosen correlation structure
def simulate_mvn(n: int, d: int, corr: np.ndarray, seed: int = 123) -> np.ndarray:
    """
    Simulate N= `n` samples in D= `d` dimensions from a multivariate normal(0, I).
    'corr' is a (d x d) correlation matrix (symmetric positive-definite).
    """
    rng = np.random.default_rng(seed)
    mean = np.zeros(d, dtype=np.float32)
    # Ensure 'corr' is SPD and valid
    # e.g. user might supply a valid correlation
    # We'll do a Cholesky
    L = np.linalg.cholesky(corr)
    # standard normal
    z = rng.standard_normal((n, d))
    # correlated
    x = z @ L.T
    return x.astype(np.float32)

# Utility: compute correlation matrix
def corrcoef(samples: np.ndarray) -> np.ndarray:
    return np.corrcoef(samples.T)

###############################################################################
# (B) PyTorch code: import from your src/DVC/ codebase
###############################################################################
try:
    from DVC_pyolder.objects import vine_obj_bin, margin_obj
    from DVC_pyolder.vine_model import fit_vine
    pytorch_available = True
except ImportError as e:
    print(f"Could not import PyTorch DVC code: {e}. Please adjust import paths.")
    pytorch_available = False
    vine_obj_bin = None
    margin_obj = None
    fit_vine = None

def fit_and_sample_pytorch(data: np.ndarray,
                           vine_family: str,
                           param: bool = True,
                           seed: int = 999,
                           n_samples: int = 2000) -> np.ndarray:
    """
    Fits a vine of family 'vine_family' to 'data' using the PyTorch DVC code,
    then generates 'n_samples' from it. Returns the generated samples.
    """
    if not pytorch_available:
        print(f"PyTorch DVC implementation not available.")
        return np.zeros((n_samples, data.shape[1]), dtype=np.float32)
        
    torch.manual_seed(seed)
    d = data.shape[1]

    # Build trivial margins = Normal(0,1)
    margins = []
    for _ in range(d):
        margins.append(margin_obj(dist="norm", theta=(0.0, 1.0), is_cont=True))

    # Create vine object
    vine = vine_obj_bin(
        vine_family=vine_family,
        families=["gaussian"]*d,  # param_families
        vine_depth=d,
        margin=margins,
        knots=50
    )

    # Fit dicts
    gen_dict = {
        "binning": False,
        "param": param,
        "fitted": False,
        "parallel": False,
        "vine_depth": d  # Add vine_depth parameter
    }
    npc_dict = {
        "opt_method": "LL1",
        "batch_paral": 1  # Add batch_paral parameter
    }
    par_dict = {
        "param_families": ["gaussian","ind"]
    }
    bin_dict = {
        "n_bin": 1  # Add n_bin parameter
    }

    # Fit the vine
    fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)

    # Sample from the vine
    samples_vine = vine.sample(n_samples)
    
    # Verify samples don't have NaN values
    if np.any(np.isnan(samples_vine)):
        print(f"Warning: NaN values in PyTorch {vine_family} samples!")
        # Replace NaNs with zeros
        samples_vine = np.nan_to_num(samples_vine, nan=0.0)
    
    return samples_vine

###############################################################################
# (C) TensorFlow code: import from your src/DVC_tensorflow/ codebase
###############################################################################
# First check if TensorFlow is available
tensorflow_available = False
try:
    import tensorflow as tf
    tensorflow_available = True
except ImportError as e:
    print(f"TensorFlow not available: {e}")

# Only try to import TF-dependent modules if TF is available
tf_dvc_available = False
if tensorflow_available:
    try:
        # Try the local imports first
        sys_path_modified = False
        try:
            from src.DVC_tensorflow.classes.objects import vine_obj_bin as vine_obj_bin_tf
            from src.DVC_tensorflow.classes.objects import margin_obj as margin_obj_tf
            tf_dvc_available = True
            print("Using src.DVC_tensorflow module imports")
        except ImportError:
            # If that fails, try adding the Code/DVC directory to path
            import sys
            import os
            code_dvc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if code_dvc_path not in sys.path:
                sys.path.append(code_dvc_path)
                sys_path_modified = True
            try:
                # Try to import from all_tensorflow
                from all_tensorflow import vine_obj_bin as vine_obj_bin_tf
                from all_tensorflow import margin_obj as margin_obj_tf
                tf_dvc_available = True
                print("Using all_tensorflow imports")
            except ImportError as e:
                print(f"Could not import from all_tensorflow: {e}")
            
            # Restore sys.path if we modified it
            if sys_path_modified and code_dvc_path in sys.path:
                sys.path.remove(code_dvc_path)
        
        # Define a fit_vine_tf function that matches the API of our PyTorch version
        if tf_dvc_available:
            def fit_vine_tf(vine, data, gen_dict, npc_dict, par_dict, bin_dict):
                # Use the fit method directly from the vine object
                vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
                return vine
    except Exception as e:
        print(f"Error setting up TensorFlow DVC code: {e}")
        tf_dvc_available = False

def fit_and_sample_tensorflow(data: np.ndarray,
                              vine_family: str,
                              param: bool = True,
                              seed: int = 999,
                              n_samples: int = 2000) -> np.ndarray:
    """
    Fits a vine of family 'vine_family' to 'data' using the TensorFlow DVC code,
    then generates 'n_samples' from it. Returns the generated samples (as numpy).
    """
    if not tensorflow_available:
        print("TensorFlow is not available.")
        return np.zeros((n_samples, data.shape[1]), dtype=np.float32)
        
    if not tf_dvc_available:
        print("TensorFlow DVC implementation not available.")
        return np.zeros((n_samples, data.shape[1]), dtype=np.float32)

    try:
        tf.random.set_seed(seed)
        d = data.shape[1]

        # Build trivial margins
        margins = []
        for _ in range(d):
            margins.append(margin_obj_tf(dist="norm", theta=(0.0, 1.0), is_cont=True))

        # Create vine object with correct constructor
        if vine_family == "r-vine":
            # For r-vine, we need to provide a matrix
            # Create a simple r-vine matrix
            r_matrix = np.zeros((d, d), dtype=np.int32)
            for i in range(d):
                r_matrix[i, i] = i + 1
            for i in range(d-1):
                r_matrix[d-1, i] = d
            
            vine_tf = vine_obj_bin_tf(
                vine_family=vine_family,
                families=["gaussian"]*d,
                vine_depth=d,
                margin=margins,
                knots=50,
                method="matrix",
                r_matrix=r_matrix
            )
        else:
            # For c-vine and d-vine
            vine_tf = vine_obj_bin_tf(
                vine_family=vine_family,
                families=["gaussian"]*d,
                vine_depth=d,
                margin=margins,
                knots=50
            )
            
        gen_dict = {
            "binning": False,
            "param": param,
            "fitted": False,
            "parallel": False,
            "vine_depth": d  # Add vine_depth as it's used in the fit method
        }
        npc_dict = {
            "opt_method": "LL1",
            "batch_paral": 1  # Add batch_paral parameter
        }
        par_dict = {
            "param_families": ["gaussian","ind"]
        }
        bin_dict = {
            "n_bin": 1  # Add n_bin parameter
        }

        # Fit the vine
        fit_vine_tf(vine_tf, data, gen_dict, npc_dict, par_dict, bin_dict)

        # Sample
        samples_vine_tf = vine_tf.sample(n_samples)
        # If it returns a TF tensor, convert to numpy
        if isinstance(samples_vine_tf, tf.Tensor):
            samples_vine_tf = samples_vine_tf.numpy()
            
        # Verify samples don't have NaN values
        if np.any(np.isnan(samples_vine_tf)):
            print(f"Warning: NaN values in TensorFlow {vine_family} samples!")
            # Replace NaNs with zeros
            samples_vine_tf = np.nan_to_num(samples_vine_tf, nan=0.0)
            
        return samples_vine_tf
        
    except Exception as e:
        print(f"Error in TensorFlow implementation: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((n_samples, data.shape[1]), dtype=np.float32)

###############################################################################
# MAIN SCRIPT
###############################################################################
def main():
    np.set_printoptions(precision=3, suppress=True)

    # 1) Define a 5D correlation matrix
    d = 5
    # Example: let it be block diagonal or random
    # We'll pick a symmetrical matrix with correlations ~0.5 off-diagonal.
    corr = np.array([
        [1.0,  0.5,  0.3,  0.0,  0.1],
        [0.5,  1.0,  0.4,  0.5,  0.2],
        [0.3,  0.4,  1.0,  0.4,  0.3],
        [0.0,  0.5,  0.4,  1.0,  0.5],
        [0.1,  0.2,  0.3,  0.5,  1.0]
    ], dtype=np.float32)

    # 2) Simulate data
    N = 5000
    data = simulate_mvn(N, d, corr, seed=42)
    print("Ground Truth Correlation (5x5):\n", corr)
    
    # 3) Fit different vine families in PyTorch
    vine_families = ["d-vine", "c-vine"]  # Removed r-vine as it requires special handling
    pyro_results = {}
    
    if pytorch_available:
        for vfam in vine_families:
            try:
                print(f"\nFitting PyTorch {vfam}...")
                samples_vine = fit_and_sample_pytorch(data, vine_family=vfam, param=True, seed=123, n_samples=2000)
                corr_vine = corrcoef(samples_vine)
                pyro_results[vfam] = corr_vine
                print(f"=== PyTorch {vfam} ===")
                print("Empirical corr of vine samples:\n", corr_vine)
            except Exception as e:
                print(f"\nError with PyTorch {vfam}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\n[Warning] PyTorch vine code not found; skipping PyTorch fitting.")

    # 4) Fit different vine families in TensorFlow (if available)
    tf_results = {}
    if tensorflow_available and tf_dvc_available:
        for vfam in vine_families:
            try:
                print(f"\nFitting TensorFlow {vfam}...")
                samples_vine_tf = fit_and_sample_tensorflow(data, vine_family=vfam, param=True, seed=123, n_samples=2000)
                corr_vine_tf = corrcoef(samples_vine_tf)
                tf_results[vfam] = corr_vine_tf
                print(f"=== TensorFlow {vfam} ===")
                print("Empirical corr of vine samples:\n", corr_vine_tf)
            except Exception as e:
                print(f"\nError with TensorFlow {vfam}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\n[Warning] TensorFlow DVC implementation not available; skipping TF fitting.")

    # 5) Compare difference from ground truth
    print("\nComparison to ground truth correlation:")
    for vfam in vine_families:
        if vfam in pyro_results:
            pyro_corr = pyro_results[vfam]
            # Check for NaN values
            if np.any(np.isnan(pyro_corr)):
                print(f"[PyTorch {vfam}] Warning: NaN values in correlation matrix")
            else:
                pyro_err = np.mean(np.abs(pyro_corr - corr))
                print(f"[PyTorch {vfam}] mean absolute error = {pyro_err:.4f}")
                
        if vfam in tf_results:
            tf_corr = tf_results[vfam]
            # Check for NaN values
            if np.any(np.isnan(tf_corr)):
                print(f"[TF      {vfam}] Warning: NaN values in correlation matrix")
            else:
                tf_err = np.mean(np.abs(tf_corr - corr))
                print(f"[TF      {vfam}] mean absolute error = {tf_err:.4f}")

if __name__ == "__main__":
    main()