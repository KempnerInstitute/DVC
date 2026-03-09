"""
Benchmark Datasets and Tests for Vine Copula Evaluation

Provides standard benchmark datasets and evaluation procedures
for comparing vine copula methods.
"""

import numpy as np
import pandas as pd
import time
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from ..core.info_estimation import mutual_information, vine_entropy
from ..core.vine_factory import create_vine
from ..core.vine_model import fit_vine

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkDataset:
    """Container for benchmark dataset information."""
    name: str
    data: np.ndarray
    scenario: str
    metadata: Dict[str, Any]
    time_data: Optional[np.ndarray] = None
    true_parameters: Optional[Dict[str, Any]] = None


def generate_benchmark_data(scenario: str,
                          dimension: int,
                          n_samples: int = 1000,
                          correlation_strength: float = 0.5,
                          noise_level: float = 0.1,
                          random_seed: int = 42) -> BenchmarkDataset:
    """
    Generate benchmark datasets for vine copula evaluation.
    
    Parameters
    ----------
    scenario : str
        Type of benchmark scenario
    dimension : int
        Number of variables
    n_samples : int
        Number of samples to generate
    correlation_strength : float
        Strength of correlations (0-1)
    noise_level : float
        Level of noise to add
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    BenchmarkDataset
        Generated benchmark dataset
    """
    np.random.seed(random_seed)
    
    if scenario == "gaussian":
        return _generate_gaussian_benchmark(
            dimension, n_samples, correlation_strength, noise_level
        )
    elif scenario == "mixed":
        return _generate_mixed_benchmark(
            dimension, n_samples, correlation_strength, noise_level
        )
    elif scenario == "heavy_tailed":
        return _generate_heavy_tailed_benchmark(
            dimension, n_samples, correlation_strength, noise_level
        )
    elif scenario == "time_varying_correlation":
        return _generate_time_varying_benchmark(
            dimension, n_samples, correlation_strength, noise_level
        )
    elif scenario == "regime_switching":
        return _generate_regime_switching_benchmark(
            dimension, n_samples, correlation_strength, noise_level
        )
    else:
        raise ValueError(f"Unknown benchmark scenario: {scenario}")


def _generate_gaussian_benchmark(dimension: int, n_samples: int,
                               correlation_strength: float, 
                               noise_level: float) -> BenchmarkDataset:
    """Generate multivariate Gaussian benchmark data."""
    
    # Create correlation matrix with specified strength
    corr_matrix = np.eye(dimension)
    
    # Add correlations
    for i in range(dimension):
        for j in range(i + 1, dimension):
            # Exponential decay with distance
            corr = correlation_strength * np.exp(-0.5 * abs(i - j))
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    
    # Generate data
    data = np.random.multivariate_normal(
        mean=np.zeros(dimension),
        cov=corr_matrix,
        size=n_samples
    )
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, data.shape)
        data += noise
    
    return BenchmarkDataset(
        name=f"gaussian_{dimension}d",
        data=data,
        scenario="gaussian",
        metadata={
            'correlation_matrix': corr_matrix.tolist(),
            'correlation_strength': correlation_strength,
            'noise_level': noise_level,
            'n_samples': n_samples,
            'dimension': dimension
        },
        true_parameters={'correlation_matrix': corr_matrix}
    )


def _generate_mixed_benchmark(dimension: int, n_samples: int,
                            correlation_strength: float,
                            noise_level: float) -> BenchmarkDataset:
    """Generate mixed copula benchmark data."""
    
    # Generate uniform marginals with different copula structures
    data = np.zeros((n_samples, dimension))
    
    # Use different copula types for different pairs
    for i in range(dimension):
        data[:, i] = np.random.uniform(0, 1, n_samples)
    
    # Apply different transformations to create dependencies
    if dimension >= 2:
        # Clayton copula structure for first two variables
        theta = 2.0 * correlation_strength
        u1, u2 = data[:, 0], data[:, 1]
        
        # Transform using Clayton copula
        t = np.random.uniform(0, 1, n_samples)
        v = ((u1**(-theta) - 1) * t + 1)**(-1/theta)
        data[:, 1] = v
    
    if dimension >= 3:
        # Frank copula structure for variables 2 and 3
        theta = 5.0 * correlation_strength
        u2, u3 = data[:, 1], data[:, 2]
        
        # Simplified Frank copula transformation
        alpha = 1 - np.exp(-theta)
        v = -np.log(1 - alpha * (1 - np.exp(-theta * u2)) / 
                   (np.exp(-theta * u3) - 1)) / theta
        data[:, 2] = np.clip(v, 0, 1)
    
    # Transform to normal marginals
    from scipy.stats import norm
    for i in range(dimension):
        data[:, i] = norm.ppf(np.clip(data[:, i], 1e-6, 1-1e-6))
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, data.shape)
        data += noise
    
    return BenchmarkDataset(
        name=f"mixed_{dimension}d",
        data=data,
        scenario="mixed",
        metadata={
            'correlation_strength': correlation_strength,
            'noise_level': noise_level,
            'n_samples': n_samples,
            'dimension': dimension,
            'copula_types': ['clayton', 'frank', 'gaussian']
        }
    )


def _generate_heavy_tailed_benchmark(dimension: int, n_samples: int,
                                   correlation_strength: float,
                                   noise_level: float) -> BenchmarkDataset:
    """Generate heavy-tailed benchmark data using t-distribution."""
    
    # Create correlation matrix
    corr_matrix = np.eye(dimension)
    for i in range(dimension):
        for j in range(i + 1, dimension):
            corr = correlation_strength * np.exp(-0.3 * abs(i - j))
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    
    # Generate multivariate t-distributed data
    df = 3  # Degrees of freedom for heavy tails
    
    # Generate standard multivariate normal
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(dimension),
        cov=corr_matrix,
        size=n_samples
    )
    
    # Generate chi-squared random variables
    chi2_samples = np.random.chisquare(df, n_samples)
    
    # Transform to t-distribution
    data = normal_data * np.sqrt(df / chi2_samples[:, np.newaxis])
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, data.shape)
        data += noise
    
    return BenchmarkDataset(
        name=f"heavy_tailed_{dimension}d",
        data=data,
        scenario="heavy_tailed",
        metadata={
            'correlation_matrix': corr_matrix.tolist(),
            'degrees_of_freedom': df,
            'correlation_strength': correlation_strength,
            'noise_level': noise_level,
            'n_samples': n_samples,
            'dimension': dimension
        },
        true_parameters={
            'correlation_matrix': corr_matrix,
            'degrees_of_freedom': df
        }
    )


def _generate_time_varying_benchmark(dimension: int, n_samples: int,
                                   correlation_strength: float,
                                   noise_level: float) -> BenchmarkDataset:
    """Generate time-varying correlation benchmark data."""
    
    # Time points
    time_points = np.linspace(0, 4*np.pi, n_samples)
    
    # Time-varying correlation
    base_corr = correlation_strength
    time_varying_corr = base_corr + 0.3 * np.sin(time_points)
    
    data = np.zeros((n_samples, dimension))
    
    # Generate data with time-varying correlations
    for t in range(n_samples):
        # Create correlation matrix for this time point
        corr_t = np.eye(dimension)
        current_corr = time_varying_corr[t]
        
        for i in range(dimension):
            for j in range(i + 1, dimension):
                corr_ij = current_corr * np.exp(-0.2 * abs(i - j))
                corr_t[i, j] = corr_ij
                corr_t[j, i] = corr_ij
        
        # Generate single sample from this correlation
        sample = np.random.multivariate_normal(
            mean=np.zeros(dimension),
            cov=corr_t,
            size=1
        )[0]
        
        data[t, :] = sample
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, data.shape)
        data += noise
    
    return BenchmarkDataset(
        name=f"time_varying_{dimension}d",
        data=data,
        scenario="time_varying_correlation",
        metadata={
            'base_correlation': base_corr,
            'correlation_amplitude': 0.3,
            'time_frequency': 1.0,
            'noise_level': noise_level,
            'n_samples': n_samples,
            'dimension': dimension
        },
        time_data=time_points,
        true_parameters={
            'time_points': time_points,
            'correlation_function': time_varying_corr
        }
    )


def _generate_regime_switching_benchmark(dimension: int, n_samples: int,
                                       correlation_strength: float,
                                       noise_level: float) -> BenchmarkDataset:
    """Generate regime-switching benchmark data."""
    
    # Define two regimes
    regime_1_corr = correlation_strength
    regime_2_corr = -0.5 * correlation_strength
    
    # Regime switching points
    switch_points = [n_samples // 3, 2 * n_samples // 3]
    regimes = np.array([0] * switch_points[0] + 
                      [1] * (switch_points[1] - switch_points[0]) +
                      [0] * (n_samples - switch_points[1]))
    
    data = np.zeros((n_samples, dimension))
    
    # Generate correlation matrices for each regime
    corr_matrices = []
    for regime_corr in [regime_1_corr, regime_2_corr]:
        corr_matrix = np.eye(dimension)
        for i in range(dimension):
            for j in range(i + 1, dimension):
                corr = regime_corr * np.exp(-0.2 * abs(i - j))
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        corr_matrices.append(corr_matrix)
    
    # Generate data according to regimes
    for t in range(n_samples):
        regime = regimes[t]
        corr_matrix = corr_matrices[regime]
        
        sample = np.random.multivariate_normal(
            mean=np.zeros(dimension),
            cov=corr_matrix,
            size=1
        )[0]
        
        data[t, :] = sample
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, data.shape)
        data += noise
    
    return BenchmarkDataset(
        name=f"regime_switching_{dimension}d",
        data=data,
        scenario="regime_switching",
        metadata={
            'regime_1_correlation': regime_1_corr,
            'regime_2_correlation': regime_2_corr,
            'switch_points': switch_points,
            'noise_level': noise_level,
            'n_samples': n_samples,
            'dimension': dimension
        },
        time_data=np.arange(n_samples),
        true_parameters={
            'regimes': regimes,
            'correlation_matrices': corr_matrices,
            'switch_points': switch_points
        }
    )


def run_benchmark_suite(vine_methods: List[str],
                       scenarios: List[str] = None,
                       dimensions: List[int] = None,
                       n_samples: int = 1000,
                       output_dir: str = "benchmark_results") -> Dict[str, Any]:
    """
    Run comprehensive benchmark suite comparing vine methods.
    
    Parameters
    ----------
    vine_methods : list of str
        Vine methods to compare
    scenarios : list of str, optional
        Benchmark scenarios to test
    dimensions : list of int, optional
        Dimensions to test
    n_samples : int
        Number of samples per dataset
    output_dir : str
        Output directory for results
        
    Returns
    -------
    dict
        Benchmark results
    """
    if scenarios is None:
        scenarios = ["gaussian", "mixed", "heavy_tailed"]
    
    if dimensions is None:
        dimensions = [3, 4, 5]
    
    logger.info(f"Running benchmark suite: {len(vine_methods)} methods, "
                f"{len(scenarios)} scenarios, {len(dimensions)} dimensions")
    
    results = {
        'methods': vine_methods,
        'scenarios': scenarios,
        'dimensions': dimensions,
        'results': {}
    }
    
    # Generate all benchmark datasets
    datasets = []
    for scenario in scenarios:
        for dim in dimensions:
            dataset = generate_benchmark_data(
                scenario=scenario,
                dimension=dim,
                n_samples=n_samples
            )
            datasets.append(dataset)
    
    # Test each method on each dataset
    for method in vine_methods:
        method_results = {}
        
        for dataset in datasets:
            dataset_key = f"{dataset.scenario}_{dataset.metadata['dimension']}d"
            
            try:
                # This is a placeholder for actual method evaluation
                # In practice, you'd fit the vine method and evaluate metrics
                method_result = _evaluate_method_on_dataset(method, dataset)
                method_results[dataset_key] = method_result
                
            except Exception as e:
                logger.error(f"Method {method} failed on dataset {dataset_key}: {e}")
                method_results[dataset_key] = {'error': str(e)}
        
        results['results'][method] = method_results
    
    return results


def _evaluate_method_on_dataset(method: str, dataset: BenchmarkDataset) -> Dict[str, Any]:
    """Evaluate a vine method on a benchmark dataset."""
    method_norm = method.strip().lower()
    is_nonparam = "nonparam" in method_norm or "kernel" in method_norm
    vine_type = "d-vine" if "d-vine" in method_norm or "dvine" in method_norm else "c-vine"
    supported_tokens = {"c-vine", "cvine", "d-vine", "dvine", "parametric", "nonparametric", "kernel"}
    if not any(token in method_norm for token in supported_tokens):
        raise NotImplementedError(
            f"Unsupported benchmark method '{method}'. Supported strings should identify "
            "a C-vine/D-vine and optionally whether the fit is nonparametric."
        )

    data = np.asarray(dataset.data, dtype=np.float32)
    n_samples, dimension = data.shape
    split = max(int(round(0.8 * n_samples)), min(n_samples - 1, dimension + 5))
    if split <= 1 or split >= n_samples:
        raise ValueError(f"Dataset {dataset.name} is too small for a train/test benchmark split")
    train = data[:split]
    test = data[split:]

    vine = create_vine(vine_type, dimension)
    gen_dict = {"param": not is_nonparam, "binning": False, "fitted": False}
    npc_dict = {
        "opt_method": "LL1",
        "max_iter_phase1": 4,
        "max_iter_phase2": 6,
        "normal_iters_phase1": 25,
        "normal_iters_phase2": 50,
        "final_normalization_iters": 100,
        "batch_size": 2,
    }
    par_dict = {"param_families": ["ind", "gaussian", "student", "clayton", "frank", "gumbel"]}
    bin_dict: Dict[str, Any] = {}

    start = time.perf_counter()
    fit_vine(vine, train, gen_dict, npc_dict, par_dict, bin_dict)
    fitting_time = float(time.perf_counter() - start)

    test_tensor = torch.tensor(test, dtype=torch.float32)
    logpdf = vine.logpdf(test_tensor)
    mean_log_likelihood = float(np.mean(np.asarray(logpdf.detach().cpu().numpy() if hasattr(logpdf, "detach") else logpdf)))

    info_dict = {"alpha": 0.05, "cases": min(200, max(50, split // 4)), "iterations": 3, "units": "bits"}
    entropy = float(vine_entropy(vine, info_dict))

    pair_candidates = [(i, i + 1) for i in range(min(dimension - 1, 3))]
    if not pair_candidates:
        mean_pairwise_mi = 0.0
    else:
        mi_vals = [float(mutual_information(vine, [i], [j], info_dict)) for i, j in pair_candidates]
        mean_pairwise_mi = float(np.mean(mi_vals))

    return {
        'method': method,
        'dataset': dataset.name,
        'entropy': entropy,
        'mutual_information': mean_pairwise_mi,
        'fitting_time': fitting_time,
        'log_likelihood': mean_log_likelihood,
        'train_size': int(split),
        'test_size': int(n_samples - split),
        'vine_type': vine_type,
        'nonparametric': bool(is_nonparam),
    }


def load_real_world_datasets() -> List[BenchmarkDataset]:
    """Load real-world datasets for benchmarking."""
    
    datasets = []
    
    # This is a placeholder for loading real datasets
    # In practice, you might load:
    # - Financial time series
    # - Climate data
    # - Biological measurements
    # - etc.
    
    logger.info("Real-world dataset loading not implemented yet")
    
    return datasets
