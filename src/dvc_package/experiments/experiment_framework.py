"""
Comprehensive experiment framework for DVC vine copula analysis.

This module provides a flexible experiment framework that can be configured
via YAML files to run various types of experiments including:
- Probability distribution analysis
- Entropy and information estimation
- Time-dependent vine copula modeling
- Comparative studies between different vine types
- Performance benchmarking
"""

import yaml
import torch
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

from ..core.vine_factory import create_vine, VineType
from ..core.param_copula import parametric_fit
from ..core.d_vine_fix import compute_correlation_matrix, compute_kendall_tau_matrix
from ..core.vine_model import fit_vine
from ..time.data import generate_synthetic_time_series, create_data_loader, preprocess_real_data
from ..time.models import create_time_dependent_vine
from ..utils.utils_tensor import replace_nan_inf, handle_small_sample_size
from .neurips_simulations import run_neurips_simulation_suite

logger = logging.getLogger("DVC.experiments")


@dataclass
class ExperimentConfig:
    """Configuration for experiments loaded from YAML."""
    
    # General experiment settings
    name: str = "default_experiment"
    description: str = ""
    output_dir: str = "experiment_results"
    seed: Optional[int] = 42
    
    # Data generation settings
    data_config: Dict[str, Any] = field(default_factory=dict)
    
    # Vine copula settings
    vine_config: Dict[str, Any] = field(default_factory=dict)
    
    # Time-dependent settings
    time_config: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis settings
    analysis_config: Dict[str, Any] = field(default_factory=dict)
    
    # Plotting settings
    plot_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, indent=2)


class BaseExperiment(ABC):
    """Base class for all experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize base experiment."""
        self.config = config
        self.results = {}
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set random seed for reproducibility
        if config.seed is not None:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
        
        # Setup logging
        self.logger = logging.getLogger(f"DVC.experiments.{self.__class__.__name__}")
        
        # Initialize results directory structure
        self._setup_output_directories()
    
    def _setup_output_directories(self):
        """Setup output directory structure."""
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the experiment and return results."""
        pass
    
    def save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        # Save as JSON
        results_file = self.output_dir / f"{self.config.name}_results.json"
        
        # Convert numpy arrays and tensors for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save as pickle for full object preservation
        import pickle
        pickle_file = self.output_dir / f"{self.config.name}_results.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and tensors to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


class ProbabilityAnalysisExperiment(BaseExperiment):
    """Experiment for probability distribution analysis."""
    
    def run(self) -> Dict[str, Any]:
        """Run probability analysis experiment."""
        self.logger.info(f"Running probability analysis experiment: {self.config.name}")
        
        results = {
            'experiment_type': 'probability_analysis',
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__
        }
        
        # Generate or load data
        data = self._generate_data()
        results['data_info'] = {
            'shape': data.shape,
            'n_samples': data.shape[0],
            'n_variables': data.shape[1] if len(data.shape) > 1 else 1
        }
        
        # Fit vine copulas
        vine_results = self._fit_vine_copulas(data)
        results['vine_analysis'] = vine_results
        
        # Probability distribution analysis
        prob_results = self._analyze_probability_distributions(data, vine_results)
        results['probability_analysis'] = prob_results
        
        # Generate plots
        self._create_probability_plots(data, vine_results, prob_results)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def _generate_data(self) -> np.ndarray:
        """Generate or load data based on configuration."""
        data_config = self.config.data_config
        
        if data_config.get('type') == 'synthetic':
            n_samples = data_config.get('n_samples', 1000)
            n_variables = data_config.get('n_variables', 3)
            correlation_type = data_config.get('correlation_type', 'moderate')
            
            if correlation_type == 'moderate':
                # Generate data with moderate correlations
                corr_matrix = np.eye(n_variables)
                for i in range(n_variables):
                    for j in range(i + 1, n_variables):
                        corr_val = 0.3 + 0.2 * np.random.randn()
                        corr_val = np.clip(corr_val, -0.8, 0.8)
                        corr_matrix[i, j] = corr_matrix[j, i] = corr_val
                
                # Ensure positive definite
                eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
                eigenvals = np.maximum(eigenvals, 0.1)
                corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                
                data = np.random.multivariate_normal(
                    np.zeros(n_variables), corr_matrix, n_samples
                )
            
            elif correlation_type == 'block':
                # Generate block-structured correlations
                block_size = n_variables // 2
                data = np.random.randn(n_samples, n_variables)
                
                # Add block correlations
                for i in range(0, n_variables, block_size):
                    block_end = min(i + block_size, n_variables)
                    block_data = data[:, i:block_end]
                    
                    # Add common factor to create correlations
                    common_factor = np.random.randn(n_samples, 1)
                    data[:, i:block_end] = 0.7 * block_data + 0.3 * common_factor
            
            else:
                # Independent variables
                data = np.random.randn(n_samples, n_variables)
        
        elif data_config.get('type') == 'file':
            # Load from file
            file_path = data_config.get('path')
            data = np.load(file_path)
        
        else:
            # Default: simple multivariate normal
            data = np.random.multivariate_normal([0, 0, 0], np.eye(3), 1000)
        
        self.logger.info(f"Generated data with shape: {data.shape}")
        return data
    
    def _fit_vine_copulas(self, data: np.ndarray) -> Dict[str, Any]:
        """Fit different types of vine copulas."""
        vine_config = self.config.vine_config
        vine_types = vine_config.get('types', ['c-vine', 'd-vine'])
        families = vine_config.get('families', ['independence', 'gaussian', 'clayton'])
        
        results = {}
        
        for vine_type in vine_types:
            self.logger.info(f"Fitting {vine_type}")
            
            try:
                vine = create_vine(
                    vine_type=vine_type,
                    vine_depth=data.shape[1],
                    families=families,
                    data=data
                )
                
                # Fit vine (simplified - you may need to adapt this)
                gen_dict = {'param': True, 'binning': False}
                npc_dict = {}
                par_dict = {'param_families': families}
                bin_dict = {}
                
                fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
                
                results[vine_type] = {
                    'fit_success': True,
                    'vine_structure': vine.ind_vine if hasattr(vine, 'ind_vine') else None,
                    'r_matrix': (
                        vine.r_matrix.tolist()
                        if hasattr(vine, 'r_matrix') and vine.r_matrix is not None
                        else None
                    ),
                    'families': families,
                    'n_edges': sum(len(level) for level in vine.ind_vine) if hasattr(vine, 'ind_vine') else 0
                }
                
                # Sample from vine to test
                if hasattr(vine, 'sample'):
                    try:
                        samples = vine.sample(500)
                        sample_corr = compute_correlation_matrix(samples)
                        original_corr = compute_correlation_matrix(data)
                        
                        corr_error = np.mean(np.abs(sample_corr - original_corr))
                        results[vine_type]['correlation_preservation'] = {
                            'error': float(corr_error),
                            'original_correlation': original_corr.tolist(),
                            'sample_correlation': sample_corr.tolist()
                        }
                    except Exception as e:
                        self.logger.warning(f"Could not sample from {vine_type}: {e}")
                
            except Exception as e:
                self.logger.error(f"Failed to fit {vine_type}: {e}")
                results[vine_type] = {'fit_success': False, 'error': str(e)}
        
        return results
    
    def _analyze_probability_distributions(self, data: np.ndarray, vine_results: Dict) -> Dict[str, Any]:
        """Analyze probability distributions."""
        analysis_config = self.config.analysis_config
        
        results = {
            'marginal_analysis': {},
            'dependence_analysis': {},
            'goodness_of_fit': {}
        }
        
        # Marginal analysis
        for i in range(data.shape[1]):
            var_data = data[:, i]
            results['marginal_analysis'][f'variable_{i}'] = {
                'mean': float(np.mean(var_data)),
                'std': float(np.std(var_data)),
                'skewness': float(self._compute_skewness(var_data)),
                'kurtosis': float(self._compute_kurtosis(var_data)),
                'min': float(np.min(var_data)),
                'max': float(np.max(var_data))
            }
        
        # Dependence analysis
        correlation_matrix = compute_correlation_matrix(data)
        tau_matrix = compute_kendall_tau_matrix(data)
        
        results['dependence_analysis'] = {
            'correlation_matrix': correlation_matrix.tolist(),
            'kendall_tau_matrix': tau_matrix.tolist(),
            'max_correlation': float(np.max(np.abs(correlation_matrix - np.eye(data.shape[1])))),
            'mean_abs_correlation': float(np.mean(np.abs(correlation_matrix - np.eye(data.shape[1]))))
        }
        
        # Information measures
        if analysis_config.get('compute_information_measures', True):
            info_results = self._compute_information_measures(data)
            results['information_measures'] = info_results
        
        return results
    
    def _compute_information_measures(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute information-theoretic measures."""
        results = {}
        
        # Differential entropy estimation (simplified)
        for i in range(data.shape[1]):
            var_data = data[:, i]
            # Gaussian entropy estimate: 0.5 * log(2 * pi * e * var)
            var_entropy = 0.5 * np.log(2 * np.pi * np.e * np.var(var_data))
            results[f'entropy_var_{i}'] = float(var_entropy)
        
        # Mutual information estimation (simplified)
        if data.shape[1] >= 2:
            # Simple correlation-based MI estimate
            for i in range(data.shape[1]):
                for j in range(i + 1, data.shape[1]):
                    corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                    # Gaussian MI: -0.5 * log(1 - corr^2)
                    mi_estimate = -0.5 * np.log(1 - corr**2 + 1e-10)
                    results[f'mutual_info_{i}_{j}'] = float(mi_estimate)
        
        return results
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0.0
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0.0
    
    def _create_probability_plots(self, data: np.ndarray, vine_results: Dict, prob_results: Dict):
        """Create probability analysis plots."""
        plot_config = self.config.plot_config
        
        if not plot_config.get('create_plots', True):
            return
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Marginal distributions
        n_vars = data.shape[1]
        fig, axes = plt.subplots(2, (n_vars + 1) // 2, figsize=(12, 8))
        if n_vars == 1:
            axes = [axes]
        elif n_vars <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(n_vars):
            axes[i].hist(data[:, i], bins=50, alpha=0.7, density=True)
            axes[i].set_title(f'Variable {i} Distribution')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Density')
        
        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "marginal_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation heatmap
        if n_vars > 1:
            correlation_matrix = np.array(prob_results['dependence_analysis']['correlation_matrix'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f')
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Pairwise scatter plots
        if n_vars >= 2 and n_vars <= 5:
            fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 12))
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j:
                        axes[i, j].hist(data[:, i], bins=30, alpha=0.7)
                        axes[i, j].set_title(f'Var {i}')
                    else:
                        axes[i, j].scatter(data[:, j], data[:, i], alpha=0.5, s=1)
                        axes[i, j].set_xlabel(f'Variable {j}')
                        axes[i, j].set_ylabel(f'Variable {i}')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / "pairwise_plots.png", dpi=300, bbox_inches='tight')
            plt.close()


class EntropyAnalysisExperiment(BaseExperiment):
    """Experiment for entropy and information estimation."""
    
    def run(self) -> Dict[str, Any]:
        """Run entropy analysis experiment."""
        self.logger.info(f"Running entropy analysis experiment: {self.config.name}")
        
        results = {
            'experiment_type': 'entropy_analysis',
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__
        }
        
        # Generate data with varying entropy levels
        data_sets = self._generate_entropy_test_data()
        results['data_info'] = {name: {'shape': data.shape} for name, data in data_sets.items()}
        
        # Analyze entropy for each dataset
        entropy_results = {}
        for name, data in data_sets.items():
            self.logger.info(f"Analyzing entropy for dataset: {name}")
            entropy_results[name] = self._analyze_entropy(data)
        
        results['entropy_analysis'] = entropy_results
        
        # Compare entropy estimation methods
        comparison_results = self._compare_entropy_methods(data_sets)
        results['method_comparison'] = comparison_results
        
        # Create plots
        self._create_entropy_plots(data_sets, entropy_results, comparison_results)
        
        self.save_results(results)
        return results
    
    def _generate_entropy_test_data(self) -> Dict[str, np.ndarray]:
        """Generate datasets with different entropy characteristics."""
        data_config = self.config.data_config
        n_samples = data_config.get('n_samples', 1000)
        
        datasets = {}
        
        # Low entropy: highly correlated
        corr_high = np.array([[1.0, 0.9], [0.9, 1.0]])
        datasets['low_entropy'] = np.random.multivariate_normal([0, 0], corr_high, n_samples)
        
        # Medium entropy: moderate correlation
        corr_med = np.array([[1.0, 0.5], [0.5, 1.0]])
        datasets['medium_entropy'] = np.random.multivariate_normal([0, 0], corr_med, n_samples)
        
        # High entropy: independent
        datasets['high_entropy'] = np.random.multivariate_normal([0, 0], np.eye(2), n_samples)
        
        # Non-Gaussian: mixture
        mixture_data = np.vstack([
            np.random.multivariate_normal([-2, -2], np.eye(2) * 0.5, n_samples // 2),
            np.random.multivariate_normal([2, 2], np.eye(2) * 0.5, n_samples // 2)
        ])
        datasets['mixture'] = mixture_data
        
        return datasets
    
    def _analyze_entropy(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze entropy of dataset."""
        results = {}
        
        # Differential entropy estimates
        results['gaussian_entropy'] = self._gaussian_entropy_estimate(data)
        results['kernel_entropy'] = self._kernel_entropy_estimate(data)
        results['knn_entropy'] = self._knn_entropy_estimate(data)
        
        # Mutual information
        if data.shape[1] >= 2:
            results['mutual_information'] = self._estimate_mutual_information(data)
        
        # Entropy decomposition
        results['entropy_decomposition'] = self._entropy_decomposition(data)
        
        return results
    
    def _gaussian_entropy_estimate(self, data: np.ndarray) -> float:
        """Estimate entropy assuming Gaussian distribution."""
        cov = np.cov(data.T)
        det_cov = np.linalg.det(cov)
        if det_cov <= 0:
            return float('inf')
        
        d = data.shape[1]
        entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.log(det_cov)
        return float(entropy)
    
    def _kernel_entropy_estimate(self, data: np.ndarray) -> float:
        """Estimate entropy using kernel density estimation."""
        from scipy.spatial.distance import pdist, squareform
        
        n_samples, d = data.shape
        
        # Adaptive bandwidth selection
        distances = pdist(data)
        h = np.median(distances) / 2
        
        # Compute kernel density at each point
        dist_matrix = squareform(distances)
        kernel_vals = np.exp(-dist_matrix**2 / (2 * h**2))
        densities = np.sum(kernel_vals, axis=1) / (n_samples * (2 * np.pi * h**2)**(d/2))
        
        # Entropy estimate
        log_densities = np.log(densities + 1e-10)
        entropy = -np.mean(log_densities)
        
        return float(entropy)
    
    def _knn_entropy_estimate(self, data: np.ndarray, k: int = 5) -> float:
        """Estimate entropy using k-nearest neighbors."""
        from scipy.spatial import cKDTree
        
        n_samples, d = data.shape
        
        # Build KD-tree
        tree = cKDTree(data)
        
        # Find k-th nearest neighbor distances
        distances, _ = tree.query(data, k=k+1)  # +1 because first neighbor is the point itself
        knn_distances = distances[:, k]  # k-th nearest neighbor
        
        # Entropy estimate using Kozachenko-Leonenko estimator
        log_distances = np.log(knn_distances + 1e-10)
        entropy = d * np.mean(log_distances) + np.log(n_samples) + np.log(2) * d - np.log(k)
        
        return float(entropy)
    
    def _estimate_mutual_information(self, data: np.ndarray) -> Dict[str, float]:
        """Estimate mutual information between variables."""
        results = {}
        
        n_vars = data.shape[1]
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Gaussian MI estimate
                corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                mi_gaussian = -0.5 * np.log(1 - corr**2 + 1e-10)
                
                results[f'mi_{i}_{j}_gaussian'] = float(mi_gaussian)
                
                # KNN-based MI estimate (simplified)
                joint_entropy = self._knn_entropy_estimate(data[:, [i, j]])
                marginal_i = self._knn_entropy_estimate(data[:, [i]])
                marginal_j = self._knn_entropy_estimate(data[:, [j]])
                
                mi_knn = marginal_i + marginal_j - joint_entropy
                results[f'mi_{i}_{j}_knn'] = float(mi_knn)
        
        return results
    
    def _entropy_decomposition(self, data: np.ndarray) -> Dict[str, float]:
        """Decompose entropy into components."""
        results = {}
        
        # Total entropy
        total_entropy = self._kernel_entropy_estimate(data)
        results['total_entropy'] = total_entropy
        
        # Sum of marginal entropies
        marginal_sum = 0
        for i in range(data.shape[1]):
            marginal_entropy = self._kernel_entropy_estimate(data[:, [i]])
            results[f'marginal_entropy_{i}'] = marginal_entropy
            marginal_sum += marginal_entropy
        
        results['marginal_sum'] = marginal_sum
        
        # Interaction information
        results['interaction_information'] = marginal_sum - total_entropy
        
        return results
    
    def _compare_entropy_methods(self, data_sets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compare different entropy estimation methods."""
        methods = ['gaussian_entropy', 'kernel_entropy', 'knn_entropy']
        results = {}
        
        for method in methods:
            results[method] = {}
            for name, data in data_sets.items():
                if method == 'gaussian_entropy':
                    entropy = self._gaussian_entropy_estimate(data)
                elif method == 'kernel_entropy':
                    entropy = self._kernel_entropy_estimate(data)
                elif method == 'knn_entropy':
                    entropy = self._knn_entropy_estimate(data)
                
                results[method][name] = entropy
        
        return results
    
    def _create_entropy_plots(self, data_sets: Dict, entropy_results: Dict, comparison_results: Dict):
        """Create entropy analysis plots."""
        # Entropy comparison plot
        methods = list(comparison_results.keys())
        datasets = list(data_sets.keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(datasets))
        width = 0.25
        
        for i, method in enumerate(methods):
            values = [comparison_results[method][dataset] for dataset in datasets]
            ax.bar(x + i * width, values, width, label=method.replace('_', ' ').title())
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Entropy')
        ax.set_title('Entropy Estimation Method Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "entropy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Data visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (name, data) in enumerate(data_sets.items()):
            if i < 4:  # Only plot first 4 datasets
                axes[i].scatter(data[:, 0], data[:, 1], alpha=0.6, s=1)
                axes[i].set_title(f'{name.replace("_", " ").title()}')
                axes[i].set_xlabel('Variable 1')
                axes[i].set_ylabel('Variable 2')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "entropy_datasets.png", dpi=300, bbox_inches='tight')
        plt.close()


class TimeDependentExperiment(BaseExperiment):
    """Experiment for time-dependent vine copula modeling."""
    
    def run(self) -> Dict[str, Any]:
        """Run time-dependent experiment."""
        self.logger.info(f"Running time-dependent experiment: {self.config.name}")
        
        results = {
            'experiment_type': 'time_dependent',
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__
        }
        
        # Generate time-dependent data
        time_data, time_indices = self._generate_time_dependent_data()
        results['data_info'] = {
            'shape': time_data.shape,
            'n_time_steps': len(time_indices),
            'time_range': [float(time_indices.min()), float(time_indices.max())]
        }
        
        # Analyze time-varying correlations
        correlation_analysis = self._analyze_time_varying_correlations(time_data)
        results['correlation_analysis'] = correlation_analysis
        
        # Fit time-dependent vine model
        model_results = self._fit_time_dependent_model(time_data, time_indices)
        results['model_results'] = model_results
        
        # Evaluate model performance
        evaluation_results = self._evaluate_model_performance(time_data, time_indices, model_results)
        results['evaluation'] = evaluation_results
        
        # Create plots
        self._create_time_dependent_plots(time_data, time_indices, correlation_analysis, model_results)
        
        self.save_results(results)
        return results
    
    def _generate_time_dependent_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate time-dependent synthetic data."""
        time_config = self.config.time_config
        
        n_time_steps = time_config.get('n_time_steps', 50)
        n_samples_per_time = time_config.get('n_samples_per_time', 100)
        n_variables = time_config.get('n_variables', 3)
        correlation_evolution = time_config.get('correlation_evolution', 'sinusoidal')
        
        data, time_indices = generate_synthetic_time_series(
            n_time_steps=n_time_steps,
            n_samples=n_samples_per_time,
            n_variables=n_variables,
            correlation_evolution=correlation_evolution,
            seed=self.config.seed
        )
        
        return data, time_indices
    
    def _analyze_time_varying_correlations(self, time_data: np.ndarray) -> Dict[str, Any]:
        """Analyze how correlations change over time."""
        n_time_steps, n_samples, n_variables = time_data.shape
        
        correlations_over_time = []
        tau_over_time = []
        
        for t in range(n_time_steps):
            data_t = time_data[t]
            
            corr_matrix = compute_correlation_matrix(data_t)
            tau_matrix = compute_kendall_tau_matrix(data_t)
            
            correlations_over_time.append(corr_matrix)
            tau_over_time.append(tau_matrix)
        
        correlations_over_time = np.array(correlations_over_time)
        tau_over_time = np.array(tau_over_time)
        
        # Compute correlation statistics
        results = {
            'correlations_over_time': correlations_over_time.tolist(),
            'tau_over_time': tau_over_time.tolist(),
            'correlation_statistics': {}
        }
        
        # Statistics for each pair
        for i in range(n_variables):
            for j in range(i + 1, n_variables):
                pair_corrs = correlations_over_time[:, i, j]
                pair_taus = tau_over_time[:, i, j]
                
                results['correlation_statistics'][f'pair_{i}_{j}'] = {
                    'correlation': {
                        'mean': float(np.mean(pair_corrs)),
                        'std': float(np.std(pair_corrs)),
                        'min': float(np.min(pair_corrs)),
                        'max': float(np.max(pair_corrs)),
                        'trend': float(np.polyfit(range(n_time_steps), pair_corrs, 1)[0])
                    },
                    'kendall_tau': {
                        'mean': float(np.mean(pair_taus)),
                        'std': float(np.std(pair_taus)),
                        'min': float(np.min(pair_taus)),
                        'max': float(np.max(pair_taus)),
                        'trend': float(np.polyfit(range(n_time_steps), pair_taus, 1)[0])
                    }
                }
        
        return results
    
    def _fit_time_dependent_model(self, time_data: np.ndarray, time_indices: np.ndarray) -> Dict[str, Any]:
        """Fit time-dependent vine copula model."""
        time_config = self.config.time_config
        n_time_steps, n_samples_per_time, n_variables = time_data.shape
        
        try:
            # Create base vine and canonical time-conditioned model.
            base_vine = create_vine('c-vine', n_variables)
            hidden_dim = time_config.get('hidden_dim', 64)
            model = create_time_dependent_vine(
                base_vine=base_vine,
                hidden_dims=[int(hidden_dim)],
                device="cpu",
            )

            fit_info = model.fit_bandwidth_flow(
                train_data_by_time=time_data,
                time_points=time_indices,
                val_fraction=float(time_config.get('val_fraction', 0.2)),
                n_epochs=int(time_config.get('n_epochs', 20)),
                lr=float(time_config.get('learning_rate', 1e-2)),
                batch_time_steps=int(time_config.get('batch_time_steps', min(8, n_time_steps))),
                seed=int(self.config.seed if self.config.seed is not None else 0),
            )

            t_tensor = torch.tensor(time_indices, dtype=torch.float32)
            bandwidths = model.get_bandwidths_over_time(t_tensor)  # [T, E]
            bw_np = bandwidths.detach().cpu().numpy()

            bandwidth_metrics: Dict[str, Any] = {}
            n_edges = int(bw_np.shape[1]) if bw_np.ndim == 2 else 0
            for edge_idx in range(n_edges):
                vals = bw_np[:, edge_idx]
                key = f"edge_{edge_idx}"
                bandwidth_metrics[f"{key}_mean_bw"] = float(np.mean(vals))
                bandwidth_metrics[f"{key}_std_bw"] = float(np.std(vals))
                bandwidth_metrics[f"{key}_min_bw"] = float(np.min(vals))
                bandwidth_metrics[f"{key}_max_bw"] = float(np.max(vals))
                if vals.shape[0] > 1:
                    bandwidth_metrics[f"{key}_temporal_variation"] = float(np.mean(np.abs(np.diff(vals))))

            results = {
                'model_created': True,
                'n_edges': int(n_edges),
                'hidden_dim': hidden_dim,
                'bandwidth_shape': list(bandwidths.shape),
                'fit_info': fit_info,
                'bandwidth_statistics': {
                    'mean': float(bandwidths.mean().detach()),
                    'std': float(bandwidths.std().detach()),
                    'min': float(bandwidths.min().detach()),
                    'max': float(bandwidths.max().detach())
                },
                'bandwidth_metrics': bandwidth_metrics,
            }
            
            # Store model for evaluation
            self._trained_model = model
            
        except Exception as e:
            self.logger.error(f"Failed to fit time-dependent model: {e}")
            results = {
                'model_created': False,
                'error': str(e)
            }
            self._trained_model = None
        
        return results
    
    def _evaluate_model_performance(self, time_data: np.ndarray, time_indices: np.ndarray, 
                                  model_results: Dict) -> Dict[str, Any]:
        """Evaluate time-dependent model performance."""
        if not model_results.get('model_created', False) or self._trained_model is None:
            return {'evaluation_performed': False, 'reason': 'No trained model available'}
        
        model = self._trained_model

        try:
            # Test forward pass on sample data
            sample_data = torch.tensor(time_data[0][:10], dtype=torch.float32)
            t0 = float(np.asarray(time_indices).reshape(-1)[0])
            sample_times = torch.full((sample_data.shape[0],), t0, dtype=torch.float32)
            
            # Compute loss
            with torch.no_grad():
                nll_vals = model(sample_data, sample_times)
                nll_loss = torch.mean(nll_vals)
            
            # Bandwidth evolution analysis
            t_tensor = torch.tensor(time_indices, dtype=torch.float32)
            bandwidths = model.get_bandwidths_over_time(t_tensor)
            
            # Compute temporal variation
            if bandwidths.shape[0] > 1:
                bandwidth_diff = bandwidths[1:] - bandwidths[:-1]
                temporal_variation = torch.mean(torch.norm(bandwidth_diff, dim=1))
            else:
                temporal_variation = torch.tensor(0.0)
            entropy_loss = nll_loss + 0.1 * temporal_variation

            results = {
                'evaluation_performed': True,
                'losses': {
                    'negative_log_likelihood': float(nll_loss),
                    'entropy_based_loss': float(entropy_loss)
                },
                'bandwidth_analysis': {
                    'temporal_variation': float(temporal_variation.detach()),
                    'smoothness_score': float((1.0 / (1.0 + temporal_variation)).detach())  # Higher is smoother
                },
                'model_complexity': {
                    'n_parameters': sum(p.numel() for p in model.parameters()),
                    'n_edges': int(bandwidths.shape[1] if bandwidths.ndim == 2 else 0)
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Primary model evaluation failed, using fallback summary: {e}")
            try:
                t_tensor = torch.tensor(time_indices, dtype=torch.float32)
                bandwidths = model.get_bandwidths_over_time(t_tensor)

                if bandwidths.shape[0] > 1:
                    bw_diff = bandwidths[1:] - bandwidths[:-1]
                    temporal_variation = float(
                        torch.mean(torch.norm(bw_diff.reshape(bw_diff.shape[0], -1), dim=1)).detach()
                    )
                else:
                    temporal_variation = 0.0

                results = {
                    'evaluation_performed': True,
                    'losses': {
                        'negative_log_likelihood': None,
                        'entropy_based_loss': None
                    },
                    'bandwidth_analysis': {
                        'temporal_variation': temporal_variation,
                        'smoothness_score': float(1.0 / (1.0 + temporal_variation))
                    },
                    'model_complexity': {
                        'n_parameters': sum(p.numel() for p in model.parameters()),
                        'n_edges': int(bandwidths.shape[1] if bandwidths.ndim == 2 else 0)
                    },
                    'note': f"Fallback evaluation used due to: {e}"
                }
            except Exception as e2:
                results = {
                    'evaluation_performed': False,
                    'error': f"Primary: {e}; Fallback: {e2}"
                }
        
        return results
    
    def _create_time_dependent_plots(self, time_data: np.ndarray, time_indices: np.ndarray,
                                   correlation_analysis: Dict, model_results: Dict):
        """Create time-dependent analysis plots."""
        n_time_steps, n_samples_per_time, n_variables = time_data.shape
        
        # Correlation evolution plot
        if n_variables >= 2:
            correlations = np.array(correlation_analysis['correlations_over_time'])
            
            plt.figure(figsize=(12, 8))
            
            # Plot correlation evolution for each pair
            pair_count = 0
            colors = plt.cm.tab10(np.linspace(0, 1, n_variables * (n_variables - 1) // 2))
            
            for i in range(n_variables):
                for j in range(i + 1, n_variables):
                    pair_corrs = correlations[:, i, j]
                    plt.plot(time_indices, pair_corrs, 
                           label=f'Var {i}-{j}', color=colors[pair_count], linewidth=2)
                    pair_count += 1
            
            plt.xlabel('Time')
            plt.ylabel('Correlation')
            plt.title('Time-Varying Correlations')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / "correlation_evolution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Bandwidth evolution plot (if model was trained)
        if model_results.get('model_created', False) and hasattr(self, '_trained_model'):
            try:
                model = self._trained_model
                t_tensor = torch.tensor(time_indices, dtype=torch.float32)
                bandwidths = model.get_bandwidths_over_time(t_tensor)
                
                plt.figure(figsize=(12, 6))
                
                # Plot bandwidth evolution for first few edges.
                n_edges = int(bandwidths.shape[1]) if bandwidths.ndim == 2 else 0
                for edge_idx in range(min(3, n_edges)):
                    bw_values = bandwidths[:, edge_idx].detach().cpu().numpy()
                    plt.plot(time_indices, bw_values, label=f'Edge {edge_idx}', linewidth=2)
                
                plt.xlabel('Time')
                plt.ylabel('Bandwidth')
                plt.title('Bandwidth Evolution Over Time')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.output_dir / "plots" / "bandwidth_evolution.png", dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                self.logger.warning(f"Could not create bandwidth plot: {e}")
        
        # Data visualization over time (sample time steps)
        sample_times = np.linspace(0, n_time_steps - 1, min(4, n_time_steps)).astype(int)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, t in enumerate(sample_times):
            if i < 4 and n_variables >= 2:
                data_t = time_data[t]
                axes[i].scatter(data_t[:, 0], data_t[:, 1], alpha=0.6, s=1)
                axes[i].set_title(f'Time Step {t}')
                axes[i].set_xlabel('Variable 0')
                axes[i].set_ylabel('Variable 1')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "data_over_time.png", dpi=300, bbox_inches='tight')
        plt.close()


class NeuripsSimulationsExperiment(BaseExperiment):
    """NeurIPS-oriented synthetic simulation suite.

    Generates paper-defining scenarios that stress-test higher-order and time-varying dependence:
    - beyond pairwise (conditional dependence with near-zero pairwise correlation)
    - dynamic tail dependence at stable second-order summaries
    - matched Kendall-tau with switching tail asymmetry (Clayton ↔ Gumbel)
    - hub/root switching recovered via structure optimization
    """

    def run(self) -> Dict[str, Any]:
        self.logger.info(f"Running NeurIPS simulation suite: {self.config.name}")

        scenarios = self.config.analysis_config.get("scenarios", [])
        if not isinstance(scenarios, list) or not scenarios:
            # Default suite if not specified in YAML.
            scenarios = [
                {"name": "multiplicative_triplet"},
                {"name": "dynamic_tail_df"},
                {"name": "tail_switch"},
                {"name": "hub_switch"},
            ]

        results = run_neurips_simulation_suite(
            output_dir=self.output_dir,
            seed=int(self.config.seed or 0),
            scenarios=scenarios,
        )

        # Add top-level metadata for consistency with other experiments.
        results["timestamp"] = datetime.now().isoformat()
        results["config"] = self.config.__dict__

        self.save_results(results)
        return results


class ExperimentRunner:
    """Main class for running experiments from YAML configurations."""
    
    def __init__(self):
        """Initialize experiment runner."""
        self.logger = logging.getLogger("DVC.experiments.runner")
    
    def run_from_config(self, config_path: str) -> Dict[str, Any]:
        """Run experiment from YAML configuration file."""
        self.logger.info(f"Loading experiment configuration from: {config_path}")
        
        # Load configuration
        config = ExperimentConfig.from_yaml(config_path)
        
        # Determine experiment type
        experiment_type = config.analysis_config.get('experiment_type', 'probability_analysis')
        
        # Create appropriate experiment instance
        if experiment_type == 'probability_analysis':
            experiment = ProbabilityAnalysisExperiment(config)
        elif experiment_type == 'entropy_analysis':
            experiment = EntropyAnalysisExperiment(config)
        elif experiment_type == 'time_dependent':
            experiment = TimeDependentExperiment(config)
        elif experiment_type == 'neurips_simulations':
            experiment = NeuripsSimulationsExperiment(config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        # Run experiment
        self.logger.info(f"Running {experiment_type} experiment: {config.name}")
        results = experiment.run()
        
        self.logger.info(f"Experiment completed successfully")
        return results
    
    def create_example_configs(self, output_dir: str = "example_configs"):
        """Create example YAML configuration files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Probability analysis config
        prob_config = ExperimentConfig(
            name="probability_analysis_example",
            description="Example probability analysis experiment",
            output_dir="results/probability_analysis",
            seed=42,
            data_config={
                'type': 'synthetic',
                'n_samples': 2000,
                'n_variables': 4,
                'correlation_type': 'moderate'
            },
            vine_config={
                'types': ['c-vine', 'd-vine', 'r-vine'],
                'families': ['independence', 'gaussian', 'clayton', 'frank']
            },
            analysis_config={
                'experiment_type': 'probability_analysis',
                'compute_information_measures': True
            },
            plot_config={
                'create_plots': True
            }
        )
        prob_config.to_yaml(output_path / "probability_analysis.yaml")
        
        # Entropy analysis config
        entropy_config = ExperimentConfig(
            name="entropy_analysis_example",
            description="Example entropy analysis experiment",
            output_dir="results/entropy_analysis",
            seed=42,
            data_config={
                'n_samples': 1500
            },
            analysis_config={
                'experiment_type': 'entropy_analysis'
            },
            plot_config={
                'create_plots': True
            }
        )
        entropy_config.to_yaml(output_path / "entropy_analysis.yaml")
        
        # Time-dependent config
        time_config = ExperimentConfig(
            name="time_dependent_example",
            description="Example time-dependent vine copula experiment",
            output_dir="results/time_dependent",
            seed=42,
            time_config={
                'n_time_steps': 100,
                'n_samples_per_time': 200,
                'n_variables': 3,
                'correlation_evolution': 'sinusoidal',
                'hidden_dim': 64
            },
            analysis_config={
                'experiment_type': 'time_dependent'
            },
            plot_config={
                'create_plots': True
            }
        )
        time_config.to_yaml(output_path / "time_dependent.yaml")
        
        self.logger.info(f"Example configuration files created in: {output_path}")


def main():
    """Main function for running experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run DVC experiments from YAML config')
    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('--create-examples', action='store_true', 
                       help='Create example configuration files')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    runner = ExperimentRunner()
    
    if args.create_examples:
        runner.create_example_configs()
        print("Example configuration files created in 'example_configs/' directory")
        return
    
    # Run experiment
    try:
        results = runner.run_from_config(args.config)
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {results.get('config', {}).get('output_dir', 'results')}")
    except Exception as e:
        print(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()


__all__ = [
    'ExperimentConfig',
    'BaseExperiment', 
    'ProbabilityAnalysisExperiment',
    'EntropyAnalysisExperiment',
    'TimeDependentExperiment',
    'ExperimentRunner'
]
