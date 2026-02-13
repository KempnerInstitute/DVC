"""
Experiment Runner for DVC Package

Main engine for executing comprehensive vine copula experiments,
including parameter sweeps, method comparisons, and benchmarking.
"""

import numpy as np
import pandas as pd
import torch
import yaml
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

from ..core.vine_factory import create_vine, VineType
from ..core.vine_model import fit_vine
from ..core.info_estimation import vine_entropy, mutual_information
from ..optimization.structure import optimize_vine_structure
from ..time.models import create_time_dependent_vine
from .benchmarks import generate_benchmark_data, BenchmarkDataset

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for vine copula experiments."""
    
    # Experiment metadata
    name: str
    description: str
    output_dir: str
    
    # Data configuration
    data_config: Dict[str, Any]
    
    # Vine configuration
    vine_types: List[str]
    copula_families: List[str]
    dimensions: List[int]
    
    # Optimization configuration
    optimization_methods: List[str]
    optimization_enabled: bool = True
    
    # Time-dependent configuration
    time_dependent: bool = False
    time_config: Optional[Dict[str, Any]] = None
    
    # Evaluation configuration
    evaluation_metrics: List[str] = field(default_factory=list)
    n_monte_carlo_samples: int = 1000
    n_bootstrap_runs: int = 10
    
    # Computational configuration
    n_parallel_jobs: int = 1
    random_seed: int = 42
    device: str = 'auto'
    
    # Output configuration
    save_models: bool = True
    save_detailed_results: bool = True
    create_plots: bool = True


class ExperimentRunner:
    """
    Main experiment runner for comprehensive vine copula analysis.
    
    Manages experiment execution, result collection, and output generation.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Set up output directory
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Set random seed
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
    
    def _setup_logging(self):
        """Set up experiment-specific logging."""
        log_path = self.output_path / "experiment.log"
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.info(f"Experiment logging initialized: {log_path}")
    
    def run(self) -> Dict[str, Any]:
        """Run the complete experiment."""
        
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Description: {self.config.description}")
        
        self.start_time = time.time()
        
        try:
            # Generate or load data
            datasets = self._prepare_datasets()
            
            # Run experiments for each configuration
            experiment_results = self._run_experiment_grid(datasets)
            
            # Aggregate and analyze results
            analysis_results = self._analyze_results(experiment_results)
            
            # Generate visualizations
            if self.config.create_plots:
                visualization_results = self._create_visualizations(analysis_results)
            else:
                visualization_results = {}
            
            # Compile final results
            self.results = {
                'config': asdict(self.config),
                'experiment_results': experiment_results,
                'analysis': analysis_results,
                'visualizations': visualization_results,
                'metadata': {
                    'start_time': self.start_time,
                    'end_time': time.time(),
                    'duration': time.time() - self.start_time
                }
            }
            
            # Save results
            self._save_results()
            
            logger.info(f"Experiment completed successfully in {self.results['metadata']['duration']:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
        
        finally:
            self.end_time = time.time()
    
    def _prepare_datasets(self) -> List[BenchmarkDataset]:
        """Generate or load datasets for experiments."""
        
        logger.info("Preparing datasets")
        datasets = []
        
        data_config = self.config.data_config
        
        if data_config.get('type') == 'benchmark':
            # Generate benchmark datasets
            for dim in self.config.dimensions:
                for scenario in data_config.get('scenarios', ['gaussian', 'mixed']):
                    dataset = generate_benchmark_data(
                        scenario=scenario,
                        dimension=dim,
                        n_samples=data_config.get('n_samples', 1000),
                        correlation_strength=data_config.get('correlation_strength', 0.5),
                        noise_level=data_config.get('noise_level', 0.1)
                    )
                    datasets.append(dataset)
        
        elif data_config.get('type') == 'file':
            # Load datasets from files
            for file_path in data_config.get('files', []):
                dataset = self._load_dataset_from_file(file_path)
                datasets.append(dataset)
        
        elif data_config.get('type') == 'synthetic':
            # Generate synthetic datasets
            for dim in self.config.dimensions:
                dataset = self._generate_synthetic_dataset(dim, data_config)
                datasets.append(dataset)
        
        else:
            raise ValueError(f"Unknown data type: {data_config.get('type')}")
        
        logger.info(f"Prepared {len(datasets)} datasets")
        return datasets
    
    def _run_experiment_grid(self, datasets: List[BenchmarkDataset]) -> Dict[str, Any]:
        """Run experiments across all parameter combinations."""
        
        logger.info("Running experiment grid")
        
        # Generate all experiment configurations
        experiment_configs = self._generate_experiment_configs(datasets)
        
        logger.info(f"Total experiment configurations: {len(experiment_configs)}")
        
        # Run experiments
        if self.config.n_parallel_jobs > 1:
            results = self._run_parallel_experiments(experiment_configs)
        else:
            results = self._run_sequential_experiments(experiment_configs)
        
        return results
    
    def _generate_experiment_configs(self, datasets: List[BenchmarkDataset]) -> List[Dict[str, Any]]:
        """Generate all experiment parameter combinations."""
        
        configs = []
        
        for dataset in datasets:
            for vine_type in self.config.vine_types:
                for optimization_method in self.config.optimization_methods:
                    config = {
                        'dataset': dataset,
                        'vine_type': vine_type,
                        'optimization_method': optimization_method,
                        'copula_families': self.config.copula_families,
                        'optimization_enabled': self.config.optimization_enabled,
                        'time_dependent': self.config.time_dependent,
                        'time_config': self.config.time_config,
                        'evaluation_metrics': self.config.evaluation_metrics,
                        'n_monte_carlo_samples': self.config.n_monte_carlo_samples
                    }
                    configs.append(config)
        
        return configs
    
    def _run_sequential_experiments(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run experiments sequentially."""
        
        results = {}
        
        for i, config in enumerate(configs):
            logger.info(f"Running experiment {i+1}/{len(configs)}")
            
            try:
                result = self._run_single_experiment(config)
                
                # Create unique key for this configuration
                key = self._create_experiment_key(config)
                results[key] = result
                
            except Exception as e:
                logger.error(f"Experiment {i+1} failed: {e}")
                continue
        
        return results
    
    def _run_parallel_experiments(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run experiments in parallel."""
        
        results = {}
        
        with ProcessPoolExecutor(max_workers=self.config.n_parallel_jobs) as executor:
            # Submit all jobs
            future_to_config = {
                executor.submit(self._run_single_experiment, config): config
                for config in configs
            }
            
            # Collect results
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                
                try:
                    result = future.result()
                    key = self._create_experiment_key(config)
                    results[key] = result
                    
                    logger.info(f"Completed experiment: {key}")
                    
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    continue
        
        return results
    
    def _run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        
        dataset = config['dataset']
        vine_type = config['vine_type']
        optimization_method = config['optimization_method']
        
        experiment_start_time = time.time()
        
        # Create vine
        vine = create_vine(
            vine_type=vine_type,
            vine_depth=dataset.data.shape[1],
            families=config['copula_families']
        )
        
        # Optimize structure if enabled
        optimization_results = {}
        if config['optimization_enabled']:
            opt_result = optimize_vine_structure(
                data=dataset.data,
                vine_type=vine_type,
                method=optimization_method,
                verbose=False
            )
            vine = opt_result.best_vine
            optimization_results = {
                'best_score': opt_result.best_score,
                'iterations': opt_result.iterations,
                'method': opt_result.method,
                'convergence_info': opt_result.convergence_info
            }
        
        # Fit vine parameters
        fitting_start_time = time.time()
        
        gen_dict = {'param': True, 'binning': False, 'fitted': False}
        npc_dict = {}
        par_dict = {'param_families': config['copula_families']}
        bin_dict = {}
        
        fit_vine(vine, dataset.data, gen_dict, npc_dict, par_dict, bin_dict)
        
        fitting_time = time.time() - fitting_start_time
        
        # Evaluate vine
        evaluation_results = self._evaluate_vine(vine, dataset, config)
        
        # Time-dependent analysis if enabled
        time_dependent_results = {}
        if config['time_dependent'] and hasattr(dataset, 'time_data'):
            time_dependent_results = self._analyze_time_dependent(vine, dataset, config)
        
        experiment_time = time.time() - experiment_start_time
        
        return {
            'dataset_info': {
                'name': dataset.name,
                'shape': dataset.data.shape,
                'scenario': dataset.scenario
            },
            'vine_config': {
                'type': vine_type,
                'families': config['copula_families'],
                'optimization_method': optimization_method if config['optimization_enabled'] else None
            },
            'optimization_results': optimization_results,
            'fitting_time': fitting_time,
            'evaluation_results': evaluation_results,
            'time_dependent_results': time_dependent_results,
            'total_time': experiment_time,
            'vine_model': vine if self.config.save_models else None
        }
    
    def _evaluate_vine(self, vine, dataset: BenchmarkDataset, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate fitted vine model."""
        
        results = {}
        
        info_dict = {
            'alpha': 0.05,
            'cases': config['n_monte_carlo_samples'],
            'iterations': 10
        }
        
        # Entropy estimation
        if 'entropy' in config['evaluation_metrics']:
            try:
                entropy = vine_entropy(vine, info_dict)
                results['entropy'] = entropy
            except Exception as e:
                logger.warning(f"Entropy estimation failed: {e}")
                results['entropy'] = None
        
        # Mutual information
        if 'mutual_information' in config['evaluation_metrics']:
            try:
                # Compute MI between all pairs of variables
                d = dataset.data.shape[1]
                mi_matrix = np.zeros((d, d))
                
                for i in range(d):
                    for j in range(i + 1, d):
                        mi = mutual_information(vine, [i], [j], info_dict)
                        mi_matrix[i, j] = mi
                        mi_matrix[j, i] = mi
                
                results['mutual_information_matrix'] = mi_matrix.tolist()
                results['average_mutual_information'] = np.mean(mi_matrix[np.triu_indices(d, k=1)])
                
            except Exception as e:
                logger.warning(f"Mutual information estimation failed: {e}")
                results['mutual_information_matrix'] = None
                results['average_mutual_information'] = None
        
        # Log-likelihood on test data
        if 'log_likelihood' in config['evaluation_metrics']:
            try:
                # Use a portion of data for testing
                n_test = min(500, dataset.data.shape[0] // 4)
                test_data = dataset.data[-n_test:]
                
                if hasattr(vine, 'logpdf'):
                    test_tensor = torch.from_numpy(test_data).float()
                    log_probs = vine.logpdf(test_tensor)
                    avg_log_likelihood = torch.mean(log_probs).item()
                else:
                    avg_log_likelihood = None
                
                results['average_log_likelihood'] = avg_log_likelihood
                
            except Exception as e:
                logger.warning(f"Log-likelihood evaluation failed: {e}")
                results['average_log_likelihood'] = None
        
        # Correlation preservation
        if 'correlation_preservation' in config['evaluation_metrics']:
            try:
                # Sample from fitted vine and compare correlations
                n_samples = min(1000, dataset.data.shape[0])
                
                if hasattr(vine, 'sample'):
                    samples = vine.sample(n_samples)
                    
                    # Compute correlation matrices
                    original_corr = np.corrcoef(dataset.data.T)
                    sample_corr = np.corrcoef(samples.T)
                    
                    # Compute correlation preservation metric
                    corr_diff = np.abs(original_corr - sample_corr)
                    corr_preservation = 1.0 - np.mean(corr_diff[np.triu_indices(original_corr.shape[0], k=1)])
                    
                    results['correlation_preservation'] = corr_preservation
                    results['correlation_rmse'] = np.sqrt(np.mean(corr_diff**2))
                else:
                    results['correlation_preservation'] = None
                    results['correlation_rmse'] = None
                
            except Exception as e:
                logger.warning(f"Correlation preservation evaluation failed: {e}")
                results['correlation_preservation'] = None
                results['correlation_rmse'] = None
        
        return results
    
    def _analyze_time_dependent(self, vine, dataset: BenchmarkDataset, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze time-dependent aspects if applicable."""
        
        # This is a placeholder for time-dependent analysis
        # In a full implementation, you'd:
        # 1. Create time-dependent vine model
        # 2. Train on time series data
        # 3. Analyze entropy evolution
        # 4. Compute time-dependent mutual information
        
        return {
            'time_dependent_entropy': None,
            'entropy_evolution': None,
            'time_dependent_mi': None
        }
    
    def _create_experiment_key(self, config: Dict[str, Any]) -> str:
        """Create a unique key for experiment configuration."""
        
        dataset_name = config['dataset'].name
        vine_type = config['vine_type']
        opt_method = config['optimization_method'] if config['optimization_enabled'] else 'no_opt'
        
        return f"{dataset_name}_{vine_type}_{opt_method}"
    
    def _analyze_results(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and summarize experiment results."""
        
        logger.info("Analyzing experiment results")
        
        analysis = {
            'summary_statistics': {},
            'method_comparison': {},
            'performance_rankings': {},
            'statistical_tests': {}
        }
        
        # Extract metrics for analysis
        metrics_data = {}
        for exp_key, result in experiment_results.items():
            eval_results = result.get('evaluation_results', {})
            
            for metric, value in eval_results.items():
                if value is not None and np.isfinite(value):
                    if metric not in metrics_data:
                        metrics_data[metric] = []
                    
                    metrics_data[metric].append({
                        'experiment': exp_key,
                        'value': value,
                        'vine_type': result['vine_config']['type'],
                        'optimization': result['vine_config']['optimization_method'],
                        'dataset': result['dataset_info']['name']
                    })
        
        # Compute summary statistics
        for metric, data_list in metrics_data.items():
            values = [d['value'] for d in data_list]
            
            analysis['summary_statistics'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'count': len(values)
            }
        
        # Method comparison
        vine_types = set()
        opt_methods = set()
        
        for result in experiment_results.values():
            vine_types.add(result['vine_config']['type'])
            if result['vine_config']['optimization_method']:
                opt_methods.add(result['vine_config']['optimization_method'])
        
        analysis['method_comparison']['vine_types'] = list(vine_types)
        analysis['method_comparison']['optimization_methods'] = list(opt_methods)
        
        # Performance rankings (simplified)
        if 'entropy' in metrics_data:
            entropy_data = metrics_data['entropy']
            entropy_ranking = sorted(entropy_data, key=lambda x: x['value'], reverse=True)
            analysis['performance_rankings']['entropy'] = [
                {'experiment': d['experiment'], 'value': d['value']} 
                for d in entropy_ranking[:10]
            ]
        
        return analysis
    
    def _create_visualizations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualizations for experiment results."""
        
        logger.info("Creating visualizations")
        
        viz_results = {}
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            viz_path = self.output_path / "visualizations"
            viz_path.mkdir(exist_ok=True)
            
            # Summary plots would go here
            # This is a placeholder for actual visualization code
            
            viz_results['plots_created'] = True
            viz_results['output_directory'] = str(viz_path)
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping visualizations")
            viz_results['plots_created'] = False
            viz_results['error'] = "Visualization libraries not available"
        
        return viz_results
    
    def _save_results(self):
        """Save experiment results to files."""
        
        logger.info("Saving experiment results")
        
        # Save main results as JSON
        results_path = self.output_path / "results.json"
        
        # Convert results to JSON-serializable format
        json_results = self._prepare_results_for_json(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save detailed results as pickle if requested
        if self.config.save_detailed_results:
            pickle_path = self.output_path / "results_detailed.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.results, f)
        
        # Save configuration
        config_path = self.output_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Results saved to {self.output_path}")
    
    def _prepare_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for JSON serialization."""
        
        # Remove non-serializable objects (like vine models)
        json_results = {}
        
        for key, value in results.items():
            if key == 'experiment_results':
                json_results[key] = {}
                for exp_key, exp_result in value.items():
                    json_results[key][exp_key] = {
                        k: v for k, v in exp_result.items() 
                        if k != 'vine_model'  # Exclude vine models
                    }
            else:
                json_results[key] = value
        
        return json_results
    
    def _load_dataset_from_file(self, file_path: str) -> BenchmarkDataset:
        """Load dataset from file."""
        
        data = np.load(file_path) if file_path.endswith('.npy') else pd.read_csv(file_path).values
        
        return BenchmarkDataset(
            name=Path(file_path).stem,
            data=data,
            scenario='file_loaded',
            metadata={'file_path': file_path}
        )
    
    def _generate_synthetic_dataset(self, dimension: int, data_config: Dict[str, Any]) -> BenchmarkDataset:
        """Generate synthetic dataset."""
        
        n_samples = data_config.get('n_samples', 1000)
        correlation_type = data_config.get('correlation_type', 'random')
        
        # Generate correlation matrix
        if correlation_type == 'random':
            # Random correlation matrix
            A = np.random.randn(dimension, dimension)
            corr_matrix = np.corrcoef(A)
        elif correlation_type == 'ar1':
            # AR(1) correlation structure
            rho = data_config.get('ar1_coefficient', 0.5)
            corr_matrix = np.array([[rho**abs(i-j) for j in range(dimension)] for i in range(dimension)])
        else:
            # Identity (independent)
            corr_matrix = np.eye(dimension)
        
        # Generate multivariate normal data
        data = np.random.multivariate_normal(
            mean=np.zeros(dimension),
            cov=corr_matrix,
            size=n_samples
        )
        
        return BenchmarkDataset(
            name=f"synthetic_{dimension}d_{correlation_type}",
            data=data,
            scenario=f"synthetic_{correlation_type}",
            metadata={
                'correlation_matrix': corr_matrix.tolist(),
                'correlation_type': correlation_type
            }
        )


def run_experiment(config: Union[Dict[str, Any], ExperimentConfig],
                  output_dir: Optional[str] = None,
                  n_runs: int = 1,
                  seed: int = 42,
                  verbose: bool = False) -> Dict[str, Any]:
    """
    Run a vine copula experiment from configuration.
    
    Parameters
    ----------
    config : dict or ExperimentConfig
        Experiment configuration
    output_dir : str, optional
        Output directory (overrides config if provided)
    n_runs : int
        Number of experimental runs
    seed : int
        Random seed
    verbose : bool
        Verbose output
        
    Returns
    -------
    dict
        Experiment results
    """
    
    # Convert config to ExperimentConfig if needed
    if isinstance(config, dict):
        if output_dir:
            config['output_dir'] = output_dir
        config['random_seed'] = seed
        exp_config = ExperimentConfig(**config)
    else:
        exp_config = config
        if output_dir:
            exp_config.output_dir = output_dir
        exp_config.random_seed = seed
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run multiple experimental runs if requested
    all_results = []
    
    for run_idx in range(n_runs):
        logger.info(f"Starting experimental run {run_idx + 1}/{n_runs}")
        
        # Create run-specific output directory
        if n_runs > 1:
            run_output_dir = Path(exp_config.output_dir) / f"run_{run_idx + 1}"
            exp_config.output_dir = str(run_output_dir)
        
        # Adjust seed for each run
        exp_config.random_seed = seed + run_idx
        
        # Run experiment
        runner = ExperimentRunner(exp_config)
        run_results = runner.run()
        
        all_results.append(run_results)
    
    # Aggregate results if multiple runs
    if n_runs > 1:
        # Combine results from all runs
        aggregated_results = {
            'n_runs': n_runs,
            'individual_runs': all_results,
            'aggregated_analysis': _aggregate_multiple_runs(all_results)
        }
        return aggregated_results
    else:
        return all_results[0]


def _aggregate_multiple_runs(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results from multiple experimental runs."""
    
    # This is a placeholder for aggregation logic
    # In a full implementation, you'd compute statistics across runs
    
    return {
        'summary': 'Multiple run aggregation not fully implemented',
        'n_runs': len(results_list)
    }
