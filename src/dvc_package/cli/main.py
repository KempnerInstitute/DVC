"""
Main CLI Entry Points for DVC Package

Provides command-line interfaces for common vine copula operations.
"""

import click
import numpy as np
import pandas as pd
import torch
import yaml
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.vine_factory import create_vine, VineType
from ..core.vine_model import fit_vine
from ..core.info_estimation import vine_entropy, mutual_information
from ..optimization.structure import optimize_vine_structure
from ..time.models import create_time_dependent_vine
from ..experiments.runner import run_experiment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.command()
@click.option('--data', '-d', required=True, help='Path to data file (CSV, NPY, or PKL)')
@click.option('--vine-type', '-t', default='r-vine', 
              type=click.Choice(['c-vine', 'd-vine', 'r-vine']),
              help='Type of vine copula to fit')
@click.option('--families', '-f', default='gaussian,clayton,frank',
              help='Comma-separated list of copula families')
@click.option('--optimize', '-o', is_flag=True, default=False,
              help='Optimize vine structure')
@click.option('--optimization-method', default='sequential',
              type=click.Choice(['sequential', 'genetic', 'entropy', 'hybrid']),
              help='Optimization method for structure selection')
@click.option('--output', '-O', required=True, help='Output path for fitted model')
@click.option('--config', '-c', help='YAML configuration file')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Verbose output')
def fit_vine_cli(data: str, vine_type: str, families: str, optimize: bool,
                optimization_method: str, output: str, config: Optional[str],
                verbose: bool):
    """Fit a vine copula model to data."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration if provided
        config_dict = {}
        if config:
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
        
        # Load data
        logger.info(f"Loading data from {data}")
        data_array = _load_data(data)
        logger.info(f"Data shape: {data_array.shape}")
        
        # Parse copula families
        family_list = [f.strip() for f in families.split(',')]
        
        # Create vine
        logger.info(f"Creating {vine_type} with families: {family_list}")
        vine = create_vine(
            vine_type=vine_type,
            vine_depth=data_array.shape[1],
            families=family_list,
            **config_dict.get('vine_params', {})
        )
        
        # Optimize structure if requested
        if optimize:
            logger.info(f"Optimizing vine structure using {optimization_method} method")
            opt_result = optimize_vine_structure(
                data=data_array,
                vine_type=vine_type,
                method=optimization_method,
                verbose=verbose,
                **config_dict.get('optimization_params', {})
            )
            vine = opt_result.best_vine
            logger.info(f"Optimization completed. Best score: {opt_result.best_score:.4f}")
        
        # Fit vine parameters
        logger.info("Fitting vine parameters")
        
        # Set up fitting parameters
        gen_dict = config_dict.get('gen_dict', {
            'param': True,
            'binning': False,
            'fitted': False
        })
        
        npc_dict = config_dict.get('npc_dict', {})
        
        par_dict = config_dict.get('par_dict', {
            'param_families': family_list
        })
        
        bin_dict = config_dict.get('bin_dict', {})
        
        # Fit the vine
        fit_vine(vine, data_array, gen_dict, npc_dict, par_dict, bin_dict)
        
        # Save fitted model
        logger.info(f"Saving fitted vine to {output}")
        _save_model(vine, output)
        
        # Print summary
        if verbose:
            _print_vine_summary(vine, data_array)
        
        logger.info("Vine fitting completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in vine fitting: {e}")
        raise click.ClickException(str(e))


@click.command()
@click.option('--model', '-m', required=True, help='Path to fitted vine model')
@click.option('--data', '-d', help='Path to data file (if different from training data)')
@click.option('--output', '-o', required=True, help='Output path for entropy results')
@click.option('--n-samples', '-n', default=1000, help='Number of Monte Carlo samples')
@click.option('--n-iterations', default=10, help='Number of iterations for convergence')
@click.option('--alpha', default=0.05, help='Significance level for confidence intervals')
@click.option('--mutual-info', '-mi', help='Comma-separated variable indices for MI (e.g., "0,1:2,3")')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Verbose output')
def estimate_entropy_cli(model: str, data: Optional[str], output: str, 
                        n_samples: int, n_iterations: int, alpha: float,
                        mutual_info: Optional[str], verbose: bool):
    """Estimate entropy and mutual information for a fitted vine model."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load model
        logger.info(f"Loading vine model from {model}")
        vine = _load_model(model)
        
        # Load data if provided
        if data:
            logger.info(f"Loading data from {data}")
            data_array = _load_data(data)
        else:
            logger.info("No data provided, using model's internal data")
            data_array = None
        
        # Set up info estimation parameters
        info_dict = {
            'alpha': alpha,
            'cases': n_samples,
            'iterations': n_iterations
        }
        
        results = {}
        
        # Estimate entropy
        logger.info("Estimating vine entropy")
        entropy = vine_entropy(vine, info_dict)
        results['entropy'] = entropy
        logger.info(f"Estimated entropy: {entropy:.4f}")
        
        # Estimate mutual information if requested
        if mutual_info:
            logger.info("Computing mutual information")
            mi_pairs = _parse_mutual_info_spec(mutual_info)
            
            for i, (x_indices, y_indices) in enumerate(mi_pairs):
                logger.info(f"Computing MI between variables {x_indices} and {y_indices}")
                mi_value = mutual_information(vine, x_indices, y_indices, info_dict)
                results[f'mutual_info_{i}'] = {
                    'x_variables': x_indices,
                    'y_variables': y_indices,
                    'value': mi_value
                }
                logger.info(f"MI({x_indices};{y_indices}) = {mi_value:.4f}")
        
        # Save results
        logger.info(f"Saving results to {output}")
        _save_results(results, output)
        
        logger.info("Entropy estimation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in entropy estimation: {e}")
        raise click.ClickException(str(e))


@click.command()
@click.option('--data', '-d', required=True, help='Path to time series data file')
@click.option('--base-vine', '-v', help='Path to base vine model (optional)')
@click.option('--vine-type', '-t', default='r-vine',
              type=click.Choice(['c-vine', 'd-vine', 'r-vine']),
              help='Type of base vine if not provided')
@click.option('--time-column', default='time', help='Name of time column in data')
@click.option('--output-dir', '-o', required=True, help='Output directory for results')
@click.option('--n-epochs', '-e', default=100, help='Number of training epochs')
@click.option('--learning-rate', '-lr', default=0.001, help='Learning rate')
@click.option('--batch-size', '-b', default=64, help='Batch size for training')
@click.option('--hidden-dims', default='64,32', help='Hidden dimensions for time flow (comma-separated)')
@click.option('--device', default='auto', help='Device for computation (auto, cpu, cuda)')
@click.option('--config', '-c', help='YAML configuration file')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Verbose output')
def time_model_cli(data: str, base_vine: Optional[str], vine_type: str,
                  time_column: str, output_dir: str, n_epochs: int,
                  learning_rate: float, batch_size: int, hidden_dims: str,
                  device: str, config: Optional[str], verbose: bool):
    """Train and analyze time-dependent vine copula models."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config_dict = {}
        if config:
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load time series data
        logger.info(f"Loading time series data from {data}")
        df = pd.read_csv(data)
        
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found in data")
        
        time_values = df[time_column].values
        data_values = df.drop(columns=[time_column]).values
        
        logger.info(f"Data shape: {data_values.shape}, Time range: [{time_values.min():.3f}, {time_values.max():.3f}]")
        
        # Load or create base vine
        if base_vine:
            logger.info(f"Loading base vine from {base_vine}")
            base_vine_obj = _load_model(base_vine)
        else:
            logger.info(f"Creating base {vine_type}")
            base_vine_obj = create_vine(
                vine_type=vine_type,
                vine_depth=data_values.shape[1],
                **config_dict.get('vine_params', {})
            )
            
            # Fit base vine to data
            logger.info("Fitting base vine to data")
            gen_dict = {'param': True, 'binning': False, 'fitted': False}
            npc_dict = {}
            par_dict = {'param_families': ['gaussian', 'clayton', 'frank']}
            bin_dict = {}
            
            fit_vine(base_vine_obj, data_values, gen_dict, npc_dict, par_dict, bin_dict)
        
        # Set up device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Parse hidden dimensions
        hidden_dim_list = [int(d.strip()) for d in hidden_dims.split(',')]
        
        # Create time-dependent vine
        logger.info("Creating time-dependent vine model")
        time_vine = create_time_dependent_vine(
            base_vine=base_vine_obj,
            hidden_dims=hidden_dim_list,
            device=device
        )
        
        # Training (simplified - you'd want a proper training loop)
        logger.info(f"Training time-dependent vine for {n_epochs} epochs")
        
        # Convert data to tensors
        data_tensor = torch.from_numpy(data_values).float()
        time_tensor = torch.from_numpy(time_values).float()
        
        # Simple training loop
        optimizer = torch.optim.Adam(time_vine.parameters(), lr=learning_rate)
        
        for epoch in range(n_epochs):
            # Create random batches
            n_samples = data_tensor.shape[0]
            indices = torch.randperm(n_samples)[:batch_size]
            
            batch_data = data_tensor[indices]
            batch_time = time_tensor[indices]
            
            # Forward pass
            nll = time_vine(batch_data, batch_time)
            loss = torch.mean(nll)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0 and verbose:
                logger.info(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item():.4f}")
        
        # Save trained model
        model_path = output_path / "time_dependent_vine.pkl"
        logger.info(f"Saving trained model to {model_path}")
        torch.save(time_vine.state_dict(), model_path)
        
        # Analyze results
        logger.info("Analyzing time-dependent entropy")
        
        from ..time.models import DynamicEntropyEstimator
        entropy_estimator = DynamicEntropyEstimator(time_vine)
        
        time_points, entropy_values = entropy_estimator.estimate_entropy_trajectory(
            time_start=time_values.min(),
            time_end=time_values.max()
        )
        
        # Save entropy trajectory
        entropy_results = {
            'time_points': time_points.numpy().tolist(),
            'entropy_values': entropy_values.numpy().tolist()
        }
        
        entropy_path = output_path / "entropy_trajectory.json"
        with open(entropy_path, 'w') as f:
            json.dump(entropy_results, f, indent=2)
        
        # Create plots if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(time_points.numpy(), entropy_values.numpy(), 'b-', linewidth=2)
            plt.xlabel('Time')
            plt.ylabel('Entropy')
            plt.title('Time-Dependent Entropy Evolution')
            plt.grid(True, alpha=0.3)
            
            plot_path = output_path / "entropy_trajectory.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Entropy plot saved to {plot_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")
        
        logger.info("Time-dependent modeling completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in time-dependent modeling: {e}")
        raise click.ClickException(str(e))


@click.command()
@click.option('--config', '-c', required=True, help='Path to experiment configuration file')
@click.option('--output-dir', '-o', required=True, help='Output directory for results')
@click.option('--n-runs', '-n', default=1, help='Number of experimental runs')
@click.option('--seed', '-s', default=42, help='Random seed')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Verbose output')
def run_experiment_cli(config: str, output_dir: str, n_runs: int, seed: int, verbose: bool):
    """Run experimental configurations for vine copula analysis."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load experiment configuration
        logger.info(f"Loading experiment configuration from {config}")
        with open(config, 'r') as f:
            experiment_config = yaml.safe_load(f)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run experiments
        logger.info(f"Running {n_runs} experimental run(s)")
        
        results = run_experiment(
            config=experiment_config,
            output_dir=str(output_path),
            n_runs=n_runs,
            seed=seed,
            verbose=verbose
        )
        
        # Save experiment summary
        summary_path = output_path / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in experiment: {e}")
        raise click.ClickException(str(e))


# Helper functions

def _load_data(file_path: str) -> np.ndarray:
    """Load data from various file formats."""
    path = Path(file_path)
    
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
        return df.values
    elif path.suffix.lower() == '.npy':
        return np.load(file_path)
    elif path.suffix.lower() in ['.pkl', '.pickle']:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def _save_model(model, file_path: str):
    """Save a model to file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)


def _load_model(file_path: str):
    """Load a model from file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def _save_results(results: Dict[str, Any], file_path: str):
    """Save results to JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def _parse_mutual_info_spec(mi_spec: str) -> List[Tuple[List[int], List[int]]]:
    """Parse mutual information specification string."""
    pairs = []
    
    for pair_spec in mi_spec.split(':'):
        parts = pair_spec.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid MI specification: {pair_spec}")
        
        x_indices = [int(i.strip()) for i in parts[0].split()]
        y_indices = [int(i.strip()) for i in parts[1].split()]
        
        pairs.append((x_indices, y_indices))
    
    return pairs


def _print_vine_summary(vine, data: np.ndarray):
    """Print a summary of the fitted vine model."""
    print("\n" + "="*60)
    print("VINE COPULA MODEL SUMMARY")
    print("="*60)
    
    print(f"Vine Type: {vine.vine_family}")
    print(f"Dimension: {vine.n_cop}")
    print(f"Data Shape: {data.shape}")
    
    if hasattr(vine, 'copulas') and vine.copulas:
        print(f"Number of Tree Levels: {len(vine.copulas)}")
        
        total_edges = sum(len(level) for level in vine.copulas)
        print(f"Total Edges: {total_edges}")
        
        # Count copula families
        family_counts = {}
        for level_copulas in vine.copulas:
            for copula in level_copulas:
                if hasattr(copula, 'family'):
                    family = copula.family
                    family_counts[family] = family_counts.get(family, 0) + 1
        
        if family_counts:
            print("Copula Families:")
            for family, count in family_counts.items():
                print(f"  {family}: {count}")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    # This allows running individual commands for testing
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'fit':
            fit_vine_cli()
        elif command == 'entropy':
            estimate_entropy_cli()
        elif command == 'time':
            time_model_cli()
        elif command == 'experiment':
            run_experiment_cli()
        else:
            print("Available commands: fit, entropy, time, experiment")
    else:
        print("Usage: python main.py <command>")
        print("Available commands: fit, entropy, time, experiment")
