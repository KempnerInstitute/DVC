#!/usr/bin/env python3
"""
Time-Dependent Data Generation for Vine Copula Analysis

This module provides utilities for generating synthetic datasets with
time-dependent interaction structures to test time-dependent vine copulas.

Scenarios Included:
1. Piecewise correlation changes
2. Sinusoidal correlation evolution  
3. Regime switching models
4. Changing vine structures over time
5. Real-world inspired temporal patterns

Author: DVC Analysis Team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal, norm
from scipy.linalg import cholesky
import os


class TimeDependentDataGenerator:
    """
    Generator for time-dependent multivariate datasets with varying interaction structures
    """
    
    def __init__(self, dim, random_seed=42):
        """
        Initialize data generator
        
        Parameters:
        -----------
        dim : int
            Data dimensionality
        random_seed : int
            Random seed for reproducibility
        """
        
        self.dim = dim
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Storage for generated datasets
        self.generated_datasets = {}
        
        # Results directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(current_dir, '..', 'results', 'time_dependent_data')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def generate_piecewise_correlation_data(self, 
                                          n_time_steps, 
                                          n_samples_per_time,
                                          breakpoints=[0.3, 0.7],
                                          correlations=[0.2, 0.8, 0.4]):
        """
        Generate data with piecewise constant correlations
        
        Parameters:
        -----------
        n_time_steps : int
            Number of time steps
        n_samples_per_time : int
            Number of samples per time step
        breakpoints : list
            Relative breakpoints (in [0,1]) where correlation changes
        correlations : list
            Correlation values for each segment
            
        Returns:
        --------
        data : np.ndarray
            Time series data, shape (n_time_steps, n_samples_per_time, dim)
        time_indices : np.ndarray
            Time indices
        metadata : dict
            Information about the generation process
        """
        
        print(f"Generating piecewise correlation data...")
        print(f"  Breakpoints: {breakpoints}")
        print(f"  Correlations: {correlations}")
        
        data = np.zeros((n_time_steps, n_samples_per_time, self.dim))
        time_indices = np.arange(n_time_steps)
        correlation_evolution = np.zeros(n_time_steps)
        
        # Convert breakpoints to time indices
        breakpoint_times = [int(bp * n_time_steps) for bp in breakpoints]
        breakpoint_times = [0] + breakpoint_times + [n_time_steps]
        
        for t in range(n_time_steps):
            # Determine which segment we're in
            segment = 0
            for i, bp_time in enumerate(breakpoint_times[1:]):
                if t < bp_time:
                    segment = i
                    break
            
            correlation = correlations[segment]
            correlation_evolution[t] = correlation
            
            # Create correlation matrix
            corr_matrix = self._create_correlation_matrix(correlation)
            
            # Generate multivariate normal data
            mean = np.zeros(self.dim)
            data[t] = multivariate_normal.rvs(mean=mean, cov=corr_matrix, size=n_samples_per_time)
        
        metadata = {
            'type': 'piecewise_correlation',
            'breakpoints': breakpoints,
            'correlations': correlations,
            'correlation_evolution': correlation_evolution,
            'n_time_steps': n_time_steps,
            'n_samples_per_time': n_samples_per_time
        }
        
        self.generated_datasets['piecewise'] = {
            'data': data,
            'time_indices': time_indices,
            'metadata': metadata
        }
        
        return data, time_indices, metadata
    
    def generate_sinusoidal_correlation_data(self,
                                           n_time_steps,
                                           n_samples_per_time, 
                                           base_correlation=0.5,
                                           amplitude=0.3,
                                           frequency=1.0,
                                           phase=0.0):
        """
        Generate data with sinusoidally varying correlations
        
        Parameters:
        -----------
        n_time_steps : int
            Number of time steps
        n_samples_per_time : int
            Number of samples per time step
        base_correlation : float
            Base correlation level
        amplitude : float
            Amplitude of correlation oscillation
        frequency : float
            Frequency of oscillation (cycles per full time series)
        phase : float
            Phase offset
            
        Returns:
        --------
        data : np.ndarray
            Time series data
        time_indices : np.ndarray
            Time indices
        metadata : dict
            Generation metadata
        """
        
        print(f"Generating sinusoidal correlation data...")
        print(f"  Base correlation: {base_correlation}")
        print(f"  Amplitude: {amplitude}")
        print(f"  Frequency: {frequency}")
        
        data = np.zeros((n_time_steps, n_samples_per_time, self.dim))
        time_indices = np.arange(n_time_steps)
        correlation_evolution = np.zeros(n_time_steps)
        
        for t in range(n_time_steps):
            # Compute time-dependent correlation
            time_normalized = t / n_time_steps
            correlation = base_correlation + amplitude * np.sin(
                2 * np.pi * frequency * time_normalized + phase
            )
            
            # Ensure correlation is in valid range [-1, 1]
            correlation = np.clip(correlation, -0.95, 0.95)
            correlation_evolution[t] = correlation
            
            # Create correlation matrix
            corr_matrix = self._create_correlation_matrix(correlation)
            
            # Generate data
            mean = np.zeros(self.dim)
            data[t] = multivariate_normal.rvs(mean=mean, cov=corr_matrix, size=n_samples_per_time)
        
        metadata = {
            'type': 'sinusoidal_correlation',
            'base_correlation': base_correlation,
            'amplitude': amplitude,
            'frequency': frequency,
            'phase': phase,
            'correlation_evolution': correlation_evolution,
            'n_time_steps': n_time_steps,
            'n_samples_per_time': n_samples_per_time
        }
        
        self.generated_datasets['sinusoidal'] = {
            'data': data,
            'time_indices': time_indices,
            'metadata': metadata
        }
        
        return data, time_indices, metadata
    
    def generate_regime_switching_data(self,
                                     n_time_steps,
                                     n_samples_per_time,
                                     n_regimes=3,
                                     regime_persistence=0.95):
        """
        Generate data with Markov regime switching
        
        Parameters:
        -----------
        n_time_steps : int
            Number of time steps
        n_samples_per_time : int
            Number of samples per time step
        n_regimes : int
            Number of regimes
        regime_persistence : float
            Probability of staying in the same regime
            
        Returns:
        --------
        data : np.ndarray
            Time series data
        time_indices : np.ndarray
            Time indices
        metadata : dict
            Generation metadata
        """
        
        print(f"Generating regime switching data with {n_regimes} regimes...")
        
        # Define regime-specific parameters
        regime_correlations = np.linspace(0.1, 0.9, n_regimes)
        regime_variances = np.linspace(0.5, 1.5, n_regimes)
        
        # Create transition matrix
        transition_prob = (1 - regime_persistence) / (n_regimes - 1)
        transition_matrix = np.full((n_regimes, n_regimes), transition_prob)
        np.fill_diagonal(transition_matrix, regime_persistence)
        
        # Generate regime sequence
        regimes = np.zeros(n_time_steps, dtype=int)
        regimes[0] = np.random.randint(n_regimes)
        
        for t in range(1, n_time_steps):
            current_regime = regimes[t-1]
            regimes[t] = np.random.choice(n_regimes, p=transition_matrix[current_regime])
        
        # Generate data
        data = np.zeros((n_time_steps, n_samples_per_time, self.dim))
        time_indices = np.arange(n_time_steps)
        
        for t in range(n_time_steps):
            regime = regimes[t]
            correlation = regime_correlations[regime]
            variance = regime_variances[regime]
            
            # Create correlation matrix
            corr_matrix = self._create_correlation_matrix(correlation) * variance
            
            # Generate data
            mean = np.zeros(self.dim)
            data[t] = multivariate_normal.rvs(mean=mean, cov=corr_matrix, size=n_samples_per_time)
        
        metadata = {
            'type': 'regime_switching',
            'n_regimes': n_regimes,
            'regime_persistence': regime_persistence,
            'regimes': regimes,
            'regime_correlations': regime_correlations,
            'regime_variances': regime_variances,
            'transition_matrix': transition_matrix,
            'n_time_steps': n_time_steps,
            'n_samples_per_time': n_samples_per_time
        }
        
        self.generated_datasets['regime_switching'] = {
            'data': data,
            'time_indices': time_indices,
            'metadata': metadata
        }
        
        return data, time_indices, metadata
    
    def generate_changing_vine_structure_data(self,
                                            n_time_steps,
                                            n_samples_per_time,
                                            structure_change_times=[0.33, 0.67]):
        """
        Generate data where the vine structure itself changes over time
        
        This creates scenarios where different variable pairs are correlated
        at different times, simulating changing vine tree structures.
        
        Parameters:
        -----------
        n_time_steps : int
            Number of time steps
        n_samples_per_time : int
            Number of samples per time step
        structure_change_times : list
            Relative times when structure changes
            
        Returns:
        --------
        data : np.ndarray
            Time series data
        time_indices : np.ndarray
            Time indices
        metadata : dict
            Generation metadata
        """
        
        print(f"Generating changing vine structure data...")
        
        if self.dim < 3:
            raise ValueError("Need at least 3 dimensions for changing vine structures")
        
        data = np.zeros((n_time_steps, n_samples_per_time, self.dim))
        time_indices = np.arange(n_time_steps)
        
        # Define different structure periods
        change_times = [0] + [int(t * n_time_steps) for t in structure_change_times] + [n_time_steps]
        
        # Define correlation patterns for each period
        structures = []
        
        if len(change_times) >= 4:  # At least 3 periods
            # Period 1: Chain structure (0-1-2-...)
            struct1 = np.eye(self.dim)
            for i in range(self.dim - 1):
                struct1[i, i+1] = struct1[i+1, i] = 0.7
            structures.append(('chain', struct1))
            
            # Period 2: Star structure (0 connected to all)
            struct2 = np.eye(self.dim)
            for i in range(1, self.dim):
                struct2[0, i] = struct2[i, 0] = 0.6
            structures.append(('star', struct2))
            
            # Period 3: Different pairs
            struct3 = np.eye(self.dim)
            if self.dim >= 4:
                struct3[0, 2] = struct3[2, 0] = 0.8
                struct3[1, 3] = struct3[3, 1] = 0.8
            else:
                struct3[0, 2] = struct3[2, 0] = 0.8
            structures.append(('pairs', struct3))
        
        # Generate data for each period
        active_structures = []
        for t in range(n_time_steps):
            # Determine which period we're in
            period = 0
            for i, change_time in enumerate(change_times[1:]):
                if t < change_time:
                    period = i
                    break
            
            if period < len(structures):
                structure_name, corr_matrix = structures[period]
                active_structures.append(structure_name)
            else:
                structure_name, corr_matrix = structures[-1]
                active_structures.append(structure_name)
            
            # Generate data
            mean = np.zeros(self.dim)
            data[t] = multivariate_normal.rvs(mean=mean, cov=corr_matrix, size=n_samples_per_time)
        
        metadata = {
            'type': 'changing_vine_structure',
            'structure_change_times': structure_change_times,
            'structures': [s[0] for s in structures],
            'correlation_matrices': [s[1] for s in structures],
            'active_structures': active_structures,
            'n_time_steps': n_time_steps,
            'n_samples_per_time': n_samples_per_time
        }
        
        self.generated_datasets['changing_structure'] = {
            'data': data,
            'time_indices': time_indices,
            'metadata': metadata
        }
        
        return data, time_indices, metadata
    
    def generate_financial_inspired_data(self,
                                        n_time_steps,
                                        n_samples_per_time,
                                        volatility_clustering=True,
                                        correlation_breaks=True):
        """
        Generate financial market inspired time-dependent data
        
        Features:
        - Volatility clustering (GARCH-like effects)
        - Correlation breaks during stress periods
        - Asymmetric correlations (higher during downturns)
        
        Parameters:
        -----------
        n_time_steps : int
            Number of time steps
        n_samples_per_time : int
            Number of samples per time step  
        volatility_clustering : bool
            Whether to include volatility clustering
        correlation_breaks : bool
            Whether to include structural breaks in correlation
            
        Returns:
        --------
        data : np.ndarray
            Time series data
        time_indices : np.ndarray
            Time indices
        metadata : dict
            Generation metadata
        """
        
        print(f"Generating financial-inspired data...")
        
        data = np.zeros((n_time_steps, n_samples_per_time, self.dim))
        time_indices = np.arange(n_time_steps)
        
        # Initialize volatility and correlation series
        volatilities = np.ones(n_time_steps)
        correlations = np.zeros(n_time_steps)
        market_shocks = np.zeros(n_time_steps)
        
        # Parameters
        base_correlation = 0.3
        shock_correlation = 0.8
        volatility_persistence = 0.9
        shock_probability = 0.05
        
        # Generate market dynamics
        for t in range(n_time_steps):
            # Generate market shocks
            if np.random.random() < shock_probability:
                market_shocks[t] = 1
                shock_intensity = np.random.exponential(2.0)
            else:
                market_shocks[t] = 0
                shock_intensity = 0
            
            # Update volatility (GARCH-like)
            if volatility_clustering and t > 0:
                volatilities[t] = (
                    0.1 + 
                    volatility_persistence * volatilities[t-1] + 
                    0.1 * shock_intensity
                )
            else:
                volatilities[t] = 1.0 + 0.5 * shock_intensity
            
            # Update correlations (higher during stress)
            if correlation_breaks:
                if market_shocks[t] == 1:
                    correlations[t] = shock_correlation
                else:
                    # Smooth transition back to base
                    if t > 0:
                        correlations[t] = 0.95 * correlations[t-1] + 0.05 * base_correlation
                    else:
                        correlations[t] = base_correlation
            else:
                correlations[t] = base_correlation
        
        # Generate data
        for t in range(n_time_steps):
            volatility = volatilities[t]
            correlation = correlations[t]
            
            # Create correlation matrix
            corr_matrix = self._create_correlation_matrix(correlation) * volatility**2
            
            # Generate data
            mean = np.zeros(self.dim)
            data[t] = multivariate_normal.rvs(mean=mean, cov=corr_matrix, size=n_samples_per_time)
        
        metadata = {
            'type': 'financial_inspired',
            'volatility_clustering': volatility_clustering,
            'correlation_breaks': correlation_breaks,
            'volatilities': volatilities,
            'correlations': correlations,
            'market_shocks': market_shocks,
            'base_correlation': base_correlation,
            'shock_correlation': shock_correlation,
            'n_time_steps': n_time_steps,
            'n_samples_per_time': n_samples_per_time
        }
        
        self.generated_datasets['financial'] = {
            'data': data,
            'time_indices': time_indices,
            'metadata': metadata
        }
        
        return data, time_indices, metadata
    
    def _create_correlation_matrix(self, correlation):
        """
        Create a valid correlation matrix with given pairwise correlation
        
        For simplicity, creates a matrix where all off-diagonal elements
        are equal to the given correlation.
        
        Parameters:
        -----------
        correlation : float
            Pairwise correlation value
            
        Returns:
        --------
        corr_matrix : np.ndarray
            Correlation matrix
        """
        
        corr_matrix = np.eye(self.dim)
        
        if self.dim == 2:
            corr_matrix[0, 1] = corr_matrix[1, 0] = correlation
        else:
            # For higher dimensions, use a more sophisticated approach
            # to ensure positive definiteness
            
            # Fill off-diagonal with correlation
            for i in range(self.dim):
                for j in range(i+1, self.dim):
                    corr_matrix[i, j] = corr_matrix[j, i] = correlation
            
            # Ensure positive definiteness by adjusting eigenvalues if needed
            eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
            min_eigenval = np.min(eigenvals)
            
            if min_eigenval < 1e-6:
                # Adjust eigenvalues to make matrix positive definite
                eigenvals = np.maximum(eigenvals, 1e-6)
                corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                
                # Rescale to unit diagonal
                diag_sqrt = np.sqrt(np.diag(corr_matrix))
                corr_matrix = corr_matrix / np.outer(diag_sqrt, diag_sqrt)
        
        return corr_matrix
    
    def visualize_generated_data(self, dataset_name, save_plots=True):
        """
        Create visualizations for generated dataset
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to visualize
        save_plots : bool
            Whether to save plots to disk
        """
        
        if dataset_name not in self.generated_datasets:
            print(f"Dataset '{dataset_name}' not found")
            return
        
        dataset = self.generated_datasets[dataset_name]
        data = dataset['data']
        time_indices = dataset['time_indices']
        metadata = dataset['metadata']
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Time-Dependent Data Analysis: {dataset_name}', fontsize=16)
        
        # 1. Time series of sample means
        axes[0, 0].plot(time_indices, np.mean(data, axis=1))
        axes[0, 0].set_title('Sample Means Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].legend([f'Var {i}' for i in range(self.dim)])
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Time series of sample correlations (first pair)
        if self.dim >= 2:
            correlations_over_time = []
            for t in range(len(time_indices)):
                corr = np.corrcoef(data[t][:, 0], data[t][:, 1])[0, 1]
                correlations_over_time.append(corr)
            
            axes[0, 1].plot(time_indices, correlations_over_time, linewidth=2, label='Empirical')
            
            # Plot true correlation if available
            if 'correlation_evolution' in metadata:
                axes[0, 1].plot(time_indices, metadata['correlation_evolution'], 
                               linewidth=2, linestyle='--', label='True')
            
            axes[0, 1].set_title('Correlation Evolution (Var 0 vs 1)')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Correlation')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sample volatilities
        volatilities_over_time = np.std(data, axis=1)
        axes[0, 2].plot(time_indices, volatilities_over_time)
        axes[0, 2].set_title('Sample Volatilities Over Time')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Standard Deviation')
        axes[0, 2].legend([f'Var {i}' for i in range(self.dim)])
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Heatmap of correlation matrix evolution
        if self.dim >= 2:
            n_time_points = min(50, len(time_indices))
            time_subset = np.linspace(0, len(time_indices)-1, n_time_points, dtype=int)
            
            corr_evolution = np.zeros((n_time_points, self.dim, self.dim))
            for i, t in enumerate(time_subset):
                corr_evolution[i] = np.corrcoef(data[t].T)
            
            # Show correlation between first two variables
            im = axes[1, 0].imshow(corr_evolution[:, 0, 1].reshape(-1, 1).T, 
                                  aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, 0].set_title('Correlation Matrix Evolution')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Var Pairs')
            plt.colorbar(im, ax=axes[1, 0])
        
        # 5. Distribution comparison (first vs last time period)
        if len(time_indices) > 1:
            # First time period
            axes[1, 1].scatter(data[0][:, 0], data[0][:, 1], alpha=0.5, s=10, label='t=0')
            # Last time period  
            axes[1, 1].scatter(data[-1][:, 0], data[-1][:, 1], alpha=0.5, s=10, label=f't={len(time_indices)-1}')
            axes[1, 1].set_title('Distribution Comparison (Var 0 vs 1)')
            axes[1, 1].set_xlabel('Variable 0')
            axes[1, 1].set_ylabel('Variable 1')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Metadata-specific plots
        if metadata['type'] == 'regime_switching':
            axes[1, 2].plot(time_indices, metadata['regimes'])
            axes[1, 2].set_title('Regime Sequence')
            axes[1, 2].set_xlabel('Time')
            axes[1, 2].set_ylabel('Regime')
            axes[1, 2].grid(True, alpha=0.3)
        elif metadata['type'] == 'financial_inspired':
            axes[1, 2].plot(time_indices, metadata['volatilities'], label='Volatility')
            axes[1, 2].scatter(time_indices[metadata['market_shocks'] == 1], 
                             metadata['volatilities'][metadata['market_shocks'] == 1],
                             color='red', s=20, label='Shocks')
            axes[1, 2].set_title('Volatility and Market Shocks')
            axes[1, 2].set_xlabel('Time')
            axes[1, 2].set_ylabel('Volatility')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            # Generic time series plot
            sample_data = data[:, 0, :5]  # First sample, first 5 variables
            axes[1, 2].plot(time_indices, sample_data)
            axes[1, 2].set_title('Sample Time Series')
            axes[1, 2].set_xlabel('Time')
            axes[1, 2].set_ylabel('Value')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.results_dir, f'{dataset_name}_visualization.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {plot_path}")
        
        plt.show()
    
    def save_dataset(self, dataset_name, format='npz'):
        """
        Save generated dataset to disk
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to save
        format : str
            File format ('npz', 'csv', 'json')
        """
        
        if dataset_name not in self.generated_datasets:
            print(f"Dataset '{dataset_name}' not found")
            return
        
        dataset = self.generated_datasets[dataset_name]
        
        if format == 'npz':
            filepath = os.path.join(self.results_dir, f'{dataset_name}_data.npz')
            np.savez(filepath,
                    data=dataset['data'],
                    time_indices=dataset['time_indices'],
                    metadata=dataset['metadata'])
            
        print(f"Saved dataset '{dataset_name}' to {self.results_dir}")
    
    def get_dataset_summary(self):
        """
        Print summary of all generated datasets
        """
        
        print("Generated Datasets Summary:")
        print("=" * 50)
        
        for name, dataset in self.generated_datasets.items():
            metadata = dataset['metadata']
            data_shape = dataset['data'].shape
            
            print(f"\nDataset: {name}")
            print(f"  Type: {metadata['type']}")
            print(f"  Shape: {data_shape}")
            print(f"  Time steps: {metadata['n_time_steps']}")
            print(f"  Samples per time: {metadata['n_samples_per_time']}")
            
            # Type-specific information
            if metadata['type'] == 'piecewise_correlation':
                print(f"  Breakpoints: {metadata['breakpoints']}")
                print(f"  Correlations: {metadata['correlations']}")
            elif metadata['type'] == 'sinusoidal_correlation':
                print(f"  Base correlation: {metadata['base_correlation']}")
                print(f"  Amplitude: {metadata['amplitude']}")
                print(f"  Frequency: {metadata['frequency']}")
            elif metadata['type'] == 'regime_switching':
                print(f"  Number of regimes: {metadata['n_regimes']}")
                print(f"  Persistence: {metadata['regime_persistence']}")


def main():
    """
    Example usage of time-dependent data generator
    """
    
    print("Time-Dependent Data Generation Demo")
    print("=" * 50)
    
    # Initialize generator
    generator = TimeDependentDataGenerator(dim=4, random_seed=42)
    
    # Generate different types of datasets
    n_time_steps = 100
    n_samples = 200
    
    # 1. Piecewise correlation data
    print("\n1. Generating piecewise correlation data...")
    data1, times1, meta1 = generator.generate_piecewise_correlation_data(
        n_time_steps, n_samples,
        breakpoints=[0.3, 0.7],
        correlations=[0.2, 0.8, 0.1]
    )
    generator.visualize_generated_data('piecewise', save_plots=True)
    
    # 2. Sinusoidal correlation data
    print("\n2. Generating sinusoidal correlation data...")
    data2, times2, meta2 = generator.generate_sinusoidal_correlation_data(
        n_time_steps, n_samples,
        base_correlation=0.4,
        amplitude=0.3,
        frequency=2.0
    )
    generator.visualize_generated_data('sinusoidal', save_plots=True)
    
    # 3. Regime switching data
    print("\n3. Generating regime switching data...")
    data3, times3, meta3 = generator.generate_regime_switching_data(
        n_time_steps, n_samples,
        n_regimes=3,
        regime_persistence=0.9
    )
    generator.visualize_generated_data('regime_switching', save_plots=True)
    
    # 4. Financial inspired data
    print("\n4. Generating financial-inspired data...")
    data4, times4, meta4 = generator.generate_financial_inspired_data(
        n_time_steps, n_samples,
        volatility_clustering=True,
        correlation_breaks=True
    )
    generator.visualize_generated_data('financial', save_plots=True)
    
    # Print summary
    generator.get_dataset_summary()
    
    # Save all datasets
    for name in generator.generated_datasets.keys():
        generator.save_dataset(name)
    
    print(f"\nAll results saved to: {generator.results_dir}")


if __name__ == "__main__":
    main() 