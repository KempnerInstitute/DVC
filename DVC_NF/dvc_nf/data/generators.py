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
        # Find the DVC_NF root directory
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to dvc_nf, then up to DVC_NF root
        dvc_nf_root = os.path.dirname(os.path.dirname(current_file_dir))
        self.results_dir = os.path.join(dvc_nf_root, 'results', 'time_dependent_data')
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
    
    def generate_block_switching_correlation_data(self,
                                                n_time_steps,
                                                n_samples_per_time,
                                                block_sizes=None,
                                                n_regimes=4,
                                                switch_probability=0.05,
                                                within_block_corr_range=(0.4, 0.8),
                                                between_block_corr_range=(-0.6, -0.2)):
        """
        Generate data with block-structured correlation matrices that switch dynamically
        
        Creates sophisticated temporal patterns where:
        - Variables are organized in blocks with positive within-block correlations
        - Between-block correlations are negative  
        - Multiple correlation regimes switch stochastically over time
        - Both correlation values and signs can change
        
        Parameters:
        -----------
        n_time_steps : int
            Number of time steps
        n_samples_per_time : int
            Number of samples per time step
        block_sizes : list, optional
            Sizes of correlation blocks. If None, creates roughly equal blocks
        n_regimes : int
            Number of different correlation regimes
        switch_probability : float
            Probability of switching regimes at each time step
        within_block_corr_range : tuple
            (min, max) correlation within blocks
        between_block_corr_range : tuple
            (min, max) correlation between blocks
            
        Returns:
        --------
        data : np.ndarray
            Time series data, shape (n_time_steps, n_samples_per_time, dim)
        time_indices : np.ndarray
            Time indices
        metadata : dict
            Generation metadata including correlation evolution
        """
        
        print(f"Generating block-structured switching correlation data...")
        print(f"  Dimensions: {self.dim}")
        print(f"  Number of regimes: {n_regimes}")
        print(f"  Switch probability: {switch_probability}")
        print(f"  Within-block correlation: {within_block_corr_range}")
        print(f"  Between-block correlation: {between_block_corr_range}")
        
        # Define block structure
        if block_sizes is None:
            # Create roughly equal blocks
            if self.dim <= 3:
                block_sizes = [self.dim]  # Single block for small dimensions
            elif self.dim <= 6:
                block_sizes = [self.dim // 2, self.dim - self.dim // 2]  # Two blocks
            else:
                # Three blocks
                block_size = self.dim // 3
                block_sizes = [block_size, block_size, self.dim - 2 * block_size]
        
        # Ensure block sizes sum to total dimensions
        if sum(block_sizes) != self.dim:
            raise ValueError(f"Block sizes {block_sizes} don't sum to {self.dim}")
        
        print(f"  Block structure: {block_sizes}")
        
        # Create block indices
        block_indices = []
        start_idx = 0
        for block_size in block_sizes:
            block_indices.append(list(range(start_idx, start_idx + block_size)))
            start_idx += block_size
        
        print(f"  Block indices: {block_indices}")
        
        # Define correlation regimes with different patterns
        regimes = []
        for regime_idx in range(n_regimes):
            # Generate regime-specific correlation parameters
            within_corr_base = np.random.uniform(
                within_block_corr_range[0], within_block_corr_range[1]
            )
            between_corr_base = np.random.uniform(
                between_block_corr_range[0], between_block_corr_range[1]
            )
            
            # Add regime-specific variations
            regime_factor = 1.0 + 0.3 * np.sin(2 * np.pi * regime_idx / n_regimes)
            within_corr = np.clip(within_corr_base * regime_factor, 0.1, 0.95)
            between_corr = np.clip(between_corr_base * regime_factor, -0.95, -0.1)
            
            # Occasionally flip signs for complex dynamics
            if regime_idx == n_regimes // 2:
                between_corr = -between_corr  # Flip between-block correlation sign
            
            regimes.append({
                'within_block_corr': within_corr,
                'between_block_corr': between_corr,
                'regime_id': regime_idx
            })
        
        print(f"  Defined {len(regimes)} correlation regimes")
        
        # Generate regime sequence with switching dynamics
        regime_sequence = np.zeros(n_time_steps, dtype=int)
        current_regime = 0
        regime_sequence[0] = current_regime
        
        for t in range(1, n_time_steps):
            if np.random.random() < switch_probability:
                # Switch to different regime
                available_regimes = [r for r in range(n_regimes) if r != current_regime]
                current_regime = np.random.choice(available_regimes)
            regime_sequence[t] = current_regime
        
        # Generate time series data
        data = np.zeros((n_time_steps, n_samples_per_time, self.dim))
        time_indices = np.arange(n_time_steps)
        correlation_matrices = []
        entropy_evolution = []
        
        for t in range(n_time_steps):
            regime = regimes[regime_sequence[t]]
            
            # Create block-structured correlation matrix
            corr_matrix = self._create_block_correlation_matrix(
                block_indices,
                regime['within_block_corr'],
                regime['between_block_corr']
            )
            
            correlation_matrices.append(corr_matrix.copy())
            
            # Compute theoretical entropy for multivariate Gaussian
            # H(X) = 0.5 * log((2πe)^k * |Σ|)
            det_corr = np.linalg.det(corr_matrix)
            if det_corr > 1e-10:  # Ensure positive definite
                entropy = 0.5 * (self.dim * np.log(2 * np.pi * np.e) + np.log(det_corr))
            else:
                entropy = np.nan
            entropy_evolution.append(entropy)
            
            # Generate multivariate normal data
            mean = np.zeros(self.dim)
            try:
                data[t] = multivariate_normal.rvs(
                    mean=mean, cov=corr_matrix, size=n_samples_per_time
                )
            except np.linalg.LinAlgError:
                # Fallback to identity if matrix is not positive definite
                print(f"Warning: Non-positive definite matrix at t={t}, using identity")
                data[t] = multivariate_normal.rvs(
                    mean=mean, cov=np.eye(self.dim), size=n_samples_per_time
                )
        
        metadata = {
            'type': 'block_switching_correlation',
            'block_sizes': block_sizes,
            'block_indices': block_indices,
            'n_regimes': n_regimes,
            'switch_probability': switch_probability,
            'within_block_corr_range': within_block_corr_range,
            'between_block_corr_range': between_block_corr_range,
            'regimes': regimes,
            'regime_sequence': regime_sequence,
            'correlation_matrices': correlation_matrices,
            'entropy_evolution': entropy_evolution,
            'n_time_steps': n_time_steps,
            'n_samples_per_time': n_samples_per_time
        }
        
        self.generated_datasets['block_switching'] = {
            'data': data,
            'time_indices': time_indices,
            'metadata': metadata
        }
        
        print(f"  Generated data with {np.sum(regime_sequence[1:] != regime_sequence[:-1])} regime switches")
        
        return data, time_indices, metadata
    
    def generate_beyond_pairwise_interactions(self,
                                             n_time_steps,
                                             n_samples_per_time,
                                             switch_times=[0.3, 0.7],
                                             corr_low=0.1,
                                             corr_high=0.8,
                                             beyond_pairwise_strength=0.3):
        """
        Generate data with beyond-pairwise interactions (triple interactions)
        
        Creates sophisticated temporal patterns where:
        - Standard pairwise correlations switch between regimes
        - Triple interactions are added: X[i] += strength * X[j] * X[k]
        - Tests vine copula's ability to capture higher-order dependencies
        
        This is particularly valuable for testing because vine copulas theoretically
        should be able to capture such interactions through their hierarchical structure.
        
        Parameters:
        -----------
        n_time_steps : int
            Number of time steps
        n_samples_per_time : int
            Number of samples per time step
        switch_times : list
            Relative switch points (in [0,1]) for correlation regimes
        corr_low : float
            Low correlation value
        corr_high : float
            High correlation value
        beyond_pairwise_strength : float
            Strength of triple interaction effects
            
        Returns:
        --------
        data : np.ndarray
            Time series data, shape (n_time_steps, n_samples_per_time, dim)
        time_indices : np.ndarray
            Time indices
        metadata : dict
            Generation metadata including interaction details
        """
        
        print(f"Generating beyond-pairwise interactions data...")
        print(f"  Dimensions: {self.dim}")
        print(f"  Switch times: {switch_times}")
        print(f"  Correlation range: [{corr_low}, {corr_high}]")
        print(f"  Triple interaction strength: {beyond_pairwise_strength}")
        
        if self.dim < 3:
            raise ValueError("Need at least 3 dimensions for triple interactions")
        
        # Convert relative switch times to absolute time indices
        switch_indices = [int(st * n_time_steps) for st in switch_times]
        switch_indices = [0] + switch_indices + [n_time_steps]
        
        # Define correlation matrices for different regimes
        correlation_matrices = []
        regime_descriptions = []
        
        # Regime 1: Strong (0,1) correlation
        C1 = np.eye(self.dim)
        C1[0, 1] = C1[1, 0] = corr_high
        correlation_matrices.append(C1)
        regime_descriptions.append("Strong (0,1) correlation")
        
        # Regime 2: Strong (1,2) correlation, weak (0,1)
        C2 = np.eye(self.dim)
        if self.dim >= 3:
            C2[1, 2] = C2[2, 1] = corr_high
        C2[0, 1] = C2[1, 0] = corr_low
        correlation_matrices.append(C2)
        regime_descriptions.append("Strong (1,2) correlation, weak (0,1)")
        
        # Regime 3: Moderate all pairs
        C3 = np.eye(self.dim)
        moderate_corr = (corr_low + corr_high) / 2.0
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                C3[i, j] = C3[j, i] = moderate_corr
        correlation_matrices.append(C3)
        regime_descriptions.append("Moderate all pairs")
        
        # Ensure all matrices are positive definite
        correlation_matrices = [self._make_positive_definite_robust(C) for C in correlation_matrices]
        
        # Generate time series data
        data = np.zeros((n_time_steps, n_samples_per_time, self.dim))
        time_indices = np.arange(n_time_steps)
        regime_sequence = np.zeros(n_time_steps, dtype=int)
        correlation_evolution = []
        triple_interaction_evolution = []
        
        for t in range(n_time_steps):
            # Determine current regime
            regime_idx = 0
            for i, switch_idx in enumerate(switch_indices[1:-1]):
                if t >= switch_idx:
                    regime_idx = i + 1
            regime_sequence[t] = regime_idx
            
            # Get correlation matrix for current regime
            cov_matrix = correlation_matrices[regime_idx]
            correlation_evolution.append(cov_matrix.copy())
            
            # Generate multivariate normal data
            mean = np.zeros(self.dim)
            X = multivariate_normal.rvs(mean=mean, cov=cov_matrix, size=n_samples_per_time)
            
            # Add beyond-pairwise (triple) interactions
            if beyond_pairwise_strength > 1e-9:
                # Add triple interaction: X[2] += strength * X[0] * X[1]
                if self.dim >= 3:
                    triple_effect = beyond_pairwise_strength * X[:, 0] * X[:, 1]
                    X[:, 2] += triple_effect
                    triple_interaction_evolution.append(np.mean(np.abs(triple_effect)))
                
                # For higher dimensions, add more triple interactions
                if self.dim >= 4:
                    # X[3] += strength * X[1] * X[2]
                    triple_effect_2 = beyond_pairwise_strength * X[:, 1] * X[:, 2]
                    X[:, 3] += triple_effect_2
                
                if self.dim >= 5:
                    # X[4] += strength * X[0] * X[3]
                    triple_effect_3 = beyond_pairwise_strength * X[:, 0] * X[:, 3]
                    X[:, 4] += triple_effect_3
            else:
                triple_interaction_evolution.append(0.0)
            
            # Standardize each dimension
            for d in range(self.dim):
                X_d = X[:, d]
                X_d = (X_d - np.mean(X_d)) / (np.std(X_d) + 1e-9)
                X[:, d] = X_d
            
            data[t] = X
        
        metadata = {
            'type': 'beyond_pairwise_interactions',
            'switch_times': switch_times,
            'switch_indices': switch_indices[1:-1],
            'corr_low': corr_low,
            'corr_high': corr_high,
            'beyond_pairwise_strength': beyond_pairwise_strength,
            'regime_sequence': regime_sequence,
            'regime_descriptions': regime_descriptions,
            'correlation_matrices': correlation_matrices,
            'correlation_evolution': correlation_evolution,
            'triple_interaction_evolution': triple_interaction_evolution,
            'n_regimes': len(correlation_matrices),
            'n_time_steps': n_time_steps,
            'n_samples_per_time': n_samples_per_time,
            'description': "Piecewise correlation regimes with beyond-pairwise triple interactions"
        }
        
        self.generated_datasets['beyond_pairwise'] = {
            'data': data,
            'time_indices': time_indices,
            'metadata': metadata
        }
        
        print(f"  Generated data with {np.sum(np.diff(regime_sequence) != 0)} regime switches")
        print(f"  Triple interaction effects: mean = {np.mean(triple_interaction_evolution):.4f}")
        
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
    
    def _create_block_correlation_matrix(self, block_indices, within_corr, between_corr):
        """
        Create block-structured correlation matrix
        
        Parameters:
        -----------
        block_indices : list
            List of lists containing indices for each block
        within_corr : float
            Correlation within blocks
        between_corr : float
            Correlation between blocks
            
        Returns:
        --------
        corr_matrix : np.ndarray
            Block-structured correlation matrix
        """
        
        corr_matrix = np.eye(self.dim)
        
        # Set within-block correlations
        for block in block_indices:
            for i in block:
                for j in block:
                    if i != j:
                        corr_matrix[i, j] = within_corr
        
        # Set between-block correlations
        for block1_idx, block1 in enumerate(block_indices):
            for block2_idx, block2 in enumerate(block_indices):
                if block1_idx != block2_idx:
                    for i in block1:
                        for j in block2:
                            corr_matrix[i, j] = between_corr
        
        # Ensure positive definiteness
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
    
    def _make_positive_definite_robust(self, corr_matrix):
        """
        Robust method to ensure correlation matrix is positive definite
        
        This is an enhanced version that properly handles eigenvalue adjustment
        and rescaling to maintain unit diagonal.
        
        Parameters:
        -----------
        corr_matrix : np.ndarray
            Correlation matrix to make positive definite
            
        Returns:
        --------
        corr_pd : np.ndarray
            Positive definite correlation matrix
        """
        
        # Compute eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        
        # Adjust negative/small eigenvalues
        eigenvals = np.where(eigenvals < 1e-8, 1e-8, eigenvals)
        
        # Reconstruct matrix
        corr_pd = (eigenvecs * eigenvals) @ eigenvecs.T
        
        # Rescale to unit diagonal
        diag = np.sqrt(np.diag(corr_pd))
        for i in range(len(diag)):
            for j in range(len(diag)):
                corr_pd[i, j] /= (diag[i] * diag[j] + 1e-12)
        
        return corr_pd
    
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
        elif metadata['type'] == 'block_switching_correlation':
            # Plot regime sequence and entropy evolution
            ax_regime = axes[1, 2]
            ax_entropy = ax_regime.twinx()
            
            # Regime sequence
            regime_line = ax_regime.plot(time_indices, metadata['regime_sequence'], 
                                       'b-', linewidth=2, label='Regime')
            ax_regime.set_xlabel('Time')
            ax_regime.set_ylabel('Regime ID', color='b')
            ax_regime.tick_params(axis='y', labelcolor='b')
            
            # Entropy evolution
            entropy_line = ax_entropy.plot(time_indices, metadata['entropy_evolution'], 
                                         'r--', linewidth=2, label='Entropy')
            ax_entropy.set_ylabel('Entropy (bits)', color='r')
            ax_entropy.tick_params(axis='y', labelcolor='r')
            
            # Mark regime switches
            switches = np.where(np.diff(metadata['regime_sequence']) != 0)[0] + 1
            for switch_time in switches:
                ax_regime.axvline(x=switch_time, color='gray', linestyle=':', alpha=0.7)
            
            ax_regime.set_title('Regime Switches & Entropy Evolution')
            ax_regime.grid(True, alpha=0.3)
        elif metadata['type'] == 'beyond_pairwise_interactions':
            # Plot regime sequence and triple interaction strength
            ax_regime = axes[1, 2]
            ax_triple = ax_regime.twinx()
            
            # Regime sequence
            regime_line = ax_regime.plot(time_indices, metadata['regime_sequence'], 
                                       'b-', linewidth=2, label='Regime')
            ax_regime.set_xlabel('Time')
            ax_regime.set_ylabel('Regime ID', color='b')
            ax_regime.tick_params(axis='y', labelcolor='b')
            
            # Triple interaction evolution
            triple_line = ax_triple.plot(time_indices, metadata['triple_interaction_evolution'], 
                                       'g--', linewidth=2, label='Triple Effect')
            ax_triple.set_ylabel('Triple Interaction |Effect|', color='g')
            ax_triple.tick_params(axis='y', labelcolor='g')
            
            # Mark regime switches with clean vertical lines
            if 'switch_indices' in metadata:
                for switch_idx in metadata['switch_indices']:
                    ax_regime.axvline(x=switch_idx, color='red', linestyle='--', alpha=0.7, linewidth=2)
            
            ax_regime.set_title('Regime Switches & Triple Interactions')
            ax_regime.grid(True, alpha=0.3)
        else:
            # Generic time series plot
            sample_data = data[:, 0, :min(5, self.dim)]  # First sample, first 5 variables
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
        
        # Create additional detailed visualization for block switching data
        if metadata['type'] == 'block_switching_correlation':
            self._create_block_correlation_detailed_plots(dataset_name, save_plots)
    
    def _create_block_correlation_detailed_plots(self, dataset_name, save_plots):
        """
        Create additional detailed visualizations for block switching data
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to visualize
        save_plots : bool
            Whether to save plots to disk
        """
        
        dataset = self.generated_datasets[dataset_name]
        data = dataset['data']
        time_indices = dataset['time_indices']
        metadata = dataset['metadata']
        
        # Create comprehensive block analysis visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Block-Structured Correlation Analysis: {dataset_name}', fontsize=16)
        
        # 1. Correlation matrix evolution heatmap
        correlation_matrices = np.array(metadata['correlation_matrices'])
        n_time_points = min(50, len(time_indices))
        time_subset = np.linspace(0, len(time_indices)-1, n_time_points, dtype=int)
        
        # Show full correlation matrix evolution
        corr_evolution = correlation_matrices[time_subset]
        
        # Flatten correlation matrices for visualization
        corr_flat = corr_evolution.reshape(n_time_points, -1)
        im1 = axes[0, 0].imshow(corr_flat.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, 0].set_title('Correlation Matrix Evolution')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Matrix Elements')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Block structure visualization at specific time points
        n_examples = min(3, len(correlation_matrices))
        example_times = np.linspace(0, len(correlation_matrices)-1, n_examples, dtype=int)
        
        for i, t_idx in enumerate(example_times):
            if i < 3:  # Ensure we don't exceed subplot limits
                row, col = (0, 1), (0, 2), (1, 0)
                ax = axes[row[i]][col[i]] if i < 2 else axes[1][0]
                
                corr_mat = correlation_matrices[t_idx]
                im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1)
                ax.set_title(f'Correlation Matrix at t={t_idx}\nRegime: {metadata["regime_sequence"][t_idx]}')
                
                # Add block boundaries
                block_boundaries = np.cumsum([0] + metadata['block_sizes'][:-1])
                for boundary in block_boundaries[1:]:
                    ax.axhline(y=boundary-0.5, color='black', linewidth=2)
                    ax.axvline(x=boundary-0.5, color='black', linewidth=2)
                
                plt.colorbar(im, ax=ax)
        
        # 3. Within-block vs between-block correlation evolution
        within_block_corrs = []
        between_block_corrs = []
        
        for t in range(len(correlation_matrices)):
            corr_mat = correlation_matrices[t]
            block_indices = metadata['block_indices']
            
            # Calculate average within-block correlation
            within_corrs = []
            for block in block_indices:
                if len(block) > 1:
                    block_corrs = []
                    for i in block:
                        for j in block:
                            if i != j:
                                block_corrs.append(corr_mat[i, j])
                    if block_corrs:
                        within_corrs.append(np.mean(block_corrs))
            
            # Calculate average between-block correlation
            between_corrs = []
            for i, block1 in enumerate(block_indices):
                for j, block2 in enumerate(block_indices):
                    if i != j:
                        for idx1 in block1:
                            for idx2 in block2:
                                between_corrs.append(corr_mat[idx1, idx2])
            
            within_block_corrs.append(np.mean(within_corrs) if within_corrs else 0)
            between_block_corrs.append(np.mean(between_corrs) if between_corrs else 0)
        
        axes[1, 1].plot(time_indices, within_block_corrs, 'b-', linewidth=2, label='Within-block')
        axes[1, 1].plot(time_indices, between_block_corrs, 'r-', linewidth=2, label='Between-block')
        
        # Mark regime switches
        switches = np.where(np.diff(metadata['regime_sequence']) != 0)[0] + 1
        for switch_time in switches:
            axes[1, 1].axvline(x=switch_time, color='gray', linestyle=':', alpha=0.7)
        
        axes[1, 1].set_title('Block Correlation Evolution')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 4. Entropy evolution and regime analysis
        entropy_evolution = metadata['entropy_evolution']
        regime_sequence = metadata['regime_sequence']
        
        # Plot entropy evolution with regime coloring
        colors = plt.cm.tab10(regime_sequence / max(regime_sequence))
        scatter = axes[1, 2].scatter(time_indices, entropy_evolution, c=colors, alpha=0.7)
        axes[1, 2].plot(time_indices, entropy_evolution, 'k-', alpha=0.3)
        
        # Mark regime switches
        for switch_time in switches:
            axes[1, 2].axvline(x=switch_time, color='gray', linestyle=':', alpha=0.7)
        
        axes[1, 2].set_title('Entropy Evolution by Regime')
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Entropy (bits)')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add regime legend
        unique_regimes = np.unique(regime_sequence)
        for regime in unique_regimes:
            regime_mask = regime_sequence == regime
            avg_entropy = np.nanmean(np.array(entropy_evolution)[regime_mask])
            axes[1, 2].scatter([], [], c=[plt.cm.tab10(regime / max(regime_sequence))], 
                             label=f'Regime {regime} (H={avg_entropy:.2f})')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.results_dir, f'{dataset_name}_detailed_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved detailed plot: {plot_path}")
        
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