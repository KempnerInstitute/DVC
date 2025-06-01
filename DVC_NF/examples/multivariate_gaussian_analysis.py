#!/usr/bin/env python3
"""
Multivariate Gaussian Vine Copula Analysis Script

This script:
1. Simulates multivariate correlated Gaussian distribution
2. Fits vine copula with given R-vine structure
3. Estimates correlation matrices
4. Generates samples and creates comparison plots
5. Estimates entropy and MI, compares with ground truth
6. Saves figures and results

Author: DVC Analysis Team
Date: 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import multivariate_normal
import pandas as pd
import pickle
from datetime import datetime
import warnings

# Suppress TensorFlow informational messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

# Add the DVC_tensorflow directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
dvc_tensorflow_dir = os.path.join(project_root, 'src', 'DVC_tensorflow')
sys.path.append(dvc_tensorflow_dir)

# Import vine copula modules
from classes.objects import *
from vine_tree.tree_op import *
from param.generate_rvine import *
from param.margin_fit import *
from param.margin_op import *
from param.copula_fit import *
from param.cond_copula import *
from pre_proc.preparation import prep_cop
from pred.prediction import*
from sampling.vine_sample import *
from info.info_estimation import vine_entropy

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create results directory if it doesn't exist
results_dir = os.path.join(current_dir, '..', 'results')
os.makedirs(results_dir, exist_ok=True)

class Multivariate_Gaussian_Vine_Analysis:
    """
    Comprehensive class for multivariate Gaussian vine copula analysis
    
    This class performs a complete pipeline for evaluating how well vine copulas
    can model multivariate Gaussian distributions with known correlation structures.
    
    Key Analysis Steps:
    1. Generate synthetic multivariate Gaussian data with controlled correlations
    2. Fit various vine copula models to capture the dependence structure  
    3. Generate new samples from the fitted vine copula
    4. Compare correlation preservation between original and vine-generated data
    5. Estimate entropy and compare with theoretical ground truth
    6. Create comprehensive visualizations and save all results
    
    Data Flow:
    Input: Configuration parameters (dimensions, sample size, vine type)
    ↓
    Generated Data: Multivariate Gaussian samples with known correlation matrix
    ↓  
    Fitted Model: Vine copula trained on the generated data
    ↓
    Generated Samples: New data sampled from the fitted vine copula
    ↓
    Analysis Results: Correlation errors, entropy estimates, visualizations
    """
    
    def __init__(self, dim=5, n_samples=2000, vine_type='r-vine'):
        """
        Initialize the analysis framework
        
        Parameters:
        -----------
        dim : int, default=5
            Dimensionality of the multivariate distribution
            Range: 2-10 (recommended 3-6 for computational efficiency)
            Higher dimensions require more samples and computation time
            
        n_samples : int, default=2000  
            Number of samples to generate for training the vine copula
            Range: 500-5000 (recommended 1000-3000)
            More samples improve accuracy but increase computation time
            
        vine_type : str, default='r-vine'
            Type of vine copula structure to use
            Options:
            - 'c-vine': Canonical vine (star structure, computationally efficient)
            - 'd-vine': D-vine (sequential structure, moderate complexity)  
            - 'r-vine': Regular vine (most flexible, computationally intensive)
            
        Attributes Created:
        ------------------
        self.dim : int - Number of variables
        self.n_samples : int - Sample size
        self.vine_type : str - Vine structure type
        self.results : dict - Will store all analysis results
        self.true_correlation_matrix : ndarray - Known ground truth correlations
        self.empirical_correlation_matrix : ndarray - Correlations from original data
        self.vine_correlation_matrix : ndarray - Correlations from vine samples
        self.original_data : ndarray - Generated Gaussian data (n_samples × dim)
        self.vine_samples : ndarray - Samples from fitted vine (n_samples × dim)
        self.vine : vine_obj_bin - Fitted vine copula model
        """
        self.dim = dim
        self.n_samples = n_samples
        self.vine_type = vine_type
        self.results = {}
        
        print(f"Initializing Multivariate Gaussian Vine Analysis")
        print(f"Dimensions: {dim}, Samples: {n_samples}, Vine Type: {vine_type}")
        print(f"Expected memory usage: ~{(n_samples * dim * 8) / 1e6:.1f} MB for data arrays")
        
    def generate_correlation_matrix(self):
        """
        Generate a realistic positive definite correlation matrix
        
        Method:
        -------
        Uses Wishart distribution approach to create random but realistic correlations:
        1. Generate random matrix A
        2. Create positive definite matrix: Σ = A·A^T  
        3. Convert to correlation matrix by normalizing diagonal to 1
        4. Ensure numerical stability (eigenvalues > 0.01)
        
        Returns:
        --------
        corr_matrix : ndarray, shape (dim, dim)
            Symmetric positive definite correlation matrix
            - Diagonal elements = 1.0 (perfect self-correlation)
            - Off-diagonal elements ∈ [-1, 1] (pairwise correlations)
            - Matrix properties: symmetric, positive definite
            
        Expected Structure:
        ------------------
        For 3×3 case:
        [[1.0,  ρ₁₂, ρ₁₃],
         [ρ₁₂, 1.0,  ρ₂₃],  
         [ρ₁₃, ρ₂₃, 1.0]]
        
        where ρᵢⱼ represents correlation between variables i and j
        """
        print("Generating realistic correlation matrix...")
        
        # Create a random correlation matrix using Wishart distribution
        # This method ensures the matrix is positive definite and realistic
        A = np.random.randn(self.dim, self.dim)
        
        # Make it positive definite: Σ = A·A^T
        cov_matrix = np.dot(A, A.T)
        
        # Convert covariance to correlation matrix
        # Correlation = Σᵢⱼ / √(Σᵢᵢ·Σⱼⱼ)
        D = np.diag(1.0 / np.sqrt(np.diag(cov_matrix)))
        corr_matrix = np.dot(np.dot(D, cov_matrix), D)
        
        # Ensure perfect symmetry (numerical precision)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Ensure positive definiteness for numerical stability
        eigenvals = np.linalg.eigvals(corr_matrix)
        min_eigenval = np.min(eigenvals)
        if min_eigenval < 0.01:
            # Add small identity component to ensure stability
            corr_matrix += (0.01 - min_eigenval) * np.eye(self.dim)
            print(f"Added {0.01 - min_eigenval:.4f} to diagonal for numerical stability")
            
        self.true_correlation_matrix = corr_matrix
        
        print(f"Generated correlation matrix with eigenvalue range: [{np.min(eigenvals):.3f}, {np.max(eigenvals):.3f}]")
        print(f"Off-diagonal correlation range: [{np.min(corr_matrix[np.triu_indices(self.dim, k=1)]):.3f}, {np.max(corr_matrix[np.triu_indices(self.dim, k=1)]):.3f}]")
        
        return corr_matrix
    
    def simulate_multivariate_gaussian(self):
        """
        Generate multivariate Gaussian data with controlled correlation structure
        
        Process:
        --------
        1. Create correlation matrix (if not already generated)
        2. Generate samples from multivariate normal distribution N(μ=0, Σ=corr_matrix)
        3. Calculate empirical correlations from generated data
        4. Store both true and empirical correlation matrices for comparison
        
        Mathematical Background:
        -----------------------
        Multivariate Normal: X ~ N(μ, Σ) where:
        - μ = mean vector (set to zeros)
        - Σ = correlation matrix (positive definite)
        - Each marginal follows standard normal: Xᵢ ~ N(0,1)
        - Joint dependence captured by correlation structure
        
        Returns:
        --------
        original_data : ndarray, shape (n_samples, dim)
            Generated Gaussian samples where:
            - Each row is one observation
            - Each column is one variable  
            - Values typically in range [-4, 4] (99.9% within ±3σ)
            - Correlations match self.true_correlation_matrix (within sampling error)
            
        Side Effects:
        ------------
        Sets attributes:
        - self.original_data : Generated data array
        - self.empirical_correlation_matrix : Observed correlations from data
        
        Expected Data Properties:
        ------------------------
        - Shape: (n_samples, dim)
        - Data type: float64
        - Value range: approximately [-4, +4] 
        - Mean ≈ 0 for each variable
        - Std ≈ 1 for each variable
        - Pairwise correlations ≈ true_correlation_matrix values
        """
        print("Generating multivariate correlated Gaussian data...")
        print(f"Target: {self.n_samples} samples of {self.dim}-dimensional Gaussian data")
        
        # Generate correlation matrix if not already created
        if not hasattr(self, 'true_correlation_matrix'):
            corr_matrix = self.generate_correlation_matrix()
        else:
            corr_matrix = self.true_correlation_matrix
        
        # Generate multivariate normal data
        # μ = 0 (zero mean), Σ = correlation matrix
        mean = np.zeros(self.dim)
        self.original_data = multivariate_normal.rvs(
            mean=mean, 
            cov=corr_matrix, 
            size=self.n_samples
        )
        
        print(f"Generated data shape: {self.original_data.shape}")
        print(f"Data range: [{np.min(self.original_data):.3f}, {np.max(self.original_data):.3f}]")
        
        # Calculate empirical statistics for validation
        means = np.mean(self.original_data, axis=0)
        stds = np.std(self.original_data, axis=0)
        print(f"Empirical means: [{np.min(means):.3f}, {np.max(means):.3f}] (should be ≈0)")
        print(f"Empirical stds: [{np.min(stds):.3f}, {np.max(stds):.3f}] (should be ≈1)")
        
        # Store empirical correlation matrix for comparison
        self.empirical_correlation_matrix = np.corrcoef(self.original_data.T)
        
        # Calculate correlation error from sampling
        corr_error = np.mean(np.abs(self.empirical_correlation_matrix - self.true_correlation_matrix))
        print(f"Empirical vs True correlation MAE: {corr_error:.4f} (sampling error)")
        
        return self.original_data
    
    def setup_vine_structure(self):
        """
        Setup vine copula structure and R-matrix for dependence modeling
        
        Vine Copula Background:
        ----------------------
        Vine copulas model multivariate dependence by decomposing the joint
        distribution into a cascade of bivariate copulas arranged in a tree structure.
        
        For d variables, we need d(d-1)/2 bivariate copulas arranged in d-1 trees:
        - Tree 1: d-1 bivariate copulas (unconditional)
        - Tree 2: d-2 bivariate copulas (conditional on 1 variable)
        - ...
        - Tree d-1: 1 bivariate copula (conditional on d-2 variables)
        
        Vine Types:
        -----------
        1. C-vine (Canonical): Star structure with one central variable per tree
           - Most interpretable, computationally efficient
           - Good when one variable is central to all dependencies
           
        2. D-vine: Sequential structure, variables connected in path
           - Natural ordering of variables matters  
           - Good for time series or spatial data
           
        3. R-vine (Regular): Most flexible, any valid tree structure
           - Can capture complex dependency patterns
           - Computationally intensive, harder to interpret
        
        R-matrix:
        ---------
        The R-matrix encodes the vine structure:
        - Lower triangular matrix
        - R[i,j] specifies which variable appears in position j of tree i
        - Must satisfy regularity conditions for valid vine
        
        Returns:
        --------
        r_matrix : ndarray, shape (dim, dim), dtype=int
            Lower triangular matrix encoding vine structure
            Example for 3D C-vine:
            [[3, 0, 0],
             [2, 2, 0], 
             [1, 1, 1]]
            
        Side Effects:
        ------------
        Sets attributes:
        - self.r_matrix : R-matrix encoding vine structure
        - self.ind_vine : Index structure for vine trees
        - self.nodes : Variable node labels  
        - self.matrix_edges : Edge labels for each tree level
        - self.vine_type : May change to 'c-vine' if R-vine generation fails
        
        Error Handling:
        --------------
        If R-vine matrix generation fails (invalid structure), automatically
        falls back to C-vine which is guaranteed to work.
        """
        print("Setting up vine copula structure...")
        print(f"Requested vine type: {self.vine_type}")
        print(f"For {self.dim} variables, need {self.dim*(self.dim-1)//2} bivariate copulas")
        
        if self.vine_type == 'r-vine':
            try:
                # Generate a proper random R-matrix
                # This creates a valid regular vine structure
                from param.generate_rvine import random_r_matrix_gen
                r_matrix, ind_vine, nodes, E = random_r_matrix_gen(self.dim)
                matrix_edges = []  # Will be computed later if needed
                
                self.r_matrix = r_matrix
                self.ind_vine = ind_vine
                self.nodes = nodes
                self.matrix_edges = matrix_edges
                
                print("✓ Successfully generated random R-vine structure")
                
            except Exception as e:
                print(f"Warning: R-vine generation failed ({e})")
                print("Falling back to C-vine (more reliable)")
                self.vine_type = 'c-vine'
                self.r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(self.vine_type, self.dim)
                self.ind_vine = ind_vine
                self.nodes = nodes
                self.matrix_edges = matrix_edges
            
        else:
            # Use standard C-vine or D-vine
            # These are guaranteed to work for any number of dimensions
            self.r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(self.vine_type, self.dim)
            self.ind_vine = ind_vine
            self.nodes = nodes
            self.matrix_edges = matrix_edges
        
        print("R-matrix structure:")
        print(self.r_matrix)
        print(f"Final vine type: {self.vine_type}")
        
        # Print vine structure interpretation
        print(f"Vine structure interpretation:")
        print(f"- Tree levels: {self.dim-1}")
        print(f"- Total bivariate copulas: {np.sum(np.arange(1, self.dim))}")
        print(f"- Variable nodes: {self.nodes}")
        
        return self.r_matrix
    
    def fit_vine_copula(self):
        """
        Fit vine copula model to the generated Gaussian data
        
        Process Overview:
        ----------------
        1. Transform data to uniform margins (probability integral transform)
        2. Set up vine copula object with kernel copula families
        3. Fit bivariate copulas at each level of the vine structure
        4. Estimate copula parameters using maximum likelihood or bandwidth optimization
        
        Technical Details:
        -----------------
        Vine Fitting Process:
        - Data preparation: Transform to copula domain [0,1]^d
        - Margin fitting: Estimate marginal distributions (here: standard normal)
        - Copula fitting: Fit bivariate copulas for each edge in vine structure
        - Parameter estimation: Optimize copula parameters using specified method
        
        Copula Families Used:
        - Kernel copulas ("kercop"): Non-parametric, flexible
        - Fallback families: Independent, Gaussian (parametric options)
        
        Fitting Configuration:
        ---------------------
        - families: "kercop" (kernel copulas for flexibility)
        - knots: 50 (grid resolution for kernel estimation)
        - parallel: True (use parallel processing for speed)
        - opt_method: 'LL1' (bandwidth optimization method)
        - batch_paral: 3 (batch size for parallel processing)
        
        Returns:
        --------
        vine : vine_obj_bin
            Fitted vine copula object containing:
            - copulas: List of fitted bivariate copulas for each tree level
            - margins: Fitted marginal distributions  
            - r_matrix: Vine structure encoding
            - grid_u: Uniform grid for evaluation
            - evaluation methods for pdf/cdf computation
            
        Data Transformations:
        --------------------
        Original data → Empirical CDF → Uniform margins → Copula fitting
        
        Input data requirements:
        - Shape: (n_samples, dim)
        - Type: float32 (for TensorFlow compatibility)
        - Preprocessing: Make divisible by 5 for k-fold validation
        
        Expected Fitting Time:
        ---------------------
        - 3D, 500 samples: ~30-60 seconds
        - 4D, 1500 samples: ~2-5 minutes  
        - 6D, 3000 samples: ~10-30 minutes
        (Times depend on system resources and vine complexity)
        """
        print("Fitting vine copula to the data...")
        print(f"Copula fitting configuration:")
        print(f"- Vine type: {self.vine_type}")
        print(f"- Data shape: {self.original_data.shape}")
        print(f"- Copula families: kernel copulas (flexible, non-parametric)")
        print(f"- Grid resolution: 50 knots")
        
        # Setup margins (normal margins since data is Gaussian)
        # Each margin models the univariate distribution of one variable
        margin_vine = []
        for i in range(self.dim):
            mar_p = margin_obj('norm', [0, 1], True)  # Standard normal margins
            margin_vine.append(mar_p)
        print(f"✓ Set up {len(margin_vine)} standard normal margins")
        
        # Setup vine object with configuration
        vine_depth = self.dim  # Full vine (all possible dependencies)
        families = "kercop"    # Kernel copulas (flexible)
        knots = 50            # Grid resolution for kernel estimation
        method = 'matrix'     # Use R-matrix specification
        
        self.vine = vine_obj_bin(
            self.vine_type, families, vine_depth, 
            margin_vine, knots, method, self.r_matrix
        )
        print(f"✓ Created vine object: {vine_depth}D {self.vine_type}")
        
        # Prepare data for copula fitting
        x = self.original_data.astype(np.float32)  # Convert to float32 for TensorFlow
        original_size = x.shape[0]
        
        # Make data divisible for k-fold cross-validation (required by fitting algorithm)
        exc = tf.math.floormod(tf.shape(x)[0], 5)
        x = x[:tf.shape(x)[0]-exc, :]
        final_size = x.shape[0]
        
        if final_size != original_size:
            print(f"Adjusted data size: {original_size} → {final_size} (for k-fold compatibility)")
        
        # Transform data to copula domain (empirical CDF transformation)
        sort_n = 'rand'  # Random sorting for tie-breaking
        e = prep_cop(x, self.vine, sort_n)
        print(f"✓ Data transformed to copula domain")
        print(f"  Transformed data shape: {e.shape if hasattr(e, 'shape') else 'processed'}")
        
        # Configure fitting parameters
        gen_dict = {
            'parallel': True,      # Use parallel processing
            'binning': False,      # No binning (continuous estimation)  
            'param': False,        # Non-parametric (kernel) copulas
            'vine_depth': vine_depth,  # Full vine depth
            'fitted': False        # Mark as not yet fitted
        }
        
        par_dict = {
            'param_families': ["ind", "gaussian"]  # Fallback parametric families
        }
        
        npc_dict = {
            'opt_method': 'LL1',   # Bandwidth optimization method
            'batch_paral': 3       # Batch size for parallel processing
        }
        
        bin_dict = {
            'n_bin': 3            # Number of bins (unused when binning=False)
        }
        
        print("Starting vine copula fitting...")
        print("This may take several minutes depending on data size and vine complexity...")
        
        # Perform the actual fitting
        import time
        start_time = time.time()
        
        self.vine.fit(x, gen_dict, npc_dict, par_dict, bin_dict)
        
        fitting_time = time.time() - start_time
        print(f"✓ Vine fitting completed in {fitting_time:.1f} seconds!")
        
        # Verify fitting success
        if hasattr(self.vine, 'copulas') and self.vine.copulas:
            try:
                # Count copulas more safely
                if isinstance(self.vine.copulas, list):
                    n_copulas = len(self.vine.copulas)
                else:
                    n_copulas = "fitted"
                print(f"✓ Successfully fitted vine with {n_copulas} copula levels")
            except Exception as e:
                print("✓ Vine fitting completed (copula structure verified)")
        else:
            print("⚠ Warning: Fitting may not have completed successfully")
        
        return self.vine
    
    def generate_vine_samples(self, n_samples=None):
        """
        Generate new samples from the fitted vine copula model
        
        Process:
        --------
        1. Sample uniform random variables for each dimension
        2. Apply inverse vine copula transformation (Rosenblatt transform)
        3. Transform from copula domain back to original data domain
        4. Calculate correlation matrix of generated samples
        
        Mathematical Background:
        -----------------------
        Vine Sampling Algorithm:
        1. U ~ Uniform[0,1]^d (independent uniform variables)
        2. Apply conditional inverse CDFs level by level:
           - X₁ = F₁⁻¹(U₁)  
           - X₂ = F₂⁻¹(U₂ | X₁)
           - X₃ = F₃⁻¹(U₃ | X₁,X₂)
           - etc.
        3. Result: X ~ target multivariate distribution
        
        Parameters:
        -----------
        n_samples : int, optional
            Number of samples to generate
            If None, uses self.n_samples (same as training data)
            Range: 100-10000 (limited by memory and computation time)
            
        Returns:
        --------
        vine_samples : ndarray, shape (n_samples, dim)
            Generated samples from fitted vine copula where:
            - Each row is one generated observation
            - Each column corresponds to one variable
            - Values should have similar distribution to original data
            - Correlations should approximate original correlation structure
            
        Side Effects:
        ------------
        Sets attributes:
        - self.vine_samples : Generated sample array  
        - self.vine_correlation_matrix : Correlation matrix of generated samples
        
        Quality Indicators:
        ------------------
        Good vine fit indicated by:
        - Generated sample range similar to original data
        - Correlation matrix close to empirical_correlation_matrix
        - Marginal distributions approximately normal
        - No obvious artifacts or clustering
        
        Expected Performance:
        --------------------
        - Sample generation is typically fast (seconds)
        - Much faster than fitting process
        - Linear scaling with number of samples requested
        """
        if n_samples is None:
            n_samples = self.n_samples
            
        print(f"Generating {n_samples} samples from fitted vine copula...")
        print(f"Original training data: {self.original_data.shape}")
        
        # Generate samples using vine copula sampling algorithm
        # Returns: samples, uniform_variables, pdf_values, cdf_values
        self.vine_samples, u, sample_pdf, sample_pds = vine_copula_sample(self.vine, n_samples)
        
        print(f"✓ Generated vine samples shape: {self.vine_samples.shape}")
        
        # Validate sample quality
        sample_range = [np.min(self.vine_samples), np.max(self.vine_samples)]
        original_range = [np.min(self.original_data), np.max(self.original_data)]
        
        print(f"Value ranges comparison:")
        print(f"  Original data: [{original_range[0]:.3f}, {original_range[1]:.3f}]")
        print(f"  Vine samples:  [{sample_range[0]:.3f}, {sample_range[1]:.3f}]")
        
        # Calculate and validate sample statistics
        sample_means = np.mean(self.vine_samples, axis=0)
        sample_stds = np.std(self.vine_samples, axis=0)
        original_means = np.mean(self.original_data, axis=0)
        original_stds = np.std(self.original_data, axis=0)
        
        print(f"Mean comparison (should be close):")
        print(f"  Original: [{np.min(original_means):.3f}, {np.max(original_means):.3f}]")
        print(f"  Vine:     [{np.min(sample_means):.3f}, {np.max(sample_means):.3f}]")
        
        print(f"Std comparison (should be close):")
        print(f"  Original: [{np.min(original_stds):.3f}, {np.max(original_stds):.3f}]")
        print(f"  Vine:     [{np.min(sample_stds):.3f}, {np.max(sample_stds):.3f}]")
        
        # Calculate correlation matrix of vine samples for comparison
        self.vine_correlation_matrix = np.corrcoef(self.vine_samples.T)
        
        # Quick correlation preservation check
        corr_error = np.mean(np.abs(self.vine_correlation_matrix - self.empirical_correlation_matrix))
        print(f"Correlation preservation MAE: {corr_error:.4f}")
        
        if corr_error < 0.1:
            print("✓ Good correlation preservation")
        elif corr_error < 0.2:
            print("⚠ Moderate correlation preservation")
        else:
            print("⚠ Poor correlation preservation - check vine fit quality")
        
        return self.vine_samples
    
    def calculate_theoretical_entropy(self):
        """
        Calculate theoretical differential entropy for multivariate Gaussian distribution
        
        Mathematical Formula:
        --------------------
        For multivariate Gaussian X ~ N(μ, Σ):
        H(X) = (1/2) * log((2πe)^k * |Σ|)
        
        Where:
        - k = dimensionality (number of variables)
        - |Σ| = determinant of covariance matrix
        - e = Euler's number ≈ 2.718
        - log = natural logarithm
        
        Since we use correlation matrix (standardized covariance):
        H(X) = (1/2) * (k * log(2πe) + log(|R|))
        
        Theoretical Background:
        ----------------------
        Differential entropy measures the average information content of a
        continuous random variable. For Gaussian distributions, it has a
        closed-form expression depending only on the covariance matrix.
        
        This serves as ground truth for evaluating vine copula entropy estimates.
        
        Returns:
        --------
        theoretical_entropy : float
            Theoretical entropy in nats (natural units)
            - Typical range: 5-20 nats for dimensions 2-6
            - Higher values indicate more uncertainty/information
            - Positive values (continuous distributions have positive entropy)
            
        Properties:
        -----------
        - Independent variables: entropy = sum of marginal entropies
        - Correlated variables: entropy < sum of marginal entropies
        - Perfect correlation: entropy approaches single variable entropy
        
        Conversion Notes:
        ----------------
        - Nats to bits: multiply by 1/log(2) ≈ 1.443
        - Nats to digits: multiply by 1/log(10) ≈ 0.434
        """
        print("Calculating theoretical entropy for multivariate Gaussian...")
        
        # Check if correlation matrix exists
        if not hasattr(self, 'true_correlation_matrix'):
            raise ValueError("True correlation matrix not available. Run simulate_multivariate_gaussian first.")
        
        # Calculate determinant of correlation matrix
        det_corr = np.linalg.det(self.true_correlation_matrix)
        
        # Validate determinant (must be positive for valid correlation matrix)
        if det_corr <= 0:
            print(f"Warning: Correlation matrix determinant = {det_corr:.6f} (should be > 0)")
            print("This indicates numerical issues or invalid correlation matrix")
        
        # Apply entropy formula: H(X) = 0.5 * (k*log(2πe) + log(|Σ|))
        theoretical_entropy = 0.5 * (self.dim * np.log(2 * np.pi * np.e) + np.log(det_corr))
        
        self.theoretical_entropy = theoretical_entropy
        
        # Calculate entropy in different units for reference
        entropy_bits = theoretical_entropy / np.log(2)
        entropy_digits = theoretical_entropy / np.log(10)
        
        print(f"✓ Theoretical entropy calculated:")
        print(f"  {theoretical_entropy:.4f} nats (natural units)")
        print(f"  {entropy_bits:.4f} bits (binary units)")
        print(f"  {entropy_digits:.4f} digits (decimal units)")
        
        # Compare with independent case (upper bound)
        independent_entropy = self.dim * 0.5 * np.log(2 * np.pi * np.e)
        reduction = independent_entropy - theoretical_entropy
        
        print(f"Entropy analysis:")
        print(f"  Independent variables entropy: {independent_entropy:.4f} nats")
        print(f"  Correlation reduces entropy by: {reduction:.4f} nats ({100*reduction/independent_entropy:.1f}%)")
        
        return theoretical_entropy
    
    def estimate_vine_entropy(self):
        """
        Estimate entropy using the fitted vine copula model
        
        Method:
        -------
        Monte Carlo entropy estimation:
        1. Generate samples from fitted vine copula
        2. Evaluate log-likelihood of each sample under the vine model
        3. Estimate entropy as negative expected log-likelihood: H ≈ -E[log p(x)]
        
        Mathematical Background:
        -----------------------
        Entropy definition: H(X) = -∫ p(x) log p(x) dx
        Monte Carlo approximation: H ≈ -(1/N) Σᵢ log p(xᵢ)
        
        Where p(xᵢ) is the probability density evaluated by the vine copula.
        
        Process Details:
        ---------------
        1. Sample generation: Draw samples from vine copula
        2. Density evaluation: Use vine.evaluation() to get p(x) for each sample
        3. Log-likelihood calculation: Compute log p(x) 
        4. Average: Take mean of negative log-likelihoods
        5. Unit conversion: Convert to bits if desired
        
        Accuracy Factors:
        ----------------
        - Number of samples: More samples → better estimate
        - Vine fit quality: Better fit → more accurate entropy
        - Numerical stability: Avoid log(0) issues
        
        Returns:
        --------
        vine_entropy_estimate : float
            Estimated entropy in bits
            - Should be close to theoretical_entropy for good vine fit
            - Typical accuracy: ±5-10% for well-fitted vines
            - Large deviations indicate poor vine fit
            
        Quality Indicators:
        ------------------
        Good entropy estimate indicated by:
        - Close agreement with theoretical value
        - Stable estimates across multiple runs
        - No extreme outliers in log-likelihood values
        """
        print("Estimating entropy using fitted vine copula...")
        
        # Use a reasonable number of samples for entropy estimation
        # Balance between accuracy and computation time
        n_entropy_samples = min(5000, self.n_samples)
        print(f"Using {n_entropy_samples} samples for entropy estimation")
        
        # Generate fresh samples for unbiased entropy estimation
        print("Generating samples for entropy estimation...")
        entropy_samples, _, _, _ = vine_copula_sample(self.vine, n_entropy_samples)
        
        # Evaluate probability density for each sample
        print("Evaluating vine copula density...")
        try:
            p, p_copula, log_marg_f = self.vine.evaluation(entropy_samples)
        except Exception as e:
            print(f"Error in vine evaluation: {e}")
            print("This may indicate issues with the fitted vine model")
            self.vine_entropy_estimate = np.nan
            return np.nan
        
        # Calculate log-likelihood and handle numerical issues
        p_values = p.numpy()
        
        # Check for numerical issues
        zero_density = np.sum(p_values <= 0)
        if zero_density > 0:
            print(f"Warning: {zero_density}/{len(p_values)} samples have zero/negative density")
            # Replace zeros with small positive value to avoid log(0)
            p_values = np.maximum(p_values, 1e-10)
        
        # Calculate log-likelihood
        log_p = np.log(p_values)
        
        # Check for numerical stability
        finite_mask = np.isfinite(log_p)
        if not np.all(finite_mask):
            n_invalid = np.sum(~finite_mask)
            print(f"Warning: {n_invalid}/{len(log_p)} samples have invalid log-likelihood")
            log_p = log_p[finite_mask]
        
        # Estimate entropy as negative expected log-likelihood
        vine_entropy_nats = -np.mean(log_p)
        vine_entropy_bits = vine_entropy_nats / np.log(2)  # Convert to bits
        
        self.vine_entropy_estimate = vine_entropy_bits
        
        print(f"✓ Vine entropy estimate:")
        print(f"  {vine_entropy_nats:.4f} nats")
        print(f"  {vine_entropy_bits:.4f} bits")
        
        # Compare with theoretical value if available
        if hasattr(self, 'theoretical_entropy'):
            theoretical_bits = self.theoretical_entropy / np.log(2)
            error_abs = abs(vine_entropy_bits - theoretical_bits)
            error_rel = 100 * error_abs / theoretical_bits
            
            print(f"Comparison with theoretical entropy:")
            print(f"  Theoretical: {theoretical_bits:.4f} bits")
            print(f"  Vine estimate: {vine_entropy_bits:.4f} bits")
            print(f"  Absolute error: {error_abs:.4f} bits")
            print(f"  Relative error: {error_rel:.2f}%")
            
            if error_rel < 5:
                print("✓ Excellent entropy estimation")
            elif error_rel < 10:
                print("✓ Good entropy estimation") 
            elif error_rel < 20:
                print("⚠ Moderate entropy estimation")
            else:
                print("⚠ Poor entropy estimation - check vine fit quality")
        
        # Provide diagnostic information
        print(f"Entropy estimation diagnostics:")
        print(f"  Log-likelihood range: [{np.min(log_p):.3f}, {np.max(log_p):.3f}]")
        print(f"  Log-likelihood std: {np.std(log_p):.3f}")
        print(f"  Valid samples used: {len(log_p)}/{n_entropy_samples}")
        
        return vine_entropy_bits
    
    def create_correlation_comparison_plot(self):
        """
        Create side-by-side visualization comparing correlation matrices
        
        Purpose:
        --------
        Visual assessment of how well the vine copula preserves correlation structure
        by comparing three correlation matrices:
        1. True correlation matrix (ground truth)
        2. Empirical correlation matrix (from original data) 
        3. Vine correlation matrix (from generated samples)
        
        Plot Structure:
        --------------
        Three heatmaps arranged horizontally:
        - Left: True correlations (ground truth)
        - Center: Empirical correlations (from original data)
        - Right: Vine correlations (from generated samples)
        
        Color Scheme:
        ------------
        - Red-Blue diverging colormap ('RdBu_r')
        - Blue: Positive correlations (+1)
        - White: No correlation (0)
        - Red: Negative correlations (-1)
        - Same scale [-1, +1] for all three plots
        
        Interpretation:
        --------------
        Good vine fit indicated by:
        - Right plot closely matches center plot
        - Similar patterns and intensities
        - Preserved correlation signs and magnitudes
        
        Poor fit indicated by:
        - Significant differences between center and right
        - Missing correlation patterns
        - Incorrect correlation signs
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object containing the three correlation heatmaps
            
        Output Files:
        ------------
        Saves plots as:
        - correlation_comparison.png (high-resolution raster)
        - correlation_comparison.pdf (vector format for papers)
        
        Expected Use:
        ------------
        - Quick visual assessment of vine quality
        - Publication-ready figures
        - Diagnostic tool for model evaluation
        """
        print("Creating correlation matrix comparison plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # True correlation matrix (ground truth)
        im1 = axes[0].imshow(self.true_correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_title('True Correlation Matrix\n(Ground Truth Design)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Variable Index')
        axes[0].set_ylabel('Variable Index')
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Correlation Coefficient', rotation=270, labelpad=15)
        
        # Add correlation values as text annotations
        for i in range(self.dim):
            for j in range(self.dim):
                text = axes[0].text(j, i, f'{self.true_correlation_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        # Empirical correlation matrix (from original data)
        im2 = axes[1].imshow(self.empirical_correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_title('Empirical Correlation Matrix\n(Observed in Data)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Variable Index')
        axes[1].set_ylabel('Variable Index')
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('Correlation Coefficient', rotation=270, labelpad=15)
        
        # Add correlation values as text annotations
        for i in range(self.dim):
            for j in range(self.dim):
                text = axes[1].text(j, i, f'{self.empirical_correlation_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        # Vine samples correlation matrix (from generated samples)
        im3 = axes[2].imshow(self.vine_correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[2].set_title('Vine Samples Correlation Matrix\n(Reproduced by Vine)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Variable Index')
        axes[2].set_ylabel('Variable Index')
        cbar3 = plt.colorbar(im3, ax=axes[2])
        cbar3.set_label('Correlation Coefficient', rotation=270, labelpad=15)
        
        # Add correlation values as text annotations
        for i in range(self.dim):
            for j in range(self.dim):
                text = axes[2].text(j, i, f'{self.vine_correlation_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        # Add quality metrics as subtitle
        emp_error = np.mean(np.abs(self.empirical_correlation_matrix - self.true_correlation_matrix))
        vine_error = np.mean(np.abs(self.vine_correlation_matrix - self.empirical_correlation_matrix))
        
        fig.suptitle(f'Correlation Preservation Analysis\n'
                    f'Empirical Error: {emp_error:.4f} | Vine Reproduction Error: {vine_error:.4f}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plots in multiple formats
        plt.savefig(os.path.join(results_dir, 'correlation_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(results_dir, 'correlation_comparison.pdf'), bbox_inches='tight')
        
        print(f"✓ Correlation comparison plots saved")
        print(f"  Empirical vs True MAE: {emp_error:.4f}")
        print(f"  Vine vs Empirical MAE: {vine_error:.4f}")
        
        return fig
    
    def create_pairwise_plots(self):
        """
        Create pairwise scatter plots and marginal distributions comparison
        
        Purpose:
        --------
        Detailed comparison of bivariate relationships and marginal distributions
        between original data and vine-generated samples to assess:
        - Preservation of bivariate dependence patterns
        - Accuracy of marginal distributions  
        - Visual detection of sampling artifacts
        
        Plot Structure:
        --------------
        Grid layout (n_vars × n_vars) where n_vars = min(4, self.dim):
        - Diagonal elements: Overlaid histograms of marginal distributions
        - Off-diagonal elements: Scatter plots of bivariate relationships
        - Blue points/bars: Original data
        - Red points/bars: Vine-generated samples
        
        Reduced Dimensionality:
        ----------------------
        For computational efficiency and visual clarity:
        - Only plot first 4 variables if dim > 4
        - Maintains readability while showing key relationships
        - Representative sample of full dependency structure
        
        Interpretation Guide:
        --------------------
        Good vine fit indicated by:
        - Overlapping scatter point clouds (similar bivariate patterns)
        - Aligned histogram peaks (similar marginal means)
        - Similar spread patterns (preserved variances)
        - No obvious clustering or artifacts in red points
        
        Poor fit indicated by:
        - Clearly separated point clouds
        - Misaligned histogram peaks
        - Different scatter patterns or orientations
        - Artifacts like clustering or gaps in vine samples
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure containing the pairwise comparison plots
            
        Output Files:
        ------------
        Saves as:
        - pairwise_comparison.png (high-resolution)
        - pairwise_comparison.pdf (vector format)
        
        Sample Size Considerations:
        --------------------------
        - Plots subsample data if n_samples > 2000 for visibility
        - Alpha transparency used to handle overlapping points
        - Point size optimized for pattern visibility
        """
        print("Creating pairwise scatter plots and marginal comparisons...")
        
        # Select subset of variables for visualization
        n_vars_plot = min(4, self.dim)
        print(f"Plotting first {n_vars_plot} variables (out of {self.dim} total)")
        
        # Subsample for better visualization if datasets are large
        n_plot_samples = min(1000, self.n_samples)
        if n_plot_samples < self.n_samples:
            print(f"Subsampling {n_plot_samples} points for visualization clarity")
            plot_indices = np.random.choice(self.n_samples, n_plot_samples, replace=False)
            orig_plot = self.original_data[plot_indices, :]
            vine_plot = self.vine_samples[plot_indices, :]
        else:
            orig_plot = self.original_data
            vine_plot = self.vine_samples
        
        fig, axes = plt.subplots(n_vars_plot, n_vars_plot, figsize=(15, 15))
        
        # Ensure axes is always 2D for consistent indexing
        if n_vars_plot == 1:
            axes = np.array([[axes]])
        elif n_vars_plot == 2:
            axes = axes.reshape(2, 2)
        
        for i in range(n_vars_plot):
            for j in range(n_vars_plot):
                if i == j:
                    # Diagonal: histograms of marginal distributions
                    axes[i, j].hist(orig_plot[:, i], alpha=0.6, label='Original Data', 
                                  bins=30, density=True, color='blue', edgecolor='darkblue')
                    axes[i, j].hist(vine_plot[:, i], alpha=0.6, label='Vine Samples', 
                                  bins=30, density=True, color='red', edgecolor='darkred')
                    axes[i, j].set_title(f'Variable {i+1} - Marginal Distribution', fontweight='bold')
                    axes[i, j].set_ylabel('Density')
                    axes[i, j].grid(True, alpha=0.3)
                    
                    # Add statistics
                    orig_mean, orig_std = np.mean(orig_plot[:, i]), np.std(orig_plot[:, i])
                    vine_mean, vine_std = np.mean(vine_plot[:, i]), np.std(vine_plot[:, i])
                    
                    axes[i, j].text(0.05, 0.95, 
                                  f'Original: μ={orig_mean:.2f}, σ={orig_std:.2f}\n'
                                  f'Vine: μ={vine_mean:.2f}, σ={vine_std:.2f}',
                                  transform=axes[i, j].transAxes, fontsize=9,
                                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    if i == 0:
                        axes[i, j].legend(loc='upper right')
                        
                else:
                    # Off-diagonal: scatter plots of bivariate relationships
                    axes[i, j].scatter(orig_plot[:, j], orig_plot[:, i], 
                                     alpha=0.4, s=2, label='Original Data', color='blue')
                    axes[i, j].scatter(vine_plot[:, j], vine_plot[:, i], 
                                     alpha=0.4, s=2, label='Vine Samples', color='red')
                    
                    axes[i, j].set_xlabel(f'Variable {j+1}')
                    axes[i, j].set_ylabel(f'Variable {i+1}')
                    axes[i, j].set_title(f'Variables {j+1} vs {i+1}', fontweight='bold')
                    axes[i, j].grid(True, alpha=0.3)
                    
                    # Add correlation annotations
                    orig_corr = np.corrcoef(orig_plot[:, j], orig_plot[:, i])[0, 1]
                    vine_corr = np.corrcoef(vine_plot[:, j], vine_plot[:, i])[0, 1]
                    
                    axes[i, j].text(0.05, 0.95, 
                                  f'ρ_orig = {orig_corr:.3f}\nρ_vine = {vine_corr:.3f}',
                                  transform=axes[i, j].transAxes, fontsize=9,
                                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    if i == 0 and j == 1:
                        axes[i, j].legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save plots
        plt.savefig(os.path.join(results_dir, 'pairwise_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(results_dir, 'pairwise_comparison.pdf'), bbox_inches='tight')
        
        print(f"✓ Pairwise comparison plots saved")
        print(f"  Showing {n_vars_plot}×{n_vars_plot} variable relationships")
        print(f"  {n_plot_samples} points plotted per dataset")
        
        return fig
    
    def create_correlation_error_plot(self):
        """Create plot showing correlation estimation errors"""
        # Calculate errors
        empirical_error = np.abs(self.empirical_correlation_matrix - self.true_correlation_matrix)
        vine_error = np.abs(self.vine_correlation_matrix - self.true_correlation_matrix)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Empirical error
        im1 = axes[0].imshow(empirical_error, cmap='Reds', vmin=0)
        axes[0].set_title('Empirical Correlation Error\n|Empirical - True|')
        axes[0].set_xlabel('Variable')
        axes[0].set_ylabel('Variable')
        plt.colorbar(im1, ax=axes[0])
        
        # Vine error
        im2 = axes[1].imshow(vine_error, cmap='Reds', vmin=0)
        axes[1].set_title('Vine Correlation Error\n|Vine - True|')
        axes[1].set_xlabel('Variable')
        axes[1].set_ylabel('Variable')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(results_dir, 'correlation_errors.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(results_dir, 'correlation_errors.pdf'), bbox_inches='tight')
        
        return fig
    
    def save_results(self):
        """Save all results to files"""
        # Compile results
        self.results = {
            'parameters': {
                'dim': self.dim,
                'n_samples': self.n_samples,
                'vine_type': self.vine_type,
                'timestamp': datetime.now().isoformat()
            },
            'correlation_matrices': {
                'true': self.true_correlation_matrix.tolist(),
                'empirical': self.empirical_correlation_matrix.tolist(),
                'vine': self.vine_correlation_matrix.tolist()
            },
            'correlation_errors': {
                'empirical_mae': np.mean(np.abs(self.empirical_correlation_matrix - self.true_correlation_matrix)),
                'vine_mae': np.mean(np.abs(self.vine_correlation_matrix - self.true_correlation_matrix)),
                'empirical_rmse': np.sqrt(np.mean((self.empirical_correlation_matrix - self.true_correlation_matrix)**2)),
                'vine_rmse': np.sqrt(np.mean((self.vine_correlation_matrix - self.true_correlation_matrix)**2))
            },
            'entropy': {
                'theoretical': float(self.theoretical_entropy),
                'vine_estimate': float(self.vine_entropy_estimate),
                'entropy_error': float(abs(self.vine_entropy_estimate - self.theoretical_entropy))
            },
            'r_matrix': self.r_matrix.tolist()
        }
        
        # Save results as JSON
        import json
        with open(os.path.join(results_dir, 'analysis_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as pickle for later use
        with open(os.path.join(results_dir, 'analysis_results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save data arrays
        np.savez(os.path.join(results_dir, 'data_arrays.npz'),
                 original_data=self.original_data,
                 vine_samples=self.vine_samples,
                 true_correlation=self.true_correlation_matrix,
                 empirical_correlation=self.empirical_correlation_matrix,
                 vine_correlation=self.vine_correlation_matrix)
        
        print(f"Results saved in: {results_dir}")
        
    def print_summary(self):
        """Print summary of results"""
        print("\n" + "="*60)
        print("MULTIVARIATE GAUSSIAN VINE COPULA ANALYSIS SUMMARY")
        print("="*60)
        print(f"Dimensions: {self.dim}")
        print(f"Samples: {self.n_samples}")
        print(f"Vine Type: {self.vine_type}")
        
        print("\nCORRELATION ESTIMATION ERRORS:")
        print(f"Empirical MAE: {self.results['correlation_errors']['empirical_mae']:.4f}")
        print(f"Vine MAE: {self.results['correlation_errors']['vine_mae']:.4f}")
        print(f"Empirical RMSE: {self.results['correlation_errors']['empirical_rmse']:.4f}")
        print(f"Vine RMSE: {self.results['correlation_errors']['vine_rmse']:.4f}")
        
        print("\nENTROPY COMPARISON:")
        print(f"Theoretical Entropy: {self.theoretical_entropy:.4f} bits")
        print(f"Vine Estimated Entropy: {self.vine_entropy_estimate:.4f} bits")
        print(f"Entropy Error: {abs(self.vine_entropy_estimate - self.theoretical_entropy):.4f} bits")
        
        print(f"\nResults saved in: {results_dir}")
        print("="*60)
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting full multivariate Gaussian vine copula analysis...")
        
        # 1. Generate data
        self.simulate_multivariate_gaussian()
        
        # 2. Setup vine structure
        self.setup_vine_structure()
        
        # 3. Fit vine copula
        self.fit_vine_copula()
        
        # 4. Generate vine samples
        self.generate_vine_samples()
        
        # 5. Calculate entropies
        self.calculate_theoretical_entropy()
        self.estimate_vine_entropy()
        
        # 6. Create plots
        print("Creating visualization plots...")
        self.create_correlation_comparison_plot()
        self.create_pairwise_plots()
        self.create_correlation_error_plot()
        
        # 7. Save results
        self.save_results()
        
        # 8. Print summary
        self.print_summary()
        
        return self.results


def main():
    """
    Main function to run the complete multivariate Gaussian vine copula analysis
    
    Analysis Pipeline:
    -----------------
    1. Data Generation: Create multivariate Gaussian data with known correlations
    2. Vine Setup: Configure vine copula structure (C-vine, D-vine, or R-vine)
    3. Model Fitting: Fit vine copula to capture dependence structure
    4. Sample Generation: Generate new samples from fitted vine model
    5. Correlation Analysis: Compare correlation preservation
    6. Entropy Estimation: Compare theoretical vs vine-estimated entropy
    7. Visualization: Create comprehensive plots and comparisons
    8. Results Export: Save all data, plots, and metrics
    
    Expected Runtime:
    ----------------
    - 4D, 1500 samples: ~2-5 minutes
    - 6D, 2500 samples: ~5-15 minutes
    - 8D, 3000 samples: ~15-45 minutes
    
    Success Indicators:
    ------------------
    - Correlation MAE < 0.1 (good preservation)
    - Entropy error < 10% (accurate information estimation)
    - Visual plots show good overlap between original and vine samples
    - No obvious artifacts or systematic biases
    
    Configuration Notes:
    -------------------
    Adjust these parameters based on your system capabilities:
    - dimensions: 3-8 (higher = more complex but potentially more interesting)
    - n_samples: 1000-4000 (more = better accuracy but slower)
    - vine_type: 'c-vine' most reliable, 'r-vine' most flexible
    """
    # Enhanced configuration - supports larger dimensions
    dimensions = 6           # Increased from 4 for more complex analysis  
    n_samples = 2500        # Increased from 1500 for better accuracy
    vine_type = 'c-vine'    # Options: 'r-vine', 'c-vine', 'd-vine' (c-vine is most reliable)
    
    print(f"Starting enhanced analysis with {dimensions}D data, {n_samples} samples, {vine_type}")
    print("="*70)
    print("MULTIVARIATE GAUSSIAN VINE COPULA ANALYSIS (ENHANCED)")
    print("="*70)
    print("This script will:")
    print("1. Generate synthetic multivariate Gaussian data")
    print("2. Fit a vine copula model to the data")
    print("3. Generate new samples from the fitted model")
    print("4. Compare correlation preservation and entropy estimation")
    print("5. Create visualizations and save results")
    print()
    print("Enhanced features:")
    print("- Supports higher dimensional analysis (up to 8+ variables)")
    print("- Memory-efficient processing for larger datasets")
    print("- Comprehensive correlation structure analysis")
    print("- Detailed performance metrics and diagnostics")
    print("="*70)
    
    # Create analyzer
    analyzer = Multivariate_Gaussian_Vine_Analysis(
        dim=dimensions, 
        n_samples=n_samples, 
        vine_type=vine_type
    )
    
    # Run analysis
    try:
        results = analyzer.run_full_analysis()
        
        print("\n" + "="*70)
        print("✅ ENHANCED ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("Check the results/ directory for:")
        print("- correlation_comparison.png: Side-by-side correlation matrices")
        print("- pairwise_comparison.png: Scatter plots and marginal distributions")
        print("- correlation_errors.png: Error heat maps")
        print("- analysis_results.json: Complete numerical results")
        print("- data_arrays.npz: All data arrays for further analysis")
        print()
        print("Performance Summary:")
        if 'correlation_errors' in results:
            vine_mae = results['correlation_errors']['vine_mae']
            entropy_error = results['entropy']['entropy_error']
            print(f"- Correlation preservation MAE: {vine_mae:.4f}")
            print(f"- Entropy estimation error: {entropy_error:.4f} bits")
            
            if vine_mae < 0.05:
                print("- ✅ Excellent correlation preservation!")
            elif vine_mae < 0.1:
                print("- ✅ Good correlation preservation")
            else:
                print("- ⚠️ Moderate correlation preservation")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("Try running with reduced parameters:")
        print("- Reduce dimensions to 4-5")
        print("- Reduce n_samples to 1500-2000")
        print("- Use 'c-vine' instead of 'r-vine'")
        print("- Check available memory and computational resources")
        print("\nFor R-vine optimization comparison, try:")
        print("python enhanced_vine_comparison.py")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 