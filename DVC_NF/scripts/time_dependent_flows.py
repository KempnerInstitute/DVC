#!/usr/bin/env python3
"""
Time-Dependent Vine Copula with Normalizing Flows

This module implements time-dependent vine copulas where the bandwidth parameters
of local likelihood estimates are governed by normalizing flows, allowing the
interaction structure to evolve over time.

Key Components:
1. TimeBandwidthFlow - Neural network that maps time -> bandwidth
2. Time-dependent local likelihood computation
3. Integration with existing R-vine optimization
4. Training procedures for flow parameters

Author: DVC Analysis Team
Date: 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Add DVC_tensorflow to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
dvc_tensorflow_dir = os.path.join(project_root, 'src', 'DVC_tensorflow')
sys.path.append(dvc_tensorflow_dir)

from classes.objects import *
from vine_tree.tree_op import *

class TimeBandwidthFlow(tf.keras.Model):
    """
    Normalizing Flow for Time-Dependent Bandwidth Parameters
    
    This neural network maps time indices to bandwidth parameters for 
    local likelihood estimation in vine copulas.
    
    Architecture:
    - Input: time index t (normalized)
    - Output: bandwidth parameter b(t) > 0
    - Constraint: b(t) must be positive for valid kernel estimation
    """
    
    def __init__(self, hidden_dim=64, name=None):
        super(TimeBandwidthFlow, self).__init__(name=name)
        
        self.hidden_dim = hidden_dim
        
        # Neural network layers
        self.fc1 = tf.keras.layers.Dense(
            hidden_dim, 
            activation='relu',
            kernel_initializer='glorot_uniform',
            name='fc1'
        )
        self.fc2 = tf.keras.layers.Dense(
            hidden_dim, 
            activation='relu',
            kernel_initializer='glorot_uniform',
            name='fc2'
        )
        self.fc3 = tf.keras.layers.Dense(
            hidden_dim // 2, 
            activation='relu',
            kernel_initializer='glorot_uniform',
            name='fc3'
        )
        self.fc_out = tf.keras.layers.Dense(
            1, 
            kernel_initializer='glorot_uniform',
            name='output'
        )
        
        # Batch normalization for stability
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.bn2 = tf.keras.layers.BatchNormalization(name='bn2')
        
        # Dropout for regularization
        self.dropout = tf.keras.layers.Dropout(0.1, name='dropout')
        
    def call(self, t, training=None):
        """
        Forward pass: time -> bandwidth
        
        Parameters:
        -----------
        t : tf.Tensor
            Time indices, shape (batch_size, 1) or (batch_size,)
            Should be normalized to [0, 1] range
        training : bool
            Whether in training mode (for dropout/batch_norm)
            
        Returns:
        --------
        bandwidth : tf.Tensor
            Positive bandwidth values, shape (batch_size, 1)
        """
        
        # Ensure proper shape
        if len(t.shape) == 1:
            t = tf.expand_dims(t, -1)
            
        # Neural network forward pass
        x = self.fc1(t)
        x = self.bn1(x, training=training)
        x = self.dropout(x, training=training)
        
        x = self.fc2(x)
        x = self.bn2(x, training=training)
        x = self.dropout(x, training=training)
        
        x = self.fc3(x)
        x = self.fc_out(x)
        
        # Ensure positivity using softplus
        # Add small constant for numerical stability
        bandwidth = tf.nn.softplus(x) + 1e-4
        
        return bandwidth
    
    def get_config(self):
        return {
            'hidden_dim': self.hidden_dim,
            'name': self.name
        }


class TimeDependentVineCopula:
    """
    Time-Dependent Vine Copula with Flow-Based Bandwidth Modelling
    
    This class extends the existing vine copula framework to handle
    time-dependent interaction structures by using normalizing flows
    to model time-varying bandwidth parameters.
    """
    
    def __init__(self, 
                 dim,
                 vine_type='c-vine',
                 optimization_method='tau',
                 n_time_steps=100,
                 device='cpu'):
        """
        Initialize time-dependent vine copula
        
        Parameters:
        -----------
        dim : int
            Data dimensionality
        vine_type : str
            Type of vine ('c-vine', 'd-vine', 'r-vine')
        optimization_method : str
            R-vine optimization method ('tau', 'entropy', 'random')
        n_time_steps : int
            Number of time steps in the dataset
        device : str
            Device for computation ('cpu' or 'gpu')
        """
        
        self.dim = dim
        self.vine_type = vine_type
        self.optimization_method = optimization_method
        self.n_time_steps = n_time_steps
        self.device = device
        
        # Initialize vine structure
        self.vine_structure = None
        self.r_matrix = None
        self.ind_vine = None
        
        # Flow models for each edge
        self.flow_models = {}
        self.edge_list = []
        
        # Training history
        self.training_history = {
            'loss': [],
            'epochs': [],
            'bandwidth_evolution': {}
        }
        
        # Results storage
        self.results_dir = os.path.join(current_dir, '..', 'results', 'time_dependent_vines')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def initialize_vine_structure(self, data=None):
        """
        Initialize vine structure using existing optimization methods
        
        Parameters:
        -----------
        data : np.ndarray, optional
            Data for structure optimization, shape (n_samples, dim)
            If None, uses random structure
        """
        
        print(f"Initializing {self.vine_type} structure...")
        
        if self.vine_type == 'c-vine':
            # Canonical vine: star structure
            self.r_matrix = np.tril(np.tile(np.array(range(self.dim, 0, -1)), (self.dim, 1)).T)
            self.ind_vine = self._build_c_vine_edges()
            
        elif self.vine_type == 'd-vine':
            # D-vine: path structure  
            self.r_matrix = self._build_d_vine_matrix()
            self.ind_vine = self._build_d_vine_edges()
            
        elif self.vine_type == 'r-vine':
            if data is not None and self.optimization_method in ['tau', 'entropy']:
                # Use optimal structure based on data
                self.ind_vine = []
                for tr in range(self.dim - 1):
                    if tr == 0:
                        edges, weights = optimal_tree(
                            data.T, None, self.ind_vine, tr, 
                            rand=(self.optimization_method == 'random')
                        )
                    else:
                        # For higher trees, use conditional data (simplified)
                        edges, weights = optimal_tree(
                            data.T, data.T, self.ind_vine, tr, 
                            rand=(self.optimization_method == 'random')
                        )
                    self.ind_vine.append(edges)
                
                # Build R-matrix from edge structure
                self.r_matrix, _, _ = prepare_optimal(self.dim, self.ind_vine)
            else:
                # Random R-vine structure
                self.r_matrix, self.ind_vine, _, _ = random_r_matrix_gen(self.dim)
        
        # Extract edge list for flow initialization
        self._extract_edge_list()
        
        print(f"Vine structure initialized with {len(self.edge_list)} edges")
        print(f"R-matrix shape: {self.r_matrix.shape}")
        
    def _build_c_vine_edges(self):
        """Build edge list for canonical vine"""
        ind_vine = []
        for tr in range(self.dim - 1):
            edges = []
            for j in range(self.dim - 1 - tr):
                if tr == 0:
                    edges.append([0, j + 1])  # Star structure centered at 0
                else:
                    edges.append([j, j + 1])  # Higher tree edges
            ind_vine.append(edges)
        return ind_vine
    
    def _build_d_vine_matrix(self):
        """Build R-matrix for D-vine"""
        r_matrix = np.zeros((self.dim, self.dim), dtype=int)
        for i in range(self.dim):
            r_matrix[i, i] = i + 1
        for i in range(self.dim - 1):
            for j in range(i + 1, self.dim):
                r_matrix[i, j] = j + 1
        return r_matrix
    
    def _build_d_vine_edges(self):
        """Build edge list for D-vine"""
        ind_vine = []
        for tr in range(self.dim - 1):
            edges = []
            for j in range(self.dim - 1 - tr):
                edges.append([j, j + 1])  # Path structure
            ind_vine.append(edges)
        return ind_vine
    
    def _extract_edge_list(self):
        """Extract complete edge list from vine structure"""
        self.edge_list = []
        for tr, tree_edges in enumerate(self.ind_vine):
            for edge_idx, edge in enumerate(tree_edges):
                edge_id = f"tree_{tr}_edge_{edge_idx}_{edge[0]}_{edge[1]}"
                self.edge_list.append({
                    'id': edge_id,
                    'tree': tr,
                    'edge_idx': edge_idx,
                    'nodes': tuple(edge)
                })
        
        print(f"Extracted {len(self.edge_list)} edges:")
        for edge in self.edge_list[:5]:  # Show first 5
            print(f"  {edge['id']}: nodes {edge['nodes']}")
        if len(self.edge_list) > 5:
            print(f"  ... and {len(self.edge_list) - 5} more")
    
    def initialize_flows(self, hidden_dim=64):
        """
        Initialize normalizing flows for each edge
        
        Parameters:
        -----------
        hidden_dim : int
            Hidden dimension for flow networks
        """
        
        print(f"Initializing flows with hidden_dim={hidden_dim}...")
        
        for edge in self.edge_list:
            edge_id = edge['id']
            
            # Create flow for this edge
            flow = TimeBandwidthFlow(
                hidden_dim=hidden_dim,
                name=f"flow_{edge_id}"
            )
            
            # Initialize with dummy input to build the model
            dummy_time = tf.random.normal((1, 1))
            _ = flow(dummy_time)
            
            self.flow_models[edge_id] = flow
            
            print(f"  Initialized flow for {edge_id}")
        
        print(f"Initialized {len(self.flow_models)} flow models")
    
    def compute_time_dependent_bandwidth(self, time_indices, edge_id):
        """
        Compute bandwidth for specific edge at given time indices
        
        Parameters:
        -----------
        time_indices : np.ndarray or tf.Tensor
            Time indices, normalized to [0, 1]
        edge_id : str
            Edge identifier
            
        Returns:
        --------
        bandwidths : tf.Tensor
            Bandwidth values at each time point
        """
        
        # Normalize time indices to [0, 1]
        if isinstance(time_indices, np.ndarray):
            time_indices = tf.constant(time_indices, dtype=tf.float32)
        
        # Ensure proper shape
        if len(time_indices.shape) == 1:
            time_indices = tf.expand_dims(time_indices, -1)
        
        # Get flow for this edge
        flow = self.flow_models[edge_id]
        
        # Compute bandwidths
        bandwidths = flow(time_indices)
        
        return bandwidths
    
    def compute_time_dependent_local_likelihood(self, 
                                               data_x, 
                                               data_t, 
                                               edge_data,
                                               edge_id):
        """
        Compute local likelihood for a specific edge with time-dependent bandwidth
        
        Parameters:
        -----------
        data_x : tf.Tensor
            Data in copula space, shape (n_time_steps, n_samples, 2)
        data_t : tf.Tensor
            Time indices, shape (n_time_steps,)
        edge_data : tf.Tensor
            Data for specific edge, shape (n_time_steps, n_samples, 2)
        edge_id : str
            Edge identifier
            
        Returns:
        --------
        log_likelihood : tf.Tensor
            Log-likelihood value for this edge across all time steps
        """
        
        # Get time-dependent bandwidths
        bandwidths = self.compute_time_dependent_bandwidth(data_t, edge_id)
        
        total_log_lik = 0.0
        n_time_steps = tf.shape(data_t)[0]
        
        # Process each time step
        for t in range(n_time_steps):
            # Get bandwidth for this time step
            b_t = bandwidths[t, 0]  # Shape: scalar
            
            # Get data for this time step
            x_t = edge_data[t]  # Shape: (n_samples, 2)
            
            # Compute local likelihood (simplified kernel density)
            log_lik_t = self._compute_kernel_log_likelihood(x_t, b_t)
            
            total_log_lik += log_lik_t
        
        return total_log_lik
    
    def _compute_kernel_log_likelihood(self, data, bandwidth):
        """
        Compute kernel-based log likelihood for bivariate data
        
        This is a simplified version of the local likelihood computation
        from the existing DVC framework.
        
        Parameters:
        -----------
        data : tf.Tensor
            Bivariate data, shape (n_samples, 2)
        bandwidth : tf.Tensor
            Bandwidth parameter, scalar
            
        Returns:
        --------
        log_likelihood : tf.Tensor
            Log-likelihood value
        """
        
        n_samples = tf.shape(data)[0]
        n_samples_f = tf.cast(n_samples, tf.float32)
        
        # Compute pairwise distances
        # data: (n_samples, 2)
        # distances: (n_samples, n_samples)
        distances = tf.norm(
            tf.expand_dims(data, 1) - tf.expand_dims(data, 0), 
            axis=2
        )
        
        # Gaussian kernel
        # K(u) = 1/(2π*h²) * exp(-||u||²/(2h²))
        normalizer = 1.0 / (2.0 * np.pi * bandwidth**2)
        kernel_values = normalizer * tf.exp(
            -distances**2 / (2.0 * bandwidth**2)
        )
        
        # Leave-one-out density estimation
        # Zero out diagonal (self-interactions)
        mask = 1.0 - tf.eye(n_samples)
        kernel_values_loo = kernel_values * mask
        
        # Density estimates for each point
        density_estimates = tf.reduce_sum(kernel_values_loo, axis=1) / (n_samples_f - 1.0)
        
        # Add small constant for numerical stability
        density_estimates = tf.maximum(density_estimates, 1e-10)
        
        # Log-likelihood
        log_likelihood = tf.reduce_sum(tf.math.log(density_estimates))
        
        return log_likelihood
    
    def fit(self, 
            data_time_series, 
            time_indices,
            learning_rate=1e-3,
            num_epochs=1000,
            batch_size=None,
            patience=50,
            min_delta=1e-6):
        """
        Fit time-dependent vine copula model
        
        Parameters:
        -----------
        data_time_series : np.ndarray
            Time series data, shape (n_time_steps, n_samples, dim)
        time_indices : np.ndarray
            Time indices, shape (n_time_steps,)
        learning_rate : float
            Learning rate for optimizer
        num_epochs : int
            Maximum number of training epochs
        batch_size : int, optional
            Batch size for training (if None, use full batch)
        patience : int
            Early stopping patience
        min_delta : float
            Minimum change in loss for early stopping
        """
        
        print("Starting time-dependent vine copula training...")
        print(f"Data shape: {data_time_series.shape}")
        print(f"Time steps: {len(time_indices)}")
        print(f"Learning rate: {learning_rate}")
        print(f"Max epochs: {num_epochs}")
        
        # Convert to tensors
        data_tf = tf.constant(data_time_series, dtype=tf.float32)
        time_tf = tf.constant(time_indices, dtype=tf.float32)
        
        # Normalize time indices to [0, 1]
        time_normalized = (time_tf - tf.reduce_min(time_tf)) / (
            tf.reduce_max(time_tf) - tf.reduce_min(time_tf)
        )
        
        # Initialize optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Get all trainable variables
        trainable_vars = []
        for flow in self.flow_models.values():
            trainable_vars.extend(flow.trainable_variables)
        
        print(f"Total trainable parameters: {sum([tf.size(v) for v in trainable_vars])}")
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                total_loss = 0.0
                
                # Compute loss for each edge
                for edge in self.edge_list:
                    edge_id = edge['id']
                    tree_idx = edge['tree']
                    edge_idx = edge['edge_idx']
                    nodes = edge['nodes']
                    
                    # Extract edge data from full dataset
                    # For simplicity, use first tree data (marginal pairs)
                    if tree_idx == 0:
                        edge_data = tf.stack([
                            data_tf[:, :, nodes[0]],
                            data_tf[:, :, nodes[1]]
                        ], axis=2)
                        
                        # Compute log-likelihood for this edge
                        edge_log_lik = self.compute_time_dependent_local_likelihood(
                            data_tf, time_normalized, edge_data, edge_id
                        )
                        
                        total_loss -= edge_log_lik  # Negative log-likelihood
                
                # Add regularization
                l2_regularization = 0.0
                for flow in self.flow_models.values():
                    for var in flow.trainable_variables:
                        l2_regularization += tf.nn.l2_loss(var)
                
                total_loss += 1e-5 * l2_regularization
            
            # Compute gradients and update
            gradients = tape.gradient(total_loss, trainable_vars)
            
            # Clip gradients for stability
            gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]
            
            optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            # Record training history
            current_loss = total_loss.numpy()
            self.training_history['loss'].append(current_loss)
            self.training_history['epochs'].append(epoch)
            
            # Early stopping check
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                patience_counter = 0
                # Save best models
                self._save_checkpoint(epoch, current_loss)
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 50 == 0 or epoch < 10:
                print(f"Epoch {epoch:4d}: Loss = {current_loss:.6f}")
                
                # Show bandwidth evolution for first edge
                if len(self.edge_list) > 0:
                    first_edge_id = self.edge_list[0]['id']
                    sample_times = tf.linspace(0.0, 1.0, 5)
                    sample_bandwidths = self.compute_time_dependent_bandwidth(
                        sample_times, first_edge_id
                    )
                    print(f"  Sample bandwidths for {first_edge_id}: {sample_bandwidths.numpy().flatten()}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break
        
        print("Training completed!")
        print(f"Final loss: {current_loss:.6f}")
        print(f"Best loss: {best_loss:.6f}")
        
        # Save final results
        self._save_final_results()
    
    def _save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for edge_id, flow in self.flow_models.items():
            checkpoint_path = os.path.join(checkpoint_dir, f'{edge_id}_best')
            flow.save_weights(checkpoint_path)
    
    def _save_final_results(self):
        """Save final training results and visualizations"""
        
        # Save training history
        history_path = os.path.join(self.results_dir, 'training_history.npz')
        np.savez(history_path,
                loss=np.array(self.training_history['loss']),
                epochs=np.array(self.training_history['epochs']))
        
        # Create visualizations
        self._create_training_plots()
        self._create_bandwidth_evolution_plots()
        
        print(f"Results saved to {self.results_dir}")
    
    def _create_training_plots(self):
        """Create training loss plots"""
        
        plt.figure(figsize=(12, 5))
        
        # Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['epochs'], self.training_history['loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Negative Log-Likelihood)')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        
        # Loss curve (log scale)
        plt.subplot(1, 2, 2)
        plt.semilogy(self.training_history['epochs'], self.training_history['loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.title('Training Loss (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_bandwidth_evolution_plots(self):
        """Create bandwidth evolution visualization"""
        
        # Generate time grid for visualization
        time_grid = np.linspace(0, 1, 100)
        time_tf = tf.constant(time_grid.reshape(-1, 1), dtype=tf.float32)
        
        # Plot bandwidth evolution for each edge
        n_edges = min(6, len(self.edge_list))  # Plot up to 6 edges
        
        plt.figure(figsize=(15, 10))
        
        for i, edge in enumerate(self.edge_list[:n_edges]):
            edge_id = edge['id']
            
            # Compute bandwidth evolution
            bandwidths = self.compute_time_dependent_bandwidth(time_tf, edge_id)
            
            plt.subplot(2, 3, i + 1)
            plt.plot(time_grid, bandwidths.numpy().flatten(), linewidth=2)
            plt.xlabel('Time (normalized)')
            plt.ylabel('Bandwidth')
            plt.title(f'{edge_id}\nNodes: {edge["nodes"]}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'bandwidth_evolution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_time_dependent_fit(self, test_data, test_times):
        """
        Evaluate model on test data
        
        Parameters:
        -----------
        test_data : np.ndarray
            Test data, shape (n_test_times, n_samples, dim)
        test_times : np.ndarray
            Test time indices
            
        Returns:
        --------
        test_loss : float
            Test loss (negative log-likelihood)
        """
        
        # Convert to tensors
        test_data_tf = tf.constant(test_data, dtype=tf.float32)
        test_times_tf = tf.constant(test_times, dtype=tf.float32)
        
        # Normalize time indices
        test_times_normalized = (test_times_tf - tf.reduce_min(test_times_tf)) / (
            tf.reduce_max(test_times_tf) - tf.reduce_min(test_times_tf)
        )
        
        total_loss = 0.0
        
        # Compute loss for each edge
        for edge in self.edge_list:
            edge_id = edge['id']
            tree_idx = edge['tree']
            nodes = edge['nodes']
            
            # Extract edge data (first tree only for simplicity)
            if tree_idx == 0:
                edge_data = tf.stack([
                    test_data_tf[:, :, nodes[0]],
                    test_data_tf[:, :, nodes[1]]
                ], axis=2)
                
                # Compute log-likelihood
                edge_log_lik = self.compute_time_dependent_local_likelihood(
                    test_data_tf, test_times_normalized, edge_data, edge_id
                )
                
                total_loss -= edge_log_lik
        
        return total_loss.numpy()
    
    def predict_bandwidth_evolution(self, time_range=None, n_points=100):
        """
        Predict bandwidth evolution over time
        
        Parameters:
        -----------
        time_range : tuple, optional
            (start_time, end_time) for prediction
        n_points : int
            Number of time points for prediction
            
        Returns:
        --------
        predictions : dict
            Dictionary with edge_id -> (times, bandwidths)
        """
        
        if time_range is None:
            time_range = (0, 1)
        
        # Generate time grid
        times = np.linspace(time_range[0], time_range[1], n_points)
        times_tf = tf.constant(times.reshape(-1, 1), dtype=tf.float32)
        
        predictions = {}
        
        for edge in self.edge_list:
            edge_id = edge['id']
            
            # Predict bandwidths
            bandwidths = self.compute_time_dependent_bandwidth(times_tf, edge_id)
            
            predictions[edge_id] = {
                'times': times,
                'bandwidths': bandwidths.numpy().flatten(),
                'nodes': edge['nodes'],
                'tree': edge['tree']
            }
        
        return predictions


def main():
    """
    Example usage of time-dependent vine copula
    """
    
    print("Time-Dependent Vine Copula Implementation")
    print("=" * 50)
    
    # Test with small example
    dim = 3
    n_time_steps = 50
    n_samples_per_time = 100
    
    # Initialize model
    model = TimeDependentVineCopula(
        dim=dim,
        vine_type='c-vine',
        optimization_method='tau',
        n_time_steps=n_time_steps
    )
    
    # Initialize structure and flows
    model.initialize_vine_structure()
    model.initialize_flows(hidden_dim=32)
    
    # Generate dummy time series data
    np.random.seed(42)
    time_indices = np.arange(n_time_steps, dtype=np.float32)
    
    # Create synthetic data with time-dependent correlations
    data_time_series = np.zeros((n_time_steps, n_samples_per_time, dim))
    
    for t in range(n_time_steps):
        # Time-dependent correlation
        correlation = 0.3 + 0.5 * np.sin(2 * np.pi * t / n_time_steps)
        
        # Generate multivariate normal data
        mean = np.zeros(dim)
        cov = np.eye(dim)
        cov[0, 1] = cov[1, 0] = correlation
        
        data_time_series[t] = np.random.multivariate_normal(mean, cov, n_samples_per_time)
    
    print(f"Generated synthetic data: {data_time_series.shape}")
    
    # Fit model
    model.fit(
        data_time_series,
        time_indices,
        learning_rate=1e-3,
        num_epochs=100,  # Reduced for example
        patience=20
    )
    
    # Make predictions
    predictions = model.predict_bandwidth_evolution()
    
    print("\nBandwidth evolution predictions:")
    for edge_id, pred in predictions.items():
        bw_range = (pred['bandwidths'].min(), pred['bandwidths'].max())
        print(f"{edge_id}: bandwidth range {bw_range[0]:.4f} - {bw_range[1]:.4f}")
    
    print(f"\nResults saved to: {model.results_dir}")


if __name__ == "__main__":
    main() 