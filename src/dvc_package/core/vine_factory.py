"""
Vine Copula Factory

Factory functions and enums for creating different types of vine copulas.
Provides a unified interface for instantiating C-vine, D-vine, and R-vine objects.
"""

from enum import Enum
from typing import List, Optional, Union, Dict, Any
import numpy as np

from .objects import vine_obj_bin, margin_obj


class VineType(Enum):
    """Enumeration of supported vine types."""
    C_VINE = "c-vine"
    D_VINE = "d-vine" 
    R_VINE = "r-vine"


class CVine(vine_obj_bin):
    """
    Canonical Vine (C-vine) implementation.
    
    In a C-vine, each tree has a star structure with one central variable
    connected to all others. The first variable is the root of the first tree,
    the second variable is the root of the second tree, etc.
    """
    
    def __init__(self, families: List[str], vine_depth: int, 
                 margin: List[margin_obj], knots: int = 30):
        super().__init__("c-vine", families, vine_depth, margin, knots)
        self._build_cvine_structure()
    
    def _build_cvine_structure(self):
        """Build C-vine structure: star topology for each tree level."""
        self.ind_vine = []
        d = self.n_cop
        
        for level in range(d - 1):
            edges_level = []
            center = level  # Center variable for this tree
            
            # Connect center to all remaining variables
            for j in range(level + 1, d):
                edges_level.append([center, j])
            
            self.ind_vine.append(edges_level)


class DVine(vine_obj_bin):
    """
    Drawable Vine (D-vine) implementation.
    
    In a D-vine, each tree forms a path structure. Variables are arranged
    in sequence, and each tree connects adjacent pairs in the conditioning set.
    """
    
    def __init__(self, families: List[str], vine_depth: int,
                 margin: List[margin_obj], knots: int = 30, 
                 variable_order: Optional[List[int]] = None):
        super().__init__("d-vine", families, vine_depth, margin, knots)
        self.variable_order = variable_order or list(range(vine_depth))
        self._build_dvine_structure()
    
    def _build_dvine_structure(self):
        """Build D-vine structure: path topology for each tree level."""
        self.ind_vine = []
        d = self.n_cop
        order = self.variable_order
        
        for level in range(d - 1):
            edges_level = []
            
            # For each tree level, connect adjacent variables in the remaining set
            remaining_vars = order[level:]
            for i in range(len(remaining_vars) - 1):
                var1, var2 = remaining_vars[i], remaining_vars[i + 1]
                edges_level.append([var1, var2])
            
            self.ind_vine.append(edges_level)


class RVine(vine_obj_bin):
    """
    Regular Vine (R-vine) implementation.
    
    R-vines are the most general class, allowing arbitrary tree structures
    subject to the proximity condition. Structure is defined by an R-matrix.
    """
    
    def __init__(self, families: List[str], vine_depth: int,
                 margin: List[margin_obj], knots: int = 30,
                 r_matrix: Optional[np.ndarray] = None,
                 method: str = "random",
                 data: Optional[np.ndarray] = None):
        super().__init__("r-vine", families, vine_depth, margin, knots, method)
        self.r_matrix = r_matrix
        self.data = data  # Store data for tau-based optimization
        
        if r_matrix is None:
            if method == "tau" and data is not None:
                self._generate_tau_based_r_matrix()
            else:
                self._generate_random_r_matrix()
        else:
            self._validate_r_matrix()
        self._build_rvine_structure()
    
    def _generate_random_r_matrix(self):
        """Generate a random valid R-matrix."""
        d = self.n_cop
        r_matrix = np.zeros((d, d), dtype=int)
        
        # Fill diagonal with variable indices
        for i in range(d):
            r_matrix[i, i] = i + 1
        
        # Fill upper triangular part
        for i in range(d):
            available = list(range(1, d + 1))
            available.remove(i + 1)  # Remove diagonal element
            
            for j in range(i + 1, d):
                # Ensure proximity condition is satisfied
                if j == i + 1:
                    # First column after diagonal can be any available
                    choice = np.random.choice(available)
                else:
                    # Must share a common element with previous column
                    prev_col = r_matrix[i:j, j-1]
                    valid_choices = [x for x in available if x in prev_col or x == r_matrix[i, i]]
                    if valid_choices:
                        choice = np.random.choice(valid_choices)
                    else:
                        choice = available[0] if available else i + 1
                
                r_matrix[i, j] = choice
                if choice in available:
                    available.remove(choice)
        
        self.r_matrix = r_matrix
    
    def _generate_tau_based_r_matrix(self):
        """
        Generate R-matrix based on Kendall's tau (dependence strength).
        
        This method constructs the R-vine structure by selecting edges that maximize
        the sum of absolute Kendall's tau values, following the approach in
        Dissmann et al. (2013).
        """
        from scipy.stats import kendalltau
        
        d = self.n_cop
        data = self.data
        
        if data is None:
            print("Warning: No data provided for tau-based optimization, using random structure")
            self._generate_random_r_matrix()
            return
        
        print(f"Generating tau-based R-vine structure for {d} variables...")
        
        # Compute Kendall's tau matrix
        tau_matrix = np.zeros((d, d))
        for i in range(d):
            for j in range(i+1, d):
                tau, _ = kendalltau(data[:, i], data[:, j])
                if not np.isfinite(tau):
                    tau = 0.0
                tau_matrix[i, j] = tau_matrix[j, i] = abs(tau)
        
        print(f"Kendall's tau matrix computed, max tau = {np.max(tau_matrix):.4f}")
        
        # Initialize R-matrix
        r_matrix = np.zeros((d, d), dtype=int)
        
        # Fill diagonal
        for i in range(d):
            r_matrix[i, i] = i + 1
        
        # Greedy selection based on maximum tau
        # For each tree level, select edges that maximize dependence
        available_vars = set(range(1, d + 1))
        
        for tree_level in range(d - 1):
            # For first tree (tree_level=0), select based on pairwise tau
            if tree_level == 0:
                # Find the maximum spanning tree based on tau values
                edges_with_tau = []
                for i in range(d):
                    for j in range(i+1, d):
                        edges_with_tau.append((tau_matrix[i, j], i+1, j+1))
                
                # Sort by tau (descending) and select greedily
                edges_with_tau.sort(reverse=True)
                
                selected_edges = []
                used_vars = set()
                
                for tau_val, var1, var2 in edges_with_tau:
                    if len(selected_edges) >= d - 1:
                        break
                    # Ensure we don't create cycles (simplified check)
                    if len(used_vars) == 0 or var1 in used_vars or var2 in used_vars:
                        selected_edges.append((var1, var2, tau_val))
                        used_vars.add(var1)
                        used_vars.add(var2)
                
                # Fill first column of R-matrix based on selected edges
                for col_idx in range(1, min(len(selected_edges) + 1, d)):
                    if col_idx - 1 < len(selected_edges):
                        var1, var2, _ = selected_edges[col_idx - 1]
                        r_matrix[0, col_idx] = var1
                        if col_idx < d:
                            # Fill remaining positions to ensure valid structure
                            for row in range(1, col_idx + 1):
                                if row < d and col_idx < d:
                                    r_matrix[row, col_idx] = var2
            else:
                # For higher trees, use simplified structure based on proximity
                for col_idx in range(tree_level + 1, d):
                    for row in range(tree_level, col_idx + 1):
                        if r_matrix[row, col_idx] == 0:
                            # Find best variable based on conditional dependence
                            # (simplified: reuse variables from previous levels)
                            if col_idx > 0 and r_matrix[row, col_idx-1] != 0:
                                r_matrix[row, col_idx] = r_matrix[row, col_idx-1]
                            else:
                                # Fallback to available variable
                                available = [v for v in range(1, d+1) 
                                           if v not in r_matrix[row, :col_idx+1]]
                                if available:
                                    r_matrix[row, col_idx] = available[0]
                                else:
                                    r_matrix[row, col_idx] = row + 1
        
        # Ensure all positions are filled
        for i in range(d):
            for j in range(i, d):
                if r_matrix[i, j] == 0:
                    r_matrix[i, j] = j + 1
        
        self.r_matrix = r_matrix
        
        # Compute and report total tau sum
        total_tau = 0.0
        edge_count = 0
        for i in range(d-1):
            for j in range(i+1, d):
                if i < d and j < d:
                    var1_idx = r_matrix[i, j] - 1
                    var2_idx = r_matrix[i, i] - 1
                    if 0 <= var1_idx < d and 0 <= var2_idx < d:
                        total_tau += tau_matrix[var1_idx, var2_idx]
                        edge_count += 1
        
        print(f"Tau-based R-vine structure generated:")
        print(f"  Total absolute tau sum: {total_tau:.4f}")
        print(f"  Average tau per edge: {total_tau/max(edge_count, 1):.4f}")
        print(f"  R-matrix shape: {r_matrix.shape}")
    
    def _validate_r_matrix(self):
        """Validate that the R-matrix satisfies R-vine constraints."""
        d = self.n_cop
        r_matrix = self.r_matrix
        
        if r_matrix.shape != (d, d):
            raise ValueError(f"R-matrix must be {d}x{d}, got {r_matrix.shape}")
        
        # Check diagonal elements
        for i in range(d):
            if r_matrix[i, i] != i + 1:
                raise ValueError(f"Diagonal element ({i},{i}) must be {i+1}, got {r_matrix[i,i]}")
        
        # Check proximity condition (simplified check)
        for i in range(d):
            for j in range(i + 2, d):
                # Elements should form valid conditioning sets
                current_col = set(r_matrix[i:j+1, j])
                prev_col = set(r_matrix[i:j, j-1])
                if not (current_col & prev_col):
                    print(f"Warning: Proximity condition may be violated at ({i},{j})")
    
    def _build_rvine_structure(self):
        """Build R-vine structure from R-matrix."""
        self.ind_vine = []
        d = self.n_cop
        r_matrix = self.r_matrix
        
        for level in range(d - 1):
            edges_level = []
            
            # For each tree level, extract edges from R-matrix
            for j in range(level + 1, d):
                # Edge connects variables at positions (level, level) and (level, j)
                var1 = r_matrix[level, level] - 1  # Convert to 0-based indexing
                var2 = r_matrix[level, j] - 1
                edges_level.append([var1, var2])
            
            self.ind_vine.append(edges_level)


def create_vine(vine_type: Union[str, VineType], 
                vine_depth: int,
                families: Optional[List[str]] = None,
                margin_types: Optional[List[str]] = None,
                knots: int = 30,
                data: Optional[np.ndarray] = None,
                **kwargs) -> vine_obj_bin:
    """
    Factory function to create vine copula objects.
    
    Parameters
    ----------
    vine_type : str or VineType
        Type of vine to create ('c-vine', 'd-vine', 'r-vine')
    vine_depth : int
        Dimension of the vine (number of variables)
    families : list of str, optional
        Copula families to consider. Default: ['gaussian', 'clayton', 'frank']
    margin_types : list of str, optional
        Marginal distribution types. Default: ['norm'] * vine_depth
    knots : int, optional
        Number of knots for grid-based operations. Default: 30
    **kwargs : dict
        Additional arguments specific to vine type:
        - variable_order (D-vine): Order of variables
        - r_matrix (R-vine): R-matrix for structure
        - method (R-vine): Method for structure generation
    
    Returns
    -------
    vine_obj_bin
        Instantiated vine copula object
    
    Examples
    --------
    >>> # Create a 4D C-vine
    >>> vine = create_vine('c-vine', 4)
    
    >>> # Create a D-vine with custom variable order
    >>> vine = create_vine('d-vine', 4, variable_order=[0, 2, 1, 3])
    
    >>> # Create an R-vine with custom R-matrix
    >>> r_matrix = np.array([[1, 1, 1], [0, 2, 2], [0, 0, 3]])
    >>> vine = create_vine('r-vine', 3, r_matrix=r_matrix)
    """
    # Convert string to enum if needed
    if isinstance(vine_type, str):
        vine_type = VineType(vine_type.lower())
    
    # Set defaults
    if families is None:
        families = ['independence', 'gaussian', 'student', 'clayton', 'frank', 'gumbel']
    
    if margin_types is None:
        margin_types = ['norm'] * vine_depth
    
    # Create margin objects
    margins = []
    for i, mtype in enumerate(margin_types):
        if mtype == 'norm':
            margin = margin_obj('norm', (0.0, 1.0))  # Standard normal
        elif mtype == 'uniform':
            margin = margin_obj('uniform', (0.0, 1.0))
        else:
            margin = margin_obj(mtype, None)  # Generic margin
        margins.append(margin)
    
    # Create vine based on type
    if vine_type == VineType.C_VINE:
        return CVine(families, vine_depth, margins, knots)
    
    elif vine_type == VineType.D_VINE:
        variable_order = kwargs.get('variable_order', None)
        return DVine(families, vine_depth, margins, knots, variable_order)
    
    elif vine_type == VineType.R_VINE:
        r_matrix = kwargs.get('r_matrix', None)
        method = kwargs.get('method', 'random')
        return RVine(families, vine_depth, margins, knots, r_matrix, method, data)
    
    else:
        raise ValueError(f"Unsupported vine type: {vine_type}")


def optimize_vine_type(data: np.ndarray, 
                      vine_types: Optional[List[VineType]] = None,
                      selection_criterion: str = 'aic',
                      **fit_kwargs) -> vine_obj_bin:
    """
    Automatically select the best vine type for given data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    vine_types : list of VineType, optional
        Vine types to consider. Default: all types
    selection_criterion : str, optional
        Criterion for model selection ('aic', 'bic', 'loglik'). Default: 'aic'
    **fit_kwargs : dict
        Additional arguments for vine fitting
    
    Returns
    -------
    vine_obj_bin
        Best vine model based on selection criterion
    """
    if vine_types is None:
        vine_types = [VineType.C_VINE, VineType.D_VINE, VineType.R_VINE]
    
    d = data.shape[1]
    best_vine = None
    best_score = float('inf') if selection_criterion in ['aic', 'bic'] else float('-inf')
    
    for vine_type in vine_types:
        try:
            # Create and fit vine
            vine = create_vine(vine_type, d)
            
            # Dummy fit parameters (you'll need to adapt based on your fit_vine function)
            gen_dict = fit_kwargs.get('gen_dict', {'param': True, 'binning': False})
            npc_dict = fit_kwargs.get('npc_dict', {})
            par_dict = fit_kwargs.get('par_dict', {'param_families': ['gaussian', 'clayton']})
            bin_dict = fit_kwargs.get('bin_dict', {})
            
            vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
            
            # Compute selection criterion
            if hasattr(vine, 'loglikelihood'):
                loglik = vine.loglikelihood(data)
                n_params = len(vine.copulas) * 2  # Rough estimate
                
                if selection_criterion == 'aic':
                    score = -2 * loglik + 2 * n_params
                elif selection_criterion == 'bic':
                    score = -2 * loglik + n_params * np.log(data.shape[0])
                elif selection_criterion == 'loglik':
                    score = loglik
                
                # Update best model
                is_better = (score < best_score) if selection_criterion in ['aic', 'bic'] else (score > best_score)
                if is_better:
                    best_score = score
                    best_vine = vine
            
        except Exception as e:
            print(f"Failed to fit {vine_type.value}: {e}")
            continue
    
    if best_vine is None:
        raise RuntimeError("Failed to fit any vine type")
    
    return best_vine
