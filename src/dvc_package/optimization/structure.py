"""
Vine Structure Optimization Algorithms

Implements various algorithms for optimizing vine copula structures:
1. Sequential (greedy) selection based on dependence measures
2. Genetic algorithm for R-vine matrix optimization  
3. Entropy-based optimization for information-theoretic criteria
4. Hybrid approaches combining multiple methods
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import kendalltau

from ..core.objects import vine_obj_bin
from ..core.vine_factory import create_vine, VineType
from .criteria import kendall_tau_criterion, entropy_criterion, aic_criterion

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from vine structure optimization."""
    best_vine: vine_obj_bin
    best_score: float
    optimization_history: List[float]
    method: str
    iterations: int
    convergence_info: Dict[str, Any]


def optimize_vine_structure(
    data: np.ndarray,
    vine_type: Union[str, VineType] = "r-vine",
    method: str = "sequential",
    criterion: str = "kendall_tau",
    max_iterations: int = 100,
    population_size: int = 50,
    mutation_rate: float = 0.1,
    convergence_tol: float = 1e-6,
    verbose: bool = True,
    **kwargs
) -> OptimizationResult:
    """
    Optimize vine copula structure using specified method and criterion.
    
    Parameters
    ----------
    data : np.ndarray
        Input data of shape (n_samples, n_features)
    vine_type : str or VineType
        Type of vine to optimize ('c-vine', 'd-vine', 'r-vine')
    method : str
        Optimization method ('sequential', 'genetic', 'entropy', 'hybrid')
    criterion : str
        Selection criterion ('kendall_tau', 'entropy', 'aic', 'bic')
    max_iterations : int
        Maximum number of optimization iterations
    population_size : int
        Population size for genetic algorithm
    mutation_rate : float
        Mutation rate for genetic algorithm
    convergence_tol : float
        Convergence tolerance
    verbose : bool
        Whether to print progress information
    **kwargs
        Additional arguments for specific methods
    
    Returns
    -------
    OptimizationResult
        Results containing optimized vine and optimization details
    """
    if verbose:
        logger.info(f"Starting vine structure optimization: {method} method, {criterion} criterion")
    
    # Select optimization method
    if method == "sequential":
        return sequential_vine_optimization(
            data, vine_type, criterion, max_iterations, verbose, **kwargs
        )
    elif method == "genetic":
        return genetic_vine_optimization(
            data, vine_type, criterion, max_iterations, population_size, 
            mutation_rate, convergence_tol, verbose, **kwargs
        )
    elif method == "entropy":
        return entropy_based_optimization(
            data, vine_type, max_iterations, verbose, **kwargs
        )
    elif method == "hybrid":
        return hybrid_optimization(
            data, vine_type, criterion, max_iterations, verbose, **kwargs
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def sequential_vine_optimization(
    data: np.ndarray,
    vine_type: Union[str, VineType],
    criterion: str = "kendall_tau",
    max_iterations: int = 100,
    verbose: bool = True,
    **kwargs
) -> OptimizationResult:
    """
    Sequential (greedy) vine structure optimization.
    
    Builds vine structure level by level, selecting edges that maximize
    the chosen dependence criterion at each step.
    """
    d = data.shape[1]
    n_samples = data.shape[0]
    
    # Convert to uniform margins for dependence computation
    data_u = np.zeros_like(data)
    for i in range(d):
        ranks = data[:, i].argsort().argsort() + 1
        data_u[:, i] = ranks / (n_samples + 1)
    
    optimization_history = []
    
    if vine_type in ["c-vine", VineType.C_VINE]:
        # C-vine: optimize variable ordering (root selection)
        best_vine = _optimize_cvine_sequential(data, data_u, criterion, verbose)
        
    elif vine_type in ["d-vine", VineType.D_VINE]:  
        # D-vine: optimize variable ordering
        best_vine = _optimize_dvine_sequential(data, data_u, criterion, verbose)
        
    elif vine_type in ["r-vine", VineType.R_VINE]:
        # R-vine: optimize R-matrix structure
        best_vine = _optimize_rvine_sequential(data, data_u, criterion, max_iterations, verbose)
        
    else:
        raise ValueError(f"Unsupported vine type for sequential optimization: {vine_type}")
    
    # Compute final score
    final_score = _evaluate_vine_criterion(best_vine, data, criterion)
    optimization_history.append(final_score)
    
    return OptimizationResult(
        best_vine=best_vine,
        best_score=final_score,
        optimization_history=optimization_history,
        method="sequential",
        iterations=1,
        convergence_info={"converged": True}
    )


def _optimize_cvine_sequential(data: np.ndarray, data_u: np.ndarray, 
                              criterion: str, verbose: bool) -> vine_obj_bin:
    """Optimize C-vine by selecting best root variable ordering."""
    d = data.shape[1]

    # Greedy ordering scales to moderate/high d and is deterministic enough for benchmarking.
    dep = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(i + 1, d):
            if criterion == "kendall_tau":
                tau, _ = kendalltau(data_u[:, i], data_u[:, j])
                val = abs(tau) if np.isfinite(tau) else 0.0
            elif criterion in {"entropy", "aic", "bic"}:
                # Proxy: pairwise MI on pseudo-observations.
                val = float(_pairwise_mutual_information(data_u[:, i], data_u[:, j]))
            else:
                tau, _ = kendalltau(data_u[:, i], data_u[:, j])
                val = abs(tau) if np.isfinite(tau) else 0.0
            dep[i, j] = dep[j, i] = val

    remaining = list(range(d))
    order: List[int] = []
    while len(remaining) > 1:
        scores = [float(dep[v, remaining].sum() - dep[v, v]) for v in remaining]
        best_idx = int(np.argmax(scores))
        best_var = int(remaining.pop(best_idx))
        order.append(best_var)
    order.extend(remaining)

    if verbose:
        logger.info(f"Greedy C-vine root order: {order}")

    vine = create_vine("c-vine", d)
    vine.variable_order = order
    _reorder_vine_structure(vine, order)
    return vine


def _optimize_dvine_sequential(data: np.ndarray, data_u: np.ndarray,
                              criterion: str, verbose: bool) -> vine_obj_bin:
    """Optimize D-vine by finding best variable ordering."""
    d = data.shape[1]
    
    # Use spanning tree approach for D-vine ordering
    # Build complete graph with edge weights = dependence measures
    edge_weights = np.zeros((d, d))
    
    for i in range(d):
        for j in range(i + 1, d):
            if criterion == "kendall_tau":
                tau, _ = kendalltau(data_u[:, i], data_u[:, j])
                edge_weights[i, j] = abs(tau)
                edge_weights[j, i] = abs(tau)
            elif criterion == "entropy":
                mi = _pairwise_mutual_information(data_u[:, i], data_u[:, j])
                edge_weights[i, j] = mi
                edge_weights[j, i] = mi
    
    # Find maximum spanning tree for D-vine ordering
    mst_edges = _maximum_spanning_tree(edge_weights)
    variable_order = _path_from_mst(mst_edges, d)
    
    if verbose:
        logger.info(f"Best D-vine variable order: {variable_order}")
    
    # Create optimized D-vine
    vine = create_vine("d-vine", d, variable_order=variable_order)
    return vine


def _optimize_rvine_sequential(data: np.ndarray, data_u: np.ndarray,
                              criterion: str, max_iterations: int, 
                              verbose: bool) -> vine_obj_bin:
    """Optimize R-vine structure using sequential tree construction."""
    d = data.shape[1]
    
    # Build R-matrix level by level
    r_matrix = np.zeros((d, d), dtype=int)
    
    # Initialize diagonal
    for i in range(d):
        r_matrix[i, i] = i + 1
    
    # Build each tree level
    for level in range(d - 1):
        if verbose and level % 5 == 0:
            logger.info(f"Optimizing R-vine tree level {level + 1}/{d - 1}")
        
        # Get available variables for this level
        available_vars = list(range(1, d + 1))
        used_vars = set(r_matrix[level, level:level+1])
        available_vars = [v for v in available_vars if v not in used_vars]
        
        # For each position in this level
        for j in range(level + 1, d):
            best_var = None
            best_score = float('-inf')
            
            # Try each available variable
            for var in available_vars:
                # Temporarily assign variable
                r_matrix[level, j] = var
                
                # Compute score for this assignment
                score = _evaluate_r_matrix_partial(r_matrix, data_u, level, j, criterion)
                
                if score > best_score:
                    best_score = score
                    best_var = var
            
            # Assign best variable
            if best_var is not None:
                r_matrix[level, j] = best_var
                available_vars.remove(best_var)
            else:
                # Fallback: assign first available
                if available_vars:
                    r_matrix[level, j] = available_vars.pop(0)
                else:
                    r_matrix[level, j] = level + 1
    
    if verbose:
        logger.info(f"Optimized R-matrix:\n{r_matrix}")
    
    # Create optimized R-vine
    vine = create_vine("r-vine", d, r_matrix=r_matrix)
    return vine


def genetic_vine_optimization(
    data: np.ndarray,
    vine_type: Union[str, VineType],
    criterion: str = "kendall_tau", 
    max_generations: int = 100,
    population_size: int = 50,
    mutation_rate: float = 0.1,
    convergence_tol: float = 1e-6,
    verbose: bool = True,
    **kwargs
) -> OptimizationResult:
    """
    Genetic algorithm for vine structure optimization.
    
    Evolves a population of vine structures, using crossover and mutation
    operators to search for optimal structures.
    """
    d = data.shape[1]
    optimization_history = []
    
    if verbose:
        logger.info(f"Starting genetic optimization: {population_size} individuals, {max_generations} generations")
    
    # Initialize population
    if vine_type in ["r-vine", VineType.R_VINE]:
        population = _initialize_rvine_population(d, population_size)
    elif vine_type in ["d-vine", VineType.D_VINE]:
        population = _initialize_dvine_population(d, population_size)
    else:
        raise ValueError(f"Genetic optimization not implemented for {vine_type}")
    
    lower_is_better = criterion in {"aic", "bic"}
    best_score = float("inf") if lower_is_better else float("-inf")
    best_individual = None
    stagnation_count = 0
    
    for generation in range(max_generations):
        # Evaluate population
        scores = []
        for individual in population:
            vine = _individual_to_vine(individual, vine_type, d)
            score = _evaluate_vine_criterion(vine, data, criterion)
            scores.append(score)
        
        # Track best individual
        gen_best_idx = int(np.argmin(scores) if lower_is_better else np.argmax(scores))
        gen_best_score = scores[gen_best_idx]
        
        improved = gen_best_score < best_score if lower_is_better else gen_best_score > best_score
        if improved:
            best_score = gen_best_score
            best_individual = population[gen_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        optimization_history.append(best_score)
        
        if verbose and generation % 10 == 0:
            logger.info(f"Generation {generation}: best score = {best_score:.4f}")
        
        # Check convergence
        if stagnation_count > 20:
            if verbose:
                logger.info(f"Converged after {generation} generations (stagnation)")
            break
        
        # Selection, crossover, mutation
        population = _genetic_operators(population, scores, mutation_rate, population_size)
    
    # Create final vine from best individual
    best_vine = _individual_to_vine(best_individual, vine_type, d)
    
    return OptimizationResult(
        best_vine=best_vine,
        best_score=best_score,
        optimization_history=optimization_history,
        method="genetic",
        iterations=generation + 1,
        convergence_info={"converged": stagnation_count > 20, "stagnation_count": stagnation_count}
    )


def entropy_based_optimization(
    data: np.ndarray,
    vine_type: Union[str, VineType],
    max_iterations: int = 50,
    verbose: bool = True,
    **kwargs
) -> OptimizationResult:
    """
    Entropy-based vine structure optimization.
    
    Uses information-theoretic criteria to build vine structures that
    maximize mutual information or minimize conditional entropy.
    """
    d = data.shape[1]
    
    if verbose:
        logger.info("Starting entropy-based optimization")
    
    # Convert to uniform margins
    data_u = np.zeros_like(data)
    for i in range(d):
        ranks = data[:, i].argsort().argsort() + 1
        data_u[:, i] = ranks / (data.shape[0] + 1)
    
    # Build mutual information matrix
    mi_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            mi = _pairwise_mutual_information(data_u[:, i], data_u[:, j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    
    optimization_history = []
    
    if vine_type in ["r-vine", VineType.R_VINE]:
        # Use entropy to guide R-matrix construction
        r_matrix = _build_entropy_optimal_r_matrix(mi_matrix, verbose)
        best_vine = create_vine("r-vine", d, r_matrix=r_matrix)
        
    elif vine_type in ["d-vine", VineType.D_VINE]:
        # Find path that maximizes total MI
        variable_order = _entropy_optimal_path(mi_matrix)
        best_vine = create_vine("d-vine", d, variable_order=variable_order)
        
    else:
        # Default to sequential method for other types
        return sequential_vine_optimization(data, vine_type, "entropy", max_iterations, verbose)
    
    final_score = entropy_criterion(best_vine, data)
    optimization_history.append(final_score)
    
    return OptimizationResult(
        best_vine=best_vine,
        best_score=final_score,
        optimization_history=optimization_history,
        method="entropy",
        iterations=1,
        convergence_info={"converged": True}
    )


def hybrid_optimization(
    data: np.ndarray,
    vine_type: Union[str, VineType],
    criterion: str = "kendall_tau",
    max_iterations: int = 100,
    verbose: bool = True,
    **kwargs
) -> OptimizationResult:
    """
    Hybrid optimization combining multiple methods.
    
    Uses sequential optimization for initialization, then genetic algorithm
    for refinement, with optional entropy-based post-processing.
    """
    if verbose:
        logger.info("Starting hybrid optimization")
    
    # Phase 1: Sequential initialization
    seq_result = sequential_vine_optimization(data, vine_type, criterion, 10, False)
    
    # Phase 2: Genetic refinement (if R-vine or D-vine)
    if vine_type in ["r-vine", "d-vine"]:
        # Use sequential result as seed for genetic algorithm
        genetic_result = genetic_vine_optimization(
            data, vine_type, criterion, max_iterations // 2, 30, 0.15, 1e-6, False
        )
        
        # Choose better result
        lower_is_better = criterion in {"aic", "bic"}
        genetic_better = (
            genetic_result.best_score < seq_result.best_score
            if lower_is_better else
            genetic_result.best_score > seq_result.best_score
        )
        if genetic_better:
            best_result = genetic_result
        else:
            best_result = seq_result
    else:
        best_result = seq_result
    
    # Phase 3: Optional entropy post-processing
    if criterion != "entropy":
        entropy_result = entropy_based_optimization(data, vine_type, 10, False)
        
        # Compare entropy-based result
        entropy_score = _evaluate_vine_criterion(entropy_result.best_vine, data, criterion)
        lower_is_better = criterion in {"aic", "bic"}
        entropy_better = (
            entropy_score < best_result.best_score
            if lower_is_better else
            entropy_score > best_result.best_score
        )
        if entropy_better:
            best_result.best_vine = entropy_result.best_vine
            best_result.best_score = entropy_score
    
    best_result.method = "hybrid"
    
    if verbose:
        logger.info(f"Hybrid optimization completed: final score = {best_result.best_score:.4f}")
    
    return best_result


# Helper functions

def _evaluate_vine_criterion(vine: vine_obj_bin, data: np.ndarray, criterion: str) -> float:
    """Evaluate vine using specified criterion."""
    try:
        if criterion == "kendall_tau":
            return kendall_tau_criterion(vine, data)
        elif criterion == "entropy":
            return entropy_criterion(vine, data)
        elif criterion == "aic":
            return aic_criterion(vine, data)
        else:
            # Fallback to simple correlation-based score
            return _compute_correlation_score(vine, data)
    except Exception:
        return float("inf") if criterion in {"aic", "bic"} else float("-inf")


def _compute_correlation_score(vine: vine_obj_bin, data: np.ndarray) -> float:
    """Simple correlation-based scoring function."""
    d = data.shape[1]
    total_score = 0
    
    # Convert to uniform margins
    data_u = np.zeros_like(data)
    for i in range(d):
        ranks = data[:, i].argsort().argsort() + 1
        data_u[:, i] = ranks / (data.shape[0] + 1)
    
    # Sum absolute correlations for edges in vine structure
    for level, edges in enumerate(vine.ind_vine):
        for edge in edges:
            i, j = edge
            if i < d and j < d:
                tau, _ = kendalltau(data_u[:, i], data_u[:, j])
                total_score += abs(tau)
    
    return total_score


def _pairwise_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate mutual information between two variables using binning."""
    try:
        from sklearn.feature_selection import mutual_info_regression
        return mutual_info_regression(x.reshape(-1, 1), y)[0]
    except Exception:
        # Fallback: use correlation as proxy
        corr = np.corrcoef(x, y)[0, 1]
        return -0.5 * np.log(1 - corr**2) if abs(corr) < 0.99 else 2.0


def _generate_variable_permutations(d: int, max_permutations: int = 24) -> List[List[int]]:
    """Generate variable permutations for optimization."""
    import itertools

    total_perms = math.factorial(d)
    if total_perms <= max_permutations:
        return [list(p) for p in itertools.permutations(range(d))]

    # Avoid materializing all permutations for moderate/large d.
    seen = set()
    out: List[List[int]] = []
    while len(out) < max_permutations:
        perm = tuple(np.random.permutation(d).tolist())
        if perm not in seen:
            seen.add(perm)
            out.append(list(perm))
    return out


def _maximum_spanning_tree(weights: np.ndarray) -> List[Tuple[int, int]]:
    """Find maximum spanning tree using Kruskal's algorithm."""
    d = weights.shape[0]
    edges = []
    
    # Get all edges with weights
    for i in range(d):
        for j in range(i + 1, d):
            edges.append((weights[i, j], i, j))
    
    # Sort by weight (descending for maximum)
    edges.sort(reverse=True)
    
    # Kruskal's algorithm
    parent = list(range(d))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    mst_edges = []
    for weight, i, j in edges:
        if union(i, j):
            mst_edges.append((i, j))
            if len(mst_edges) == d - 1:
                break
    
    return mst_edges


def _path_from_mst(mst_edges: List[Tuple[int, int]], d: int) -> List[int]:
    """Extract a path ordering from maximum spanning tree."""
    # Build adjacency list
    adj = [[] for _ in range(d)]
    for i, j in mst_edges:
        adj[i].append(j)
        adj[j].append(i)
    
    # Find path: start from node with degree 1 (leaf) if exists
    start_node = 0
    for i in range(d):
        if len(adj[i]) == 1:
            start_node = i
            break
    
    # DFS to get path
    visited = [False] * d
    path = []
    
    def dfs(node):
        visited[node] = True
        path.append(node)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                dfs(neighbor)
    
    dfs(start_node)
    return path


def _reorder_vine_structure(vine: vine_obj_bin, new_order: List[int]):
    """Reorder vine structure according to new variable ordering."""
    d = len(new_order)

    if vine.vine_family == "c-vine":
        reordered = []
        for level in range(d - 1):
            center = new_order[level]
            edges_level = [[center, new_order[j]] for j in range(level + 1, d)]
            reordered.append(edges_level)
        vine.ind_vine = reordered
        return

    if vine.vine_family == "d-vine":
        reordered = []
        for level in range(d - 1):
            remaining = new_order[level:]
            edges_level = [[remaining[i], remaining[i + 1]] for i in range(len(remaining) - 1)]
            reordered.append(edges_level)
        vine.ind_vine = reordered
        return


def _evaluate_r_matrix_partial(r_matrix: np.ndarray, data_u: np.ndarray,
                              level: int, j: int, criterion: str) -> float:
    """Evaluate partial R-matrix assignment."""
    # Simplified evaluation - compute dependence for current assignment
    if level < r_matrix.shape[0] and j < r_matrix.shape[1]:
        var1 = r_matrix[level, level] - 1  # Convert to 0-based
        var2 = r_matrix[level, j] - 1
        
        if 0 <= var1 < data_u.shape[1] and 0 <= var2 < data_u.shape[1]:
            tau, _ = kendalltau(data_u[:, var1], data_u[:, var2])
            return abs(tau)
    
    return 0.0


def _initialize_rvine_population(d: int, population_size: int) -> List[np.ndarray]:
    """Initialize population of R-matrices for genetic algorithm."""
    population = []
    
    for _ in range(population_size):
        # Create random valid R-matrix
        r_matrix = np.zeros((d, d), dtype=int)
        
        # Fill diagonal
        for i in range(d):
            r_matrix[i, i] = i + 1
        
        # Fill upper triangle randomly
        for i in range(d):
            available = list(range(1, d + 1))
            available.remove(i + 1)
            
            for j in range(i + 1, d):
                if available:
                    choice = np.random.choice(available)
                    r_matrix[i, j] = choice
                    available.remove(choice)
                else:
                    r_matrix[i, j] = i + 1
        
        population.append(r_matrix)
    
    return population


def _initialize_dvine_population(d: int, population_size: int) -> List[List[int]]:
    """Initialize population of variable orderings for D-vine genetic algorithm."""
    population = []
    
    for _ in range(population_size):
        order = list(range(d))
        np.random.shuffle(order)
        population.append(order)
    
    return population


def _individual_to_vine(individual: Union[np.ndarray, List[int]], 
                       vine_type: Union[str, VineType], d: int) -> vine_obj_bin:
    """Convert genetic algorithm individual to vine object."""
    if vine_type in ["r-vine", VineType.R_VINE]:
        return create_vine("r-vine", d, r_matrix=individual)
    elif vine_type in ["d-vine", VineType.D_VINE]:
        return create_vine("d-vine", d, variable_order=individual)
    else:
        raise ValueError(f"Unsupported vine type: {vine_type}")


def _genetic_operators(population: List, scores: List[float], 
                      mutation_rate: float, population_size: int) -> List:
    """Apply genetic operators: selection, crossover, mutation."""
    # Tournament selection
    new_population = []
    
    for _ in range(population_size):
        # Tournament selection
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_scores = [scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        
        # Add winner (with possible mutation)
        individual = population[winner_idx].copy()
        
        # Mutation
        if np.random.random() < mutation_rate:
            individual = _mutate_individual(individual)
        
        new_population.append(individual)
    
    return new_population


def _mutate_individual(individual: Union[np.ndarray, List[int]]) -> Union[np.ndarray, List[int]]:
    """Mutate an individual in the genetic algorithm."""
    if isinstance(individual, np.ndarray):
        # R-matrix mutation: swap two random entries
        individual = individual.copy()
        d = individual.shape[0]
        
        # Choose random position in upper triangle
        i = np.random.randint(0, d)
        j = np.random.randint(i + 1, d)
        
        # Choose new random value (simple mutation)
        available = list(range(1, d + 1))
        if available:
            individual[i, j] = np.random.choice(available)
        
        return individual
        
    else:
        # Variable order mutation: swap two random positions
        individual = individual.copy()
        if len(individual) > 1:
            i, j = np.random.choice(len(individual), 2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]
        
        return individual


def _build_entropy_optimal_r_matrix(mi_matrix: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Build R-matrix that maximizes mutual information."""
    d = mi_matrix.shape[0]
    r_matrix = np.zeros((d, d), dtype=int)
    
    # Fill diagonal
    for i in range(d):
        r_matrix[i, i] = i + 1
    
    # Build each level greedily based on MI
    for level in range(d - 1):
        available_vars = list(range(1, d + 1))
        used_vars = {level + 1}
        
        for j in range(level + 1, d):
            best_var = None
            best_mi = -1
            
            for var in available_vars:
                if var not in used_vars:
                    # Compute MI for this assignment
                    var1_idx = level  # Center variable
                    var2_idx = var - 1  # Convert to 0-based
                    
                    if 0 <= var2_idx < d:
                        mi = mi_matrix[var1_idx, var2_idx]
                        if mi > best_mi:
                            best_mi = mi
                            best_var = var
            
            if best_var is not None:
                r_matrix[level, j] = best_var
                used_vars.add(best_var)
            else:
                # Fallback
                remaining = [v for v in available_vars if v not in used_vars]
                if remaining:
                    r_matrix[level, j] = remaining[0]
                    used_vars.add(remaining[0])
                else:
                    r_matrix[level, j] = level + 1
    
    if verbose:
        logger.info(f"Entropy-optimal R-matrix:\n{r_matrix}")
    
    return r_matrix


def _entropy_optimal_path(mi_matrix: np.ndarray) -> List[int]:
    """Find variable ordering that maximizes total MI for D-vine."""
    d = mi_matrix.shape[0]
    
    # Use greedy approach: start with highest MI pair, then extend path
    max_mi = -1
    best_start_pair = (0, 1)
    
    for i in range(d):
        for j in range(i + 1, d):
            if mi_matrix[i, j] > max_mi:
                max_mi = mi_matrix[i, j]
                best_start_pair = (i, j)
    
    # Build path starting from best pair
    path = [best_start_pair[0], best_start_pair[1]]
    used = set(path)
    
    while len(path) < d:
        best_next = None
        best_mi = -1
        
        # Try extending from either end of current path
        for end_idx in [0, len(path) - 1]:
            current_var = path[end_idx]
            
            for next_var in range(d):
                if next_var not in used:
                    mi = mi_matrix[current_var, next_var]
                    if mi > best_mi:
                        best_mi = mi
                        best_next = (next_var, end_idx)
        
        if best_next is not None:
            var, end_idx = best_next
            if end_idx == 0:
                path.insert(0, var)
            else:
                path.append(var)
            used.add(var)
        else:
            # Add remaining variables arbitrarily
            remaining = [v for v in range(d) if v not in used]
            if remaining:
                path.append(remaining[0])
                used.add(remaining[0])
    
    return path
