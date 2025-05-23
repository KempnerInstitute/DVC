# DVC Package - Deep Vine Copulas in PyTorch

from .objects import copula_obj, cop_par_obj, margin_obj, vine_obj_bin
from .vine_model import fit_vine
from .info_estimation import (
    vine_entropy, 
    cond_vine_entropy, 
    mutual_information,
    compute_max,
    theoretic_mutual_information_AWGN
)
from .prediction import (
    predict_vine,
    predict_response,
    predict_conditional,
    predict_conditional_quantiles,
    create_points,
    smooth,
    replace_nan_inf
)
from .vine_eval import (
    evaluate_fit,
    evaluate_points,
    evaluate_fit_bin
)
from .preparation import (
    prep_cop,
    prep_copula,
    define_copulas
)
from .sampling import (
    kerncopccdfinv,
    vine_copula_sample,
    vine_cop_par_sample
)
from .vine_tree import (
    parent_var,
    optimal_tree,
    random_tree,
    random_r_matrix_gen,
    prepare_optimal,
    prepare_regular,
    prepare_vine,
    flip_check_all
)
from .transformation import Transform
from .dataset_ops import (
    kfold,
    data_split,
    create_bins,
    check_bins
)
from .utils_prob import (
    biv_norm,
    kernel_cdf,
    kernel_cdf_batch,
    kernel_pdf2,
    kernel_pdf1d,
    copulapdf,
    copulaccdf,
    copulainvccdf
)
from .utils_tensor import (
    check_bound,
    check_bound3,
    replace_nan_inf,
    clamp_probs,
    replace_negative,
    create_points,
    moving_average,
    update_tensor_2d
)

__all__ = [
    # Core objects
    'copula_obj',
    'cop_par_obj', 
    'margin_obj',
    'vine_obj_bin',
    
    # Main functions
    'fit_vine',
    'define_copulas',
    'prep_cop',
    'prep_copula',
    
    # Information theory
    'vine_entropy',
    'cond_vine_entropy',
    'mutual_information',
    'compute_max',
    'theoretic_mutual_information_AWGN',
    
    # Prediction
    'predict_vine',
    'predict_response',
    'predict_conditional',
    'predict_conditional_quantiles',
    
    # Evaluation
    'evaluate_fit',
    'evaluate_points',
    'evaluate_fit_bin',
    
    # Sampling
    'kerncopccdfinv',
    'vine_copula_sample',
    'vine_cop_par_sample',
    
    # Tree operations
    'parent_var',
    'optimal_tree',
    'random_tree',
    'random_r_matrix_gen',
    'prepare_optimal',
    'prepare_regular',
    'prepare_vine',
    'flip_check_all',
    
    # Transformations
    'Transform',
    
    # Dataset operations
    'kfold',
    'data_split',
    'create_bins',
    'check_bins',
    
    # Probability utilities
    'biv_norm',
    'kernel_cdf',
    'kernel_cdf_batch',
    'kernel_pdf2',
    'kernel_pdf1d',
    'copulapdf',
    'copulaccdf',
    'copulainvccdf',
    
    # Tensor utilities
    'check_bound',
    'check_bound3',
    'replace_nan_inf',
    'clamp_probs',
    'replace_negative',
    'create_points',
    'moving_average',
    'update_tensor_2d'
]

__version__ = '0.1.0'
