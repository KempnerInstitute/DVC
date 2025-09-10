# DVC PyTorch Code Structure

## Directory Organization

### `classes/`
- `objects.py` - Core vine copula classes (vine_obj_bin, margin_obj, etc.)

### `param/`
- `parametric_copulas.py` - Parametric copula implementations
- `cond_copula.py` - Conditional CDF functions
- `copula_fit.py` - Copula fitting routines
- `margin_pdf.py` - Marginal PDF functions
- `margin_cost.py` - Marginal cost functions

### `utils/`
- `prob_op.py` - Probability operations and transformations
- `kde.py` - Kernel density estimation (bounded KDE)
- `bijector.py` - Bijection transformations
- `interpolation.py` - Interpolation utilities
- `tensor_op.py` - Tensor operations
- `dataset_op.py` - Dataset operations

### `sampling/`
- `vine_sampler.py` - Main vine sampling class (VineSampler)

### `optim/`
- `vine_fit.py` - Vine fitting optimization
- `bandwidth.py` - Bandwidth selection methods
- `local_lik.py` - Local likelihood optimization
- `MISE.py` - Mean integrated squared error
- `nadam.py` - Nadam optimizer implementation

### `pre_proc/`
- `preparation.py` - Data preparation utilities
- `transformation.py` - Data transformations

### `vine_tree/`
- `tree_op.py` - Vine tree operations

### `evalu/`
- `vine_eval.py` - Vine evaluation metrics
- `cop_eval.py` - Copula evaluation
- `vine_entropy.py` - Vine entropy calculations

### `pred/`
- `prediction.py` - Prediction utilities
- `vine_conditional.py` - Conditional vine operations

### `grid/`
- `grid_class.py` - Grid class definitions
- `grid_op.py` - Grid operations

### `info/`
- `info_estimation.py` - Information theory estimations

### `plot/`
- `plot_vine.py` - Vine visualization utilities

### `experiments/`
- (Empty - for experiment configurations)

## Key Classes and Functions

### Main Classes
- `vine_obj_bin` - Main vine copula object
- `margin_obj` - Marginal distribution object
- `VineSampler` - Vine sampling class

### Key Functions
- Copula fitting and parameter estimation
- Kernel density estimation with boundary correction
- H-function computations for vine sampling
- Probability integral transforms
- Vine structure construction and evaluation

## Usage Pattern

1. Create margin objects
2. Initialize vine object with desired structure
3. Fit vine to data
4. Sample from fitted vine or perform predictions

See `example.py` for a complete usage example. 