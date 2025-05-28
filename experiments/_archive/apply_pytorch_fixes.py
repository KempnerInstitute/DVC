"""
Script to apply PyTorch DVC fixes to the existing codebase.

This script:
1. Backs up original files
2. Applies fixes to vine_model.py in src/DVC
3. Updates related files for consistency
4. Runs tests to verify fixes
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
import pytest
import re

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_type_hints(content):
    """Convert Python 3.9+ type hints to Python 3.8 compatible format."""
    # First, handle function definitions with type hints
    def convert_function(match):
        # Extract function name and parameters
        func_name = match.group(1)
        params = match.group(2)
        return_type = match.group(3)
        
        # Convert parameters
        param_list = []
        type_hints = []
        
        # Split parameters and handle each one
        for param in params.split(','):
            param = param.strip()
            if ':' in param:
                name, type_hint = param.split(':', 1)
                name = name.strip()
                type_hint = type_hint.strip()
                if '=' in name:
                    name, default = name.split('=', 1)
                    name = name.strip()
                    param_list.append(f"{name}={default.strip()}")
                else:
                    param_list.append(name)
                type_hints.append(f"{name}: {type_hint}")
            else:
                param_list.append(param)
        
        # Build new function definition
        new_def = f"def {func_name}({', '.join(param_list)}):\n"
        if type_hints or return_type:
            new_def += "    # type: ("
            if type_hints:
                new_def += ", ".join(type_hints)
            if return_type:
                if type_hints:
                    new_def += ", "
                new_def += f"return -> {return_type}"
            new_def += ")\n"
        
        return new_def
    
    # Pattern to match function definitions with type hints
    pattern = r"(?<!#)def\s+(\w+)\s*\(([\s\S]*?)\)\s*(?:->\s*([\w\[\],\s\.]*?))?\s*:"
    
    # Apply the conversion
    content = re.sub(pattern, convert_function, content)
    
    # Handle any remaining type hints in function bodies
    def remove_type_hints(line):
        if ':' in line and '->' in line:
            # Skip lines in docstrings or comments
            if line.strip().startswith('#') or line.strip().startswith('"""') or line.strip().startswith("'''"):
                return line
            # Remove type hints from the line
            line = re.sub(r'\s*:\s*[\w\[\],\s\.]*?\s*->\s*[\w\[\],\s\.]*?(?=\s|$)', '', line)
        return line
    
    # Process each line
    lines = content.split('\n')
    in_docstring = False
    processed_lines = []
    
    for line in lines:
        # Check if we're entering/leaving a docstring
        if '"""' in line or "'''" in line:
            # Count occurrences
            triple_quotes = line.count('"""') + line.count("'''")
            if triple_quotes % 2 == 1:  # Odd number means we're toggling
                in_docstring = not in_docstring
        
        # Only process type hints if we're not in a docstring
        if not in_docstring:
            line = remove_type_hints(line)
        
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def backup_files():
    """Create backups of files we'll modify."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backup_{timestamp}")
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "src/DVC/vine_model.py",
        "src/DVC/vine_eval.py",
        "src/DVC/param_copula.py",
        "src/DVC/utils_prob.py",
        "src/DVC/utils_locallik.py"
    ]
    
    for file in files_to_backup:
        if os.path.exists(file):
            backup_path = backup_dir / Path(file).name
            shutil.copy2(file, backup_path)
            logger.info(f"Backed up {file}")
    
    return backup_dir

def update_vine_model():
    """Update vine_model.py with fixes."""
    vine_model_path = Path("src/DVC/vine_model.py")
    
    # Read existing file
    with open(vine_model_path, 'r') as f:
        content = f.read()
    
    # Convert type hints
    content = convert_type_hints(content)
    
    # Add imports
    imports_to_add = """
from .core_fixes import (
    eval_rs_cop_fixed,
    cdf_grid_fun_fixed,
    h_function_fixed,
    update_theta_fixed,
    sample_vine_fixed,
    generate_conditional
)
"""
    
    # Add after existing imports
    import_end = content.find("# Basic objects")
    if import_end == -1:
        import_end = content.find("\n\n")
    new_content = content[:import_end] + imports_to_add + content[import_end:]
    
    # Replace eval_rs_cop implementation
    old_eval_rs = "def eval_rs_cop"
    new_eval_rs = """def eval_rs_cop(pd_grid_uv):
    # type: (torch.Tensor) -> torch.Tensor
    \"\"\"Row/column normalization with 500 iterations and 1e-30 epsilon.\"\"\"
    return eval_rs_cop_fixed(pd_grid_uv)
"""
    new_content = new_content.replace(old_eval_rs, new_eval_rs)
    
    # Update kernel CDF usage
    old_kernel = "kernel_cdf(ccdf_data"
    new_kernel = "cdf_grid_fun_fixed(ccdf_data, grid_u.ex)"
    new_content = new_content.replace(old_kernel, new_kernel)
    
    # Update h-function
    old_h = "def h_function"
    new_h = """def h_function(u_root, u_other, cobj, grid_u=None, side="left"):
    # type: (torch.Tensor, torch.Tensor, Any, Optional[grid_obj], str) -> torch.Tensor
    \"\"\"Fixed h-function implementation matching TensorFlow.\"\"\"
    return h_function_fixed(u_root, u_other, cobj, grid_u, side)
"""
    new_content = new_content.replace(old_h, new_h)
    
    # Update theta updates
    old_update = "def update_theta"
    new_update = """def update_theta(vine, tr, edge, cobj, u_i, u_j, parent, is_flip):
    # type: (vine_obj_bin, int, List[int], Any, torch.Tensor, torch.Tensor, int, bool) -> None
    \"\"\"Fixed theta/theta_flip update matching TensorFlow.\"\"\"
    return update_theta_fixed(vine, tr, edge, cobj, u_i, u_j, parent, is_flip)
"""
    new_content = new_content.replace(old_update, new_update)
    
    # Update sampling
    old_sample = "def sample_vine"
    new_sample = """def sample_vine(vine, nsamples, cfg=None):
    # type: (vine_obj_bin, int, Optional[dict]) -> np.ndarray
    \"\"\"Fixed vine sampling matching TensorFlow.\"\"\"
    return sample_vine_fixed(vine, nsamples, cfg)
"""
    new_content = new_content.replace(old_sample, new_sample)
    
    # Write updated file
    with open(vine_model_path, 'w') as f:
        f.write(new_content)
    logger.info("Updated vine_model.py with fixes")

def update_vine_eval():
    """Update vine_eval.py for consistency."""
    vine_eval_path = Path("src/DVC/vine_eval.py")
    
    # Read existing file
    with open(vine_eval_path, 'r') as f:
        content = f.read()
    
    # Convert type hints
    content = convert_type_hints(content)
    
    # Add imports
    imports_to_add = """
from .core_fixes import eval_rs_cop_fixed, cdf_grid_fun_fixed
"""
    new_content = imports_to_add + content
    
    # Update eval_rs_cop usage
    old_eval = "eval_rs_cop(pd_grid_uv"
    new_eval = "eval_rs_cop_fixed(pd_grid_uv"
    new_content = new_content.replace(old_eval, new_eval)
    
    # Update kernel CDF usage
    old_kernel = "kernel_cdf(ccdf_data"
    new_kernel = "cdf_grid_fun_fixed(ccdf_data, grid_u.ex"
    new_content = new_content.replace(old_kernel, new_kernel)
    
    # Write updated file
    with open(vine_eval_path, 'w') as f:
        f.write(new_content)
    logger.info("Updated vine_eval.py for consistency")

def update_utils_prob():
    """Update utils_prob.py for consistency."""
    utils_prob_path = Path("src/DVC/utils_prob.py")
    
    # Read existing file
    with open(utils_prob_path, 'r') as f:
        content = f.read()
    
    # Convert type hints
    content = convert_type_hints(content)
    
    # Add imports
    imports_to_add = """
from .core_fixes import h_function_fixed
"""
    new_content = imports_to_add + content
    
    # Update h-function usage
    old_h = "def h_function"
    new_h = """def h_function(u_root, u_other, cobj, grid_u=None, side="left"):
    # type: (torch.Tensor, torch.Tensor, Any, Optional[grid_obj], str) -> torch.Tensor
    \"\"\"Fixed h-function implementation matching TensorFlow.\"\"\"
    return h_function_fixed(u_root, u_other, cobj, grid_u, side)
"""
    new_content = new_content.replace(old_h, new_h)
    
    # Write updated file
    with open(utils_prob_path, 'w') as f:
        f.write(new_content)
    logger.info("Updated utils_prob.py for consistency")

def update_utils_locallik():
    """Update utils_locallik.py for consistency."""
    utils_locallik_path = Path("src/DVC/utils_locallik.py")
    
    # Read existing file
    with open(utils_locallik_path, 'r') as f:
        content = f.read()
    
    # Convert type hints
    content = convert_type_hints(content)
    
    # Add imports
    imports_to_add = """
from .core_fixes import eval_rs_cop_fixed
"""
    new_content = imports_to_add + content
    
    # Update eval_rs_cop usage
    old_eval = "eval_rs_cop(pd_grid_uv"
    new_eval = "eval_rs_cop_fixed(pd_grid_uv"
    new_content = new_content.replace(old_eval, new_eval)
    
    # Write updated file
    with open(utils_locallik_path, 'w') as f:
        f.write(new_content)
    logger.info("Updated utils_locallik.py for consistency")

def run_tests():
    """Run tests to verify fixes."""
    test_result = pytest.main([
        "src/DVC/test_integrated_fixes.py",
        "-v"
    ])
    
    if test_result == 0:
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed!")
        raise Exception("Fix verification failed")

def main():
    """Main function to apply fixes."""
    try:
        # Backup files
        backup_dir = backup_files()
        logger.info(f"Files backed up to {backup_dir}")
        
        # Apply fixes
        update_vine_model()
        update_vine_eval()
        update_utils_prob()
        update_utils_locallik()
        
        # Run tests
        run_tests()
        
        logger.info("Successfully applied all fixes!")
        
    except Exception as e:
        logger.error(f"Error applying fixes: {str(e)}")
        # Restore from backup if needed
        if 'backup_dir' in locals():
            logger.info("Restoring from backup...")
            for file in backup_dir.iterdir():
                dest_path = Path("src/DVC") / file.name
                shutil.copy2(file, dest_path)
            logger.info("Restored from backup")
        raise

if __name__ == "__main__":
    main() 