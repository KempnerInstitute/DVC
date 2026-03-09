#!/usr/bin/env python3
"""
Installation verification script for DVC environment.

This script tests that all required packages are installed and working correctly.
"""

import sys
import importlib
import traceback
from pathlib import Path

def test_import(module_name, display_name=None):
    """Test if a module can be imported."""
    if display_name is None:
        display_name = module_name
    
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"[OK] {display_name:<20} {version}")
        return True
    except ImportError as e:
        print(f"[FAIL] {display_name:<20} {e}")
        return False

def test_dvc_framework():
    """Test DVC framework components."""
    print("\nTesting DVC Framework Components:")
    print("-" * 50)
    
    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    
    dvc_components = [
        ("dvc_package.core.param_copula", "Core Parametric Copulas"),
        ("dvc_package.core.vine_factory", "Vine Factory"),
        ("dvc_package.core.vine_model", "Vine Model"),
        ("dvc_package.time.flows", "Time Flows"),
        ("dvc_package.time.models", "Time Models"),
        ("dvc_package.experiments.experiment_framework", "Experiment Framework"),
    ]
    
    success_count = 0
    for module_name, display_name in dvc_components:
        if test_import(module_name, display_name):
            success_count += 1
    
    print(f"\nDVC Framework: {success_count}/{len(dvc_components)} components working")
    return success_count == len(dvc_components)

def test_functionality():
    """Test basic functionality."""
    print("\nTesting Basic Functionality:")
    print("-" * 50)
    
    try:
        # Test PyTorch
        import torch
        x = torch.randn(10, 2)
        y = torch.matmul(x, x.T)
        print(f"[OK] PyTorch operations: tensor shape {y.shape}")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print(f"[OK] CUDA GPU detected: {device}")
        else:
            print("[INFO] CUDA not available (CPU-only mode)")
        
        # Test NumPy/SciPy
        import numpy as np
        from scipy.stats import multivariate_normal, kendalltau
        
        data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
        tau, _ = kendalltau(data[:, 0], data[:, 1])
        print(f"[OK] SciPy statistics: Kendall's tau = {tau:.3f}")
        
        # Test plotting
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(4, 3))
        plt.plot([1, 2, 3], [1, 4, 2])
        plt.close()
        print("[OK] Plotting libraries working")
        
        # Test YAML
        import yaml
        test_config = {"test": True, "value": 42}
        yaml_str = yaml.dump(test_config)
        loaded_config = yaml.safe_load(yaml_str)
        assert loaded_config["test"] == True
        print("[OK] YAML configuration working")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main verification function."""
    print("DVC Environment Verification")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Python path: {sys.executable}")
    
    # Test core packages
    print("\nTesting Core Packages:")
    print("-" * 50)
    
    core_packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("sklearn", "Scikit-learn"),
        ("yaml", "PyYAML"),
        ("jupyter", "Jupyter"),
        ("statsmodels", "Statsmodels"),
    ]
    
    success_count = 0
    for module_name, display_name in core_packages:
        if test_import(module_name, display_name):
            success_count += 1
    
    print(f"\nCore Packages: {success_count}/{len(core_packages)} working")
    
    # Test optional packages
    print("\nTesting Optional Packages:")
    print("-" * 50)
    
    optional_packages = [
        ("tensorboard", "TensorBoard"),
        ("tqdm", "TQDM"),
        ("rich", "Rich"),
        ("plotly", "Plotly"),
        ("optuna", "Optuna"),
    ]
    
    optional_success = 0
    for module_name, display_name in optional_packages:
        if test_import(module_name, display_name):
            optional_success += 1
    
    print(f"\nOptional Packages: {optional_success}/{len(optional_packages)} working")
    
    # Test functionality
    functionality_ok = test_functionality()
    
    # Test DVC framework
    dvc_framework_ok = test_dvc_framework()
    
    # Final summary
    print("\nFinal Summary:")
    print("=" * 50)
    
    if success_count == len(core_packages):
        print("[OK] Core packages: ALL WORKING")
    else:
        print(f"[WARN] Core packages: {success_count}/{len(core_packages)} working")
    
    if optional_success >= len(optional_packages) * 0.8:  # 80% threshold
        print("[OK] Optional packages: MOSTLY WORKING")
    else:
        print(f"[WARN] Optional packages: {optional_success}/{len(optional_packages)} working")
    
    if functionality_ok:
        print("[OK] Basic functionality: WORKING")
    else:
        print("[FAIL] Basic functionality: FAILED")
    
    if dvc_framework_ok:
        print("[OK] DVC framework: WORKING")
    else:
        print("[WARN] DVC framework: PARTIAL (some imports may fail)")
    
    # Overall assessment
    overall_ok = (success_count == len(core_packages) and functionality_ok)
    
    if overall_ok:
        print("\nINSTALLATION VERIFICATION SUCCESSFUL!")
        print("\nReady to run DVC experiments!")
        print("\nNext steps:")
        print("  1. Activate environment: conda activate dvc-env")
        print("  2. Run example: python scripts/run_experiment.py drafts/configs/probability_analysis.yaml")
        print("  3. Check docs/user-guide/ for detailed usage")
    else:
        print("\nINSTALLATION VERIFICATION INCOMPLETE")
        print("\nSome components may not work correctly. Common issues:")
        print("  - Missing dependencies: reinstall with conda/pip")
        print("  - PyTorch CUDA issues: try CPU-only version")
        print("  - Import errors: check Python path and working directory")
        print("\nFor help, see docs/setup.md")
    
    return 0 if overall_ok else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
