# DVC-NF Tests

This directory contains test files for validating the functionality of the DVC-NF framework.

## 🧪 Available Tests

### Core Functionality Tests

**[test_entropy_optimization.py](test_entropy_optimization.py)** - Entropy-based R-vine optimization tests
- Tests the entropy decomposition algorithms
- Validates R-vine structure optimization methods
- Compares performance with classical Kendall's tau approach
- Includes numerical stability and edge case testing

**[lightweight_test.py](lightweight_test.py)** - Quick functionality validation
- Fast integration test for core components
- Basic workflow validation
- Minimal parameter testing for CI/CD pipelines
- Quick smoke tests for development

**[test_script.py](test_script.py)** - Basic integration tests
- Simple end-to-end workflow testing
- Import validation
- Basic data generation and model fitting
- Regression testing for core functionality

## 🚀 Running Tests

### Individual Tests
```bash
# Run specific test
python test_entropy_optimization.py

# Quick validation
python lightweight_test.py

# Basic integration
python test_script.py
```

### Batch Testing
```bash
# Run all tests (if pytest is available)
pytest tests/

# Or run all manually
cd tests
for test in test_*.py; do python $test; done
```

## 📊 Test Coverage

Current test coverage includes:

**Core Components:**
- ✅ TimeBandwidthFlow neural network architecture
- ✅ TimeDependentVineCopula model initialization and fitting
- ✅ TimeDependentDataGenerator synthetic data creation
- ✅ Entropy-based R-vine optimization algorithms

**Integration Tests:**
- ✅ End-to-end workflow validation
- ✅ Package import consistency
- ✅ Basic performance benchmarks
- ⚠️ Extended stress testing (planned)

**Edge Cases:**
- ✅ Small dimension scenarios (2D, 3D)
- ⚠️ High-dimensional testing (>6D, planned)
- ✅ Numerical stability edge cases
- ⚠️ Memory limit testing (planned)

## 🔧 Test Configuration

Tests use standardized parameters for consistency:

```python
# Standard test configuration
TEST_CONFIG = {
    'dim': 3,
    'n_time_steps': 20,
    'n_samples_per_time': 50,
    'num_epochs': 10,
    'hidden_dim': 16,
    'learning_rate': 1e-2
}
```

Small parameters ensure tests run quickly while validating core functionality.

## 📈 Performance Benchmarks

Tests include basic performance monitoring:

- **Training time**: Typical training should complete within 30-60 seconds
- **Memory usage**: Should handle test data without memory errors
- **Numerical stability**: Loss should decrease monotonically in most cases
- **Convergence**: Model should show learning progress within 10 epochs

## 🐛 Test Troubleshooting

**Common test failures:**

1. **Import errors**: 
   - Ensure DVC-NF is in Python path
   - Check TensorFlow installation
   - Verify DVC framework accessibility

2. **Training failures**:
   - Normal for some random seeds
   - Tests include retry logic where appropriate
   - Check GPU/CPU memory availability

3. **Numerical issues**:
   - Expected with very small test parameters
   - Tests validate convergence trends, not exact values
   - Some randomness is expected in flow training

## 🤝 Adding New Tests

To add new tests:

1. Follow naming convention: `test_<component>.py`
2. Use standardized test configuration
3. Include both positive and negative test cases
4. Add performance benchmarks where appropriate
5. Document expected behavior and edge cases
6. Update this README with test description

### Test Template

```python
#!/usr/bin/env python3
"""
Test template for DVC-NF components
"""

import unittest
import numpy as np
from dvc_nf import ComponentToTest

class TestComponent(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            'dim': 3,
            'n_samples': 100,
            # ... other parameters
        }
    
    def test_basic_functionality(self):
        """Test basic component functionality"""
        # Test implementation
        pass
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Edge case testing
        pass

if __name__ == '__main__':
    unittest.main() 