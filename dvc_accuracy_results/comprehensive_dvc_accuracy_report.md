# Comprehensive DVC Accuracy Test Report

Generated on: 2025-05-28 14:14:05

## Executive Summary

- Total scenarios tested: 1
- PyTorch successful runs: 0
- TensorFlow successful runs: 2
- Total successful runs: 2

## Overall Performance

- **Average Correlation MAE**: 0.1681
- **Average Structure Recovery**: 0.9859
- **Average Entropy Relative Error**: 0.0269
- **Average Fit Time**: 1.360s
- **Average Sample Time**: 0.048s

## Best Results

**Best Correlation Accuracy**: TensorFlow c-vine on AR1_4D (MAE: 0.1648)

**Best Structure Recovery**: TensorFlow d-vine on AR1_4D (Recovery: 0.9946)

## Detailed Results by Scenario

### AR1_4D - TensorFlow c-vine

**Data**: 800 samples, 4 dimensions, ar1 correlation

**Performance**:
- Fit time: 2.591s
- Sample time: 0.054s

**Accuracy**:
- Correlation MAE: 0.1648
- Structure Recovery: 0.9772
- Entropy Relative Error: 0.0224
- Max Correlation Difference: 0.2626

---

### AR1_4D - TensorFlow d-vine

**Data**: 800 samples, 4 dimensions, ar1 correlation

**Performance**:
- Fit time: 0.129s
- Sample time: 0.041s

**Accuracy**:
- Correlation MAE: 0.1715
- Structure Recovery: 0.9946
- Entropy Relative Error: 0.0315
- Max Correlation Difference: 0.2700

---

