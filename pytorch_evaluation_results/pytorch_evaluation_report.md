# PyTorch DVC Implementation Comprehensive Evaluation Report

Generated on: 2025-05-28 12:28:48

## Executive Summary

- Total tests conducted: 14
- Successful executions: 14/14 (100.0%)
- Parametric method success: 7
- Non-parametric method success: 7

### Performance Highlights

- Average fit time: 10.097 seconds
- Average sample time: 0.002 seconds
- Average correlation recovery: 0.2005
- Average correlation MAE: 0.2871

## Detailed Test Results

### Small_Weak - Parametric

**Configuration:** 500 samples, 3 dimensions, weak correlation, normal marginals

**Execution:** ✓ Success
- Fit time: 0.160s
- Sample time: 0.001s

**Correlation Recovery:**
- MAE: 0.2191
- RMSE: 0.3433
- Max absolute difference: 0.6992
- Pearson correlation of correlations: 0.5179
- Spearman correlation of correlations: 0.8660

**Marginal Distribution Quality:**
- Average KS statistic: 0.5227
- Average Wasserstein distance: 0.7331

---

### Small_Weak - Non-parametric

**Configuration:** 500 samples, 3 dimensions, weak correlation, normal marginals

**Execution:** ✓ Success
- Fit time: 1.105s
- Sample time: 0.001s

**Correlation Recovery:**
- MAE: 0.5103
- RMSE: 0.6252
- Max absolute difference: 0.7968
- Pearson correlation of correlations: -0.5091
- Spearman correlation of correlations: -0.8660

**Marginal Distribution Quality:**
- Average KS statistic: 0.5227
- Average Wasserstein distance: 0.7265

---

### Medium_Moderate - Parametric

**Configuration:** 1000 samples, 4 dimensions, moderate correlation, normal marginals

**Execution:** ✓ Success
- Fit time: 0.249s
- Sample time: 0.003s

**Correlation Recovery:**
- MAE: 0.1598
- RMSE: 0.2093
- Max absolute difference: 0.3594
- Pearson correlation of correlations: 0.6381
- Spearman correlation of correlations: 0.7715

**Marginal Distribution Quality:**
- Average KS statistic: 0.5018
- Average Wasserstein distance: 0.7163

---

### Medium_Moderate - Non-parametric

**Configuration:** 1000 samples, 4 dimensions, moderate correlation, normal marginals

**Execution:** ✓ Success
- Fit time: 9.223s
- Sample time: 0.001s

**Correlation Recovery:**
- MAE: 0.4236
- RMSE: 0.4920
- Max absolute difference: 0.6703
- Pearson correlation of correlations: -0.1350
- Spearman correlation of correlations: -0.0617

**Marginal Distribution Quality:**
- Average KS statistic: 0.5037
- Average Wasserstein distance: 0.7115

---

### Large_Strong - Parametric

**Configuration:** 1500 samples, 3 dimensions, strong correlation, normal marginals

**Execution:** ✓ Success
- Fit time: 0.149s
- Sample time: 0.002s

**Correlation Recovery:**
- MAE: 0.0670
- RMSE: 0.0895
- Max absolute difference: 0.1339
- Pearson correlation of correlations: 0.5747
- Spearman correlation of correlations: 0.8660

**Marginal Distribution Quality:**
- Average KS statistic: 0.5082
- Average Wasserstein distance: 0.7048

---

### Large_Strong - Non-parametric

**Configuration:** 1500 samples, 3 dimensions, strong correlation, normal marginals

**Execution:** ✓ Success
- Fit time: 15.945s
- Sample time: 0.001s

**Correlation Recovery:**
- MAE: 0.2562
- RMSE: 0.3170
- Max absolute difference: 0.4634
- Pearson correlation of correlations: -0.0372
- Spearman correlation of correlations: 0.0000

**Marginal Distribution Quality:**
- Average KS statistic: 0.5087
- Average Wasserstein distance: 0.7069

---

### Uniform_Marginals - Parametric

**Configuration:** 1000 samples, 4 dimensions, moderate correlation, uniform marginals

**Execution:** ✓ Success
- Fit time: 0.248s
- Sample time: 0.002s

**Correlation Recovery:**
- MAE: 0.1248
- RMSE: 0.1927
- Max absolute difference: 0.3487
- Pearson correlation of correlations: 0.6074
- Spearman correlation of correlations: 0.6172

**Marginal Distribution Quality:**
- Average KS statistic: 0.0245
- Average Wasserstein distance: 0.0088

---

### Uniform_Marginals - Non-parametric

**Configuration:** 1000 samples, 4 dimensions, moderate correlation, uniform marginals

**Execution:** ✓ Success
- Fit time: 7.424s
- Sample time: 0.001s

**Correlation Recovery:**
- MAE: 0.4237
- RMSE: 0.4922
- Max absolute difference: 0.6701
- Pearson correlation of correlations: -0.2014
- Spearman correlation of correlations: -0.2469

**Marginal Distribution Quality:**
- Average KS statistic: 0.0240
- Average Wasserstein distance: 0.0076

---

### Mixed_Marginals - Parametric

**Configuration:** 1000 samples, 4 dimensions, moderate correlation, mixed marginals

**Execution:** ✓ Success
- Fit time: 0.211s
- Sample time: 0.003s

**Correlation Recovery:**
- MAE: 0.1483
- RMSE: 0.2366
- Max absolute difference: 0.4441
- Pearson correlation of correlations: 0.6775
- Spearman correlation of correlations: 0.7715

**Marginal Distribution Quality:**
- Average KS statistic: 0.4725
- Average Wasserstein distance: 0.5461

---

### Mixed_Marginals - Non-parametric

**Configuration:** 1000 samples, 4 dimensions, moderate correlation, mixed marginals

**Execution:** ✓ Success
- Fit time: 83.386s
- Sample time: 0.001s

**Correlation Recovery:**
- MAE: 0.4221
- RMSE: 0.4904
- Max absolute difference: 0.6702
- Pearson correlation of correlations: -0.6917
- Spearman correlation of correlations: -0.6172

**Marginal Distribution Quality:**
- Average KS statistic: 0.4725
- Average Wasserstein distance: 0.5602

---

### High_Dim - Parametric

**Configuration:** 800 samples, 6 dimensions, moderate correlation, normal marginals

**Execution:** ✓ Success
- Fit time: 0.579s
- Sample time: 0.006s

**Correlation Recovery:**
- MAE: 0.2178
- RMSE: 0.2903
- Max absolute difference: 0.4605
- Pearson correlation of correlations: 0.5841
- Spearman correlation of correlations: 0.6326

**Marginal Distribution Quality:**
- Average KS statistic: 0.5010
- Average Wasserstein distance: 0.6921

---

### High_Dim - Non-parametric

**Configuration:** 800 samples, 6 dimensions, moderate correlation, normal marginals

**Execution:** ✓ Success
- Fit time: 9.337s
- Sample time: 0.002s

**Correlation Recovery:**
- MAE: 0.5098
- RMSE: 0.5644
- Max absolute difference: 0.7786
- Pearson correlation of correlations: 0.3399
- Spearman correlation of correlations: 0.4648

**Marginal Distribution Quality:**
- Average KS statistic: 0.5019
- Average Wasserstein distance: 0.6929

---

### Large_Sample - Parametric

**Configuration:** 2000 samples, 4 dimensions, moderate correlation, normal marginals

**Execution:** ✓ Success
- Fit time: 0.375s
- Sample time: 0.003s

**Correlation Recovery:**
- MAE: 0.1125
- RMSE: 0.1648
- Max absolute difference: 0.3325
- Pearson correlation of correlations: 0.6495
- Spearman correlation of correlations: 0.6172

**Marginal Distribution Quality:**
- Average KS statistic: 0.4978
- Average Wasserstein distance: 0.6841

---

### Large_Sample - Non-parametric

**Configuration:** 2000 samples, 4 dimensions, moderate correlation, normal marginals

**Execution:** ✓ Success
- Fit time: 12.962s
- Sample time: 0.001s

**Correlation Recovery:**
- MAE: 0.4239
- RMSE: 0.4923
- Max absolute difference: 0.6705
- Pearson correlation of correlations: -0.2081
- Spearman correlation of correlations: 0.0309

**Marginal Distribution Quality:**
- Average KS statistic: 0.4975
- Average Wasserstein distance: 0.6812

---

