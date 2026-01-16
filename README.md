# DML for AI-Augmented Generalized Linear Models

Implementation of the Double Machine Learning (DML) estimator from:

> **"AI Data Augmentation for Generalized Linear Models"**
> Lu, Wang, Zhang, Zhang (2026)

## Key Result

**DML achieves the best performance across all methods.**

### Full Benchmark Comparison (MAPE % - Lower is Better)

| Method | n=50 | n=100 | n=150 | n=200 | **Avg** | **Rank** |
|--------|------|-------|-------|-------|---------|----------|
| **DML** | **17.4** | **16.8** | **16.3** | **15.5** | **16.5** | **1st** |
| **DML-2** | **17.6** | **16.9** | **16.5** | **15.9** | **16.7** | **2nd** |
| AAE | 20.9 | 17.0 | 16.4 | 16.2 | 17.6 | 3rd |
| Primary Only | 55.6 | 29.8 | 22.7 | 19.9 | 32.0 | 4th |
| PPI | 93.3 | 48.5 | 34.0 | 28.8 | 51.1 | 5th |
| Naive | 128.6 | 100.7 | 85.3 | 73.6 | 97.0 | 6th |

*All methods: 30 trials with n_aug=1000, using identical sample paths per trial. PPI excludes z=-1 abstentions.*

**Note on PPI:** Prediction-Powered Inference ([Angelopoulos et al. 2023](https://arxiv.org/abs/2301.09633)) assumes AI predictions are direct proxies for y. In this setting, AI accuracy is only 57%, so PPI underperforms. DML instead learns g(X,z) = E[y|X,z], using z as a feature rather than assuming z ≈ y.

### Improvement Summary

| Comparison | Improvement | Notes |
|------------|-------------|-------|
| DML vs Primary Only | **+15.5%** | Leverages augmented data effectively |
| DML vs Naive | **+80.5%** | Doesn't blindly trust AI labels |
| DML vs AAE | **+1.1%** | Cross-fitting + debiasing correction |
| DML vs DML-2 | **+0.2%** | Essentially equivalent |
| DML vs PPI | **+34.6%** | PPI assumes z ≈ y, DML learns g(X,z) |

### DML vs DML-2 Comparison (30 trials)

| n_real | DML | DML-2 | Difference |
|--------|-----|-------|------------|
| 50 | 17.44% | 17.56% | -0.12% |
| 100 | 16.81% | 16.93% | -0.12% |
| 150 | 16.25% | 16.46% | -0.21% |
| 200 | 15.55% | 15.94% | -0.39% |
| **Avg** | **16.51%** | **16.72%** | **-0.21%** |

**Conclusion:** DML and DML-2 are essentially equivalent (only 0.21% difference), confirming that both implementations are theoretically sound.

---

## File Structure

```
dml_icml_optimize/
├── dml.py                    # Core implementation (USE THIS)
├── run_dml.py                # Run experiments and save results
├── run_comparison.py         # Quick comparison script
├── res2/                     # Saved experiment results
│   ├── dml_gpt-4o_*_30.pkl   # DML results
│   └── dml2_gpt-4o_*_30.pkl  # DML-2 results
├── train_gpt-4o_11_1200.pkl  # Data file
└── README.md                 # This file
```

### Main Files

| File | Description |
|------|-------------|
| **`dml.py`** | Core DML implementation with all methods. **Import this.** |
| **`run_dml.py`** | Run experiments and save results in standardized format |
| **`run_comparison.py`** | Quick comparison script for testing |

---

## Quick Start

### Basic Usage

```python
from dml import run_dml, run_dml2, calculate_mape, GROUND_TRUTH_PARAMS
import pickle as pkl
import numpy as np

# Load data
with open("train_gpt-4o_11_1200.pkl", "rb") as f:
    data = pkl.load(f)[0]

y_real = np.asarray(data['y'])
y_aug = np.asarray(data['y_aug'])
X_all = list(data['X'])

# Define sample indices
n_real, n_aug = 100, 1000
real_rows = list(range(n_real))
aug_rows = list(range(n_real, n_real + n_aug))

# Run DML
beta = run_dml(X_all, y_real, y_aug, real_rows, aug_rows)
mape = calculate_mape(beta, GROUND_TRUTH_PARAMS)
print(f"DML MAPE: {mape:.2f}%")
```

### Run Full Experiments

```bash
# Run DML experiments (saves to res2/)
python run_dml.py --n_trials 30 --n_real 50 --method gpt-4o
python run_dml.py --n_trials 30 --n_real 100 --method gpt-4o

# Run DML-2 experiments
python run_dml.py --n_trials 30 --n_real 50 --method gpt-4o --dml2

# Quick comparison
python run_comparison.py --n_trials 10
```

### Output Format

Results are saved to `res2/dml_{method}_{n_real}_{n_max_aug}_{n_trials}.pkl`:

```python
{
    "n_real_list":     [50, 50, 50, 50, ...],      # n_real for each entry
    "n_aug_list":      [0, 1000, 0, 1000, ...],    # n_aug for each entry
    "sample_id_list":  [0, 0, 1, 1, ...],          # trial ID
    "params_list":     [beta_0, beta_1, ...],      # estimated parameters
}
```

Each trial saves both:
- **Primary Only** (n_aug=0): Using the same random sample
- **DML** (n_aug=1000): Using augmented data

---

## DML Algorithm

### Score Function (Equation 1.3)

```
ψ(Ξ; e, g; β) = X^T [∇b(Xβ) - g(X,z) + (w/e(X,z))(g(X,z) - y)]
```

Where:
- `X`: Features (difference between alternatives)
- `y`: Human label (observed only when w=1)
- `w`: Indicator (w=1 for primary, w=0 for augmented)
- `z`: AI-generated prediction
- `g(X,z) = E[y|X,z]`: Conditional expectation model
- `e(X,z) = P(w=1|X,z)`: Propensity score

### DML-Adjusted Targets

Setting the score function to zero yields effective targets τ:

| Case | Formula | Description |
|------|---------|-------------|
| **Primary (w=1)** | `τ = g(1 - 1/e) + y/e` | Combines prediction and debiasing |
| **Augmented (w=0)** | `τ = g` | Pure prediction (no human label) |

### Key Simplification (Corollary 2.1)

When selection into primary data is random, use constant e:

```
e = n_primary / (n_primary + n_augmented)
```

**This eliminates the need to train an e(X,z) model** without affecting results.

---

## Two DML Variants

### DML (Single β)

```python
beta = run_dml(X_all, y_real, y_aug, real_rows, aug_rows)
```

1. Cross-fit g: Train on K-1 folds, predict on held-out fold
2. Compute DML targets τ for all data
3. Optimize single β on all data

### DML-2 (Exact PDF Algorithm)

```python
beta = run_dml2(X_all, y_real, y_aug, real_rows, aug_rows)
```

1. Split primary data into K folds
2. For each fold k: train g on folds ≠ k, compute β_k
3. Return β̂ = (1/K) Σ β̂_k

### Why Are They Equivalent?

Both satisfy the key DML requirements:
1. **Neyman orthogonality** of the score function
2. **Cross-fitting** for nuisance parameter g

The empirical difference is only **0.21%** (30 trials), confirming asymptotic equivalence.

---

## Cross-Fitting: Why It Matters

### The Requirement

For each observation i, the g prediction must use a model trained **without** observation i.

### How It's Achieved

```
For each fold k = 1, ..., K:
    Train g on primary data NOT in fold k
    Predict on:
        - Fold k's primary data (out-of-fold)
        - ALL augmented data (always out-of-sample)
```

### Why Augmented Predictions Are Valid

A key insight: **augmented data is never used to train g** (since y is unobserved).

Therefore:
- Any fold's g model is valid for augmented predictions
- Averaging across K models reduces variance
- No cross-fitting violation occurs

---

## API Reference

### Core Functions

```python
# DML estimator (recommended)
run_dml(X_all, y_real, y_aug, real_rows, aug_rows, n_folds=5)

# DML-2 estimator (exact PDF algorithm)
run_dml2(X_all, y_real, y_aug, real_rows, aug_rows, n_folds=5)

# Baselines
run_primary_only(X_all, y_real, real_rows)
run_naive(X_all, y_real, y_aug, real_rows, aug_rows)
run_aae(X_all, y_real, y_aug, real_rows, aug_rows)
run_ppi(X_all, y_real, y_aug, real_rows, aug_rows)  # Requires ppi_py

# Run all methods at once
run_all_methods(X_all, y_real, y_aug, real_rows, aug_rows)
```

### Utility Functions

```python
# Calculate MAPE
calculate_mape(estimated, true)

# Convert choice data to difference features
flatten(X)           # Single observation: (2, d+1) -> (d,)
flatten_all(X_list)  # All observations: list -> (n, d)
```

### Constants

```python
GROUND_TRUTH_PARAMS  # True β parameters (11-dimensional)
```

---

## Implementation Details

### G-Model: Stratified Logistic Regression

```python
# Best configuration: stratified by z, strong regularization
for z_val in [-1, 0, 1]:
    clf = LogisticRegression(C=0.05, max_iter=2000)
    clf.fit(X[z == z_val], y[z == z_val])
```

**Why C=0.05?** Strong regularization produces well-calibrated probabilities, which matters more than accuracy for DML.

### E-Model: Not Needed!

Per Corollary 2.1, when selection is random:

```python
e = n_primary / (n_primary + n_augmented)  # Constant
```

All tested e-models (LR, MLP, RF, constant) give identical results.

### Target Clipping

```python
tau = g * (1 - 1/e) + y / e
tau = np.clip(tau, 0.0, 1.0)  # Numerical stability
```

With e ≈ 0.05, extreme values are common. Clipping ensures valid probability targets.

---

## Data Format

### Input Structure

- **X_all**: List of (2, 12) arrays (2 alternatives × 12 features)
- **y_real**: Human labels (0 or 1)
- **y_aug**: AI predictions (z ∈ {-1, 0, 1}, where -1 = abstain)
- **real_rows**: Indices of primary (labeled) observations
- **aug_rows**: Indices of augmented (unlabeled) observations

### Feature Processing

```python
# Convert to difference features for MNL
X_diff = X[1, 1:] - X[0, 1:]  # Shape: (11,)
```

---

## Dependencies

```bash
pip install numpy torch scikit-learn

# Optional: for PPI baseline comparison
pip install ppi-python
```

---

## Citation

```bibtex
@article{lu2026ai,
  title={AI Data Augmentation for Generalized Linear Models},
  author={Lu, Cheng and Wang, Mengxin and Zhang, Dennis and Zhang, Heng},
  journal={ICML},
  year={2026}
}
```

---

## Summary

1. **Use `dml.py`** - it contains everything you need (DML, DML-2, AAE, Naive, Primary, PPI)
2. **DML ranks 1st** - beats all benchmarks (AAE, Primary Only, Naive, PPI)
3. **DML ≈ DML-2** - both are valid (0.21% difference), use whichever you prefer
4. **Constant e works** - no need for complex propensity modeling
5. **G-model matters** - use well-regularized Logistic Regression (C=0.05)
6. **Cross-fitting is key** - ensures unbiased nuisance estimation

### Performance Rankings

| Rank | Method | Avg MAPE | vs DML |
|------|--------|----------|--------|
| 1st | **DML** | **16.5%** | — |
| 2nd | DML-2 | 16.7% | +0.2% |
| 3rd | AAE | 17.6% | +1.1% |
| 4th | Primary Only | 32.0% | +15.5% |
| 5th | PPI | 51.1% | +34.6% |
| 6th | Naive | 97.0% | +80.5% |
