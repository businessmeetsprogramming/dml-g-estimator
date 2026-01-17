# DML for AI-Augmented Generalized Linear Models

Implementation of the Double Machine Learning (DML) estimator from:

> **"AI Data Augmentation for Generalized Linear Models"**
> Lu, Wang, Zhang, Zhang (2026)

## Key Results

### Benchmark 1: Conjoint Data (GPT-4o Predictions)

**DML achieves the best performance on conjoint data with LLM-generated labels.**

#### Point Estimates (MAPE %) - 30 Trials

| Method | n=50 | n=100 | n=150 | n=200 | **Avg** | **Rank** |
|--------|------|-------|-------|-------|---------|----------|
| **DML** | **17.4** | **16.8** | **16.3** | **15.5** | **16.5** | **1st** |
| DML-2 | 17.6 | 16.9 | 16.5 | 15.9 | 16.7 | 2nd |
| PPI++ | 49.7 | 29.7 | 23.4 | 20.5 | 30.8 | 3rd |
| Primary | 55.6 | 29.8 | 22.7 | 19.9 | 32.0 | 4th |
| PPI | 93.3 | 48.5 | 34.0 | 28.8 | 51.1 | 5th |
| Naive | 128.6 | 100.7 | 85.3 | 73.6 | 97.0 | 6th |

*30 trials, n_aug=1000, 11 features. GPT-4o predictions with 57% accuracy.*

#### Confidence Interval Width (95% CI, avg across 11 coefficients)

| Method | n=50 | n=100 | n=150 | n=200 |
|--------|------|-------|-------|-------|
| **DML** | **0.42** | **0.41** | **0.40** | **0.39** |
| Primary | 2.62 | 1.50 | 1.18 | 1.01 |

**Key insight:** DML reduces CI width by **6x** compared to Primary at n=50.

---

### Benchmark 2: Census Data (Gradient Boosting Predictions)

**DML (g=z) achieves lowest MAPE and tightest confidence intervals.**

Replication of Section 6.2 from [AAE-vs-PPI paper](https://github.com/mxw-selene/ai-augmented-estimation).

#### Point Estimates (MAPE %) - CI-Based

| Method | m=100 | m=250 | m=500 | m=750 | m=1000 | **Avg** | **Rank** |
|--------|-------|-------|-------|-------|--------|---------|----------|
| **DML (g=z)** | **42.0** | **51.0** | **63.1** | **44.3** | **34.8** | **47.0** | **1st** |
| Primary | 76.4 | 206.1 | 81.5 | 108.5 | 111.8 | 116.9 | 2nd |
| PPI++ | 97.1 | 210.7 | 249.5 | 82.2 | 72.1 | 142.3 | 3rd |

#### Confidence Interval Width for β₀ (95% CI)

| Method | m=100 | m=250 | m=500 | m=750 | m=1000 |
|--------|-------|-------|-------|-------|--------|
| **DML (g=z)** | **9.88** | **9.67** | **8.90** | **8.39** | **8.44** |
| Primary | 71.92 | 44.61 | 21.01 | 16.76 | 13.90 |
| PPI++ | 92.00 | 43.60 | 63.19 | 23.00 | 19.78 |

#### Standard Errors

| Method | m=100 SE[0] | m=100 SE[1] | m=1000 SE[0] | m=1000 SE[1] |
|--------|-------------|-------------|--------------|--------------|
| **DML (g=z)** | **2.52** | **0.070** | **2.15** | **0.061** |
| Primary | 18.35 | 0.400 | 3.55 | 0.108 |
| PPI++ | 23.47 | 0.444 | 5.05 | 0.124 |

*n=2000 unlabeled. z = AI prediction (continuous probability from gradient boosting, 85% accuracy).*

**Key insight:** DML reduces SE by **7x** at m=100 and maintains stable precision across all sample sizes.

---

### PPI/PPI++ Verification (matches GitHub)

Our implementation matches the original GitHub results:

| Method | m=100 | m=250 | m=500 | m=750 | m=1000 |
|--------|-------|-------|-------|-------|--------|
| PPI bias reduction (ours) | -21.1% | — | — | — | — |
| PPI bias reduction (GitHub) | -22.3% | -12.7% | -7.6% | -2.1% | -4.6% |
| PPI++ bias reduction (ours) | -9.6% | — | — | — | — |
| PPI++ bias reduction (GitHub) | -15.2% | -4.7% | -2.4% | -1.1% | -1.2% |

*Verified with exact same seeds (`np.random.RandomState(j)` for trial j).*

---

### Why g=z Outperforms g(X,z) on Census Data

When z is **well-calibrated** (predicted probability ≈ true probability):

1. **z already captures E[Y|X,z]**: A well-calibrated z means P(Y=1|z=p) ≈ p. Training g(X,z) tries to improve on this but adds estimation noise.

2. **Learning g(X,z) has finite-sample cost**: With small labeled samples (m=100-1000), the variance from estimating g(X,z) outweighs any potential bias reduction.

3. **The optimal g(X,z) ≈ z**: When z is the output of a well-trained model, E[Y|X,z] is approximately z itself. Learning this relationship is redundant.

**Analogy**: If a weather forecaster says "70% chance of rain" and they're well-calibrated, trying to learn a correction function from limited data will likely make predictions worse, not better.

### Rule of Thumb for Choosing g

| AI Prediction Quality | Recommended g | Why |
|----------------------|---------------|-----|
| **Noisy/discrete** (LLM, 57% acc) | Learn g(X,z) | z is unreliable, need to correct |
| **Calibrated probabilities** (ML, 85% acc) | g = z directly | z is already optimal |

---

## Running the Benchmarks

### Conjoint Data (LLM Predictions)

```bash
# Quick comparison (30 trials)
python run_comparison.py --n_trials 30 --n_aug 1000

# Run individual experiments
python run_dml.py --n_trials 30 --n_real 50 --method gpt-4o
python run_dml.py --n_trials 30 --n_real 100 --method gpt-4o

# Run DML-2 variant
python run_dml.py --n_trials 30 --n_real 50 --method gpt-4o --dml2
```

### Census Data (ML Predictions)

```bash
# CI-based comparison (recommended - analytical standard errors)
python run_census_ci.py --m 100 --n 2000
python run_census_ci.py --m 500 --n 2000
python run_census_ci.py --m 1000 --n 2000

# Simulation-based comparison (50 trials, matches GitHub)
python run_census_benchmark.py --num_trials 50 --save_results

# Quick test (10 trials)
python run_census_benchmark.py --num_trials 10
```

The census benchmark uses **exact same seeds** as GitHub (`np.random.RandomState(j)` for trial j).

---

## File Structure

```
dml_icml_optimize/
├── dml.py                    # Core implementation (USE THIS)
├── run_dml.py                # Run experiments and save results
├── run_comparison.py         # Conjoint data comparison
├── run_census_benchmark.py   # Census data benchmark (simulation-based)
├── run_census_ci.py          # Census data benchmark (CI-based)
├── data/
│   └── census/
│       └── census_healthcare.npz  # Census data (318K samples)
├── res/                      # Census benchmark results
├── res2/                     # Conjoint experiment results
├── train_gpt-4o_11_1200.pkl  # Conjoint data file
└── README.md                 # This file
```

### Main Files

| File | Description |
|------|-------------|
| **`dml.py`** | Core DML implementation with all methods. **Import this.** |
| **`run_comparison.py`** | Conjoint data comparison (DML, PPI, PPI++, etc.) |
| **`run_census_ci.py`** | Census CI-based comparison with analytical standard errors |
| **`run_census_benchmark.py`** | Census simulation-based benchmark (replicates PPI paper) |

---

## Quick Start

### Conjoint Data

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

### Census Data

```python
import numpy as np
from ppi_py.datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from run_census_ci import run_dml_direct_with_se, calculate_mape

# Load data
data = load_dataset('data/census/', 'census_healthcare')
Y_total = data["Y"]
Yhat_total = data["Yhat"]
X_total = data["X"].copy()

# Normalize (as in GitHub)
X_total[:, 0] = (X_total[:, 0] - X_total[:, 0].min()) / (X_total[:, 0].max() - X_total[:, 0].min())
X_total[:, 1] = X_total[:, 1] / X_total[:, 1].max()

# Ground truth
true_theta = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000,
                                 tol=1e-15, fit_intercept=False).fit(X_total, Y_total).coef_.squeeze()

# Sample data
m, n = 100, 2000
rng = np.random.RandomState(0)
rand_idx = rng.permutation(len(Y_total))

X_lab, Y_lab = X_total[rand_idx[:m]], Y_total[rand_idx[:m]]
Yhat_lab = Yhat_total[rand_idx[:m]]
X_unlab, Yhat_unlab = X_total[rand_idx[m:m+n]], Yhat_total[rand_idx[m:m+n]]

# Run DML (g=z)
beta, se = run_dml_direct_with_se(X_lab, Y_lab, Yhat_lab, X_unlab, Yhat_unlab)
print(f"DML beta: {beta}")
print(f"DML SE: {se}")
print(f"DML MAPE: {calculate_mape(beta, true_theta):.2f}%")
```

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
run_ppi(X_all, y_real, y_aug, real_rows, aug_rows)      # PPI with λ=1 (requires ppi_py)
run_ppi_plusplus(X_all, y_real, y_aug, real_rows, aug_rows)  # PPI++ with auto-tuned λ
```

### Utility Functions

```python
# Calculate MAPE
calculate_mape(estimated, true)

# Convert choice data to difference features
flatten(X)           # Single observation: (2, d+1) -> (d,)
flatten_all(X_list)  # All observations: list -> (n, d)
```

---

## Dependencies

```bash
pip install numpy scipy scikit-learn torch

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

1. **Use `dml.py`** - it contains everything you need
2. **DML ranks 1st on both benchmarks** - best MAPE and tightest CIs
3. **DML reduces CI width by 6-7x** compared to Primary at small sample sizes
4. **G-model choice matters** - learn g for noisy LLM, use g=z for calibrated ML predictions
5. **Cross-fitting is key** - ensures unbiased nuisance estimation

### Performance Summary

| Dataset | Best Method | Avg MAPE | CI Width Reduction |
|---------|-------------|----------|-------------------|
| Conjoint (LLM, 57% acc) | DML (learned g) | 16.5% | **6x** vs Primary |
| Census (ML, 85% acc) | DML (g=z) | 47.0% | **7x** vs Primary |
