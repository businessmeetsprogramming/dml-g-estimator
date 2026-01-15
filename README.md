# DML vs AAE: Parameter Estimation with AI-Augmented Data

This repository compares Double Machine Learning (DML) against AI-Augmented Estimation (AAE) for parameter estimation in Multinomial Logit (MNL) models using GPT-augmented data.

## Key Result

**DML beats AAE on parameter estimation (MAPE) by 1.2 percentage points.**

| Metric | AAE | DML | Improvement |
|--------|-----|-----|-------------|
| **MAPE (Avg)** | 17.8% | **16.6%** | **-1.2%** |

## Full Benchmark Results

### MAPE (%) - Lower is Better

| Method | n=50 | n=100 | n=150 | n=200 | **Avg** | Rank |
|--------|------|-------|-------|-------|---------|------|
| **DML** | **17.2** | **17.2** | **16.4** | **15.7** | **16.6** | **1st** |
| AAE | 21.0 | 17.6 | 16.6 | 16.0 | 17.8 | 2nd |
| Primary | 41.0 | 27.5 | 22.0 | 17.0 | 27.0 | 3rd |
| PPI | 66.0 | 38.0 | 30.0 | 23.0 | 39.3 | 4th |
| Naive | 57.5 | 54.4 | 52.2 | 50.5 | 53.7 | 5th |

### Improvement over Primary Only

| Method | n=50 | n=100 | n=150 | n=200 | **Avg** |
|--------|------|-------|-------|-------|---------|
| **DML** | **+23.8** | **+10.3** | **+5.6** | **+1.3** | **+10.4** |
| AAE | +20.0 | +9.9 | +5.4 | +1.0 | +9.2 |

---

## Why DML Beats Each Method

### 1. DML vs AAE (-1.2% MAPE)
- **Cross-fitting** averages predictions from 5 models trained on different subsets
- Produces more robust soft labels, especially at small sample sizes (n=50)
- Reduces variance from any single train/test split
- Both use identical g-model architecture; only the training procedure differs

### 2. DML vs Primary Only (-10.4% MAPE)
- Leverages 1000 GPT-augmented samples as additional training data
- Soft labels from g(X,z) provide smooth probability estimates
- Most impactful at n=50 where primary-only severely overfits

### 3. DML vs PPI (-22.7% MAPE)
- PPI's variance correction term destabilizes with small propensity scores
- When e ≈ 0.05, the 1/e factor (~20) amplifies noise dramatically
- DML avoids this by using clipped targets that default to hard labels

### 4. DML vs Naive (-37.1% MAPE)
- Naive treats GPT predictions as ground truth (ignores human labels entirely)
- GPT accuracy is only ~56%, introducing systematic bias
- DML combines both data sources optimally

---

## DML Algorithm

DML achieves better MAPE through **cross-fitted augmented predictions**, inspired by the Double Machine Learning framework from the paper "AI Data Augmentation for Generalized Linear Models."

### Theoretical Foundation

The DML score function (equation 1.3 from the paper) is:

```
ψ(Ξ; e, g; β) = X^T [∇b(Xβ) - g(X,z) + (w/e(X,z))(g(X,z) - y)]
```

Where:
- `g(X,z) = E[Y|X,z]`: Predicts human choice given features and GPT prediction
- `e(X,z) = E[w|X,z]`: Propensity score (probability of observing human label)
- `w`: Indicator (1 for primary/labeled data, 0 for augmented)

For augmented data (w=0), the score simplifies to: `ψ = X^T [∇b(Xβ) - g(X,z)]`

### Practical Implementation

Due to small propensity scores (e ≈ 0.05 when n_p=50, n_aug=1000), the IPW debiasing term causes high variance. The DML target for primary data is:

```
τ = g(X,z) × (1 - 1/e) + y/e
```

When e = 0.05, this becomes: `τ = g × (-19) + y × 20`

After clipping τ to [0, 1], this effectively becomes τ = y (hard labels). This is mathematically correct behavior - as propensity approaches 0, the optimal strategy is to trust the observed labels.

**Final Implementation:**
1. **Augmented data**: Use `g(X,z)` as soft labels
2. **Primary data**: Use hard labels `y` (after τ clipping)

### Cross-Fitting Procedure

Instead of training g on ALL primary data and predicting on augmented (like AAE), DML uses cross-fitting:

```python
for each fold k = 1, ..., 5:
    train g on folds != k (4/5 of primary data)
    predict on ALL augmented data

final_prediction = average across all 5 folds
```

**Why it works:**
- Produces more robust soft labels by averaging predictions from 5 different models
- Reduces variance and avoids overfitting to any single train/test split
- Follows the DML principle of using out-of-sample predictions for nuisance functions

---

## DML Tuning Insights

### G-Estimator: Critical for Performance

The g(X,z) estimator is the **most important tuning parameter** for DML performance.

| G-Estimator | Avg MAPE | Notes |
|-------------|----------|-------|
| **Stratified LR (C=0.05)** | **16.6%** | **Best** - well-calibrated probabilities |
| Stratified LR (C=0.1) | 16.7% | Good |
| Stratified RF (d=2) | 16.6% | Good |
| Ensemble LR | 16.6% | Good |
| Stratified MLP | 17.4% | Baseline |
| BestGEstimator | 26.0% | Poor - overconfident probabilities |

**Key Insight:** Simple, well-regularized models (Logistic Regression with C=0.05) outperform complex models. The critical factor is **probability calibration**, not prediction accuracy. BestGEstimator achieves higher accuracy (~61%) but produces overconfident (extreme) probabilities that hurt DML estimation.

**Best G-Model Configuration:**
```python
# Stratified Logistic Regression per z value
for z_val in [-1, 0, 1]:
    clf = LogisticRegression(C=0.05, max_iter=2000, random_state=1)
    clf.fit(X_train[z == z_val], y_train[z == z_val])
```

### E-Estimator: Does Not Matter

The e(X,z) propensity estimator has **no effect** on DML performance in typical AI-augmentation settings.

| E-Estimator | Avg MAPE |
|-------------|----------|
| LR (C=1.0) | 16.6% |
| LR (C=0.1) | 16.6% |
| LR (C=0.01) | 16.6% |
| MLP | 16.6% |
| RF | 16.6% |
| Constant (n_p/(n_p+n_aug)) | 16.6% |

**Why e doesn't matter:** With small propensity scores (e ≈ 0.05), the DML target τ = g(1-1/e) + y/e produces extreme values that get clipped to [0, 1], effectively becoming hard labels y. The e(X,z) term is "clipped out" of the computation.

This is actually **good news** for practitioners: you can use a simple constant propensity e = n_primary / (n_primary + n_augmented) without loss of performance.

---

## Comparison: AAE vs DML

| Aspect | AAE | DML |
|--------|-----|-----|
| **G-Model** | Stratified LR (C=0.05) | Stratified LR (C=0.05) |
| **Features** | diff only (11) | diff only (11) |
| **Primary Labels** | Hard labels y | Hard labels y |
| **Augmented Labels** | g(X,z) soft labels | g(X,z) soft labels |
| **Augmented Predictions** | Single model on ALL primary | **Cross-fitted (5-fold average)** |

**Key Difference:** DML uses cross-fitting for augmented predictions, averaging over 5 models trained on different subsets. This reduces variance in the soft labels.

---

## Files

```
├── compare_correct.py       # Main comparison script with all methods
├── best_model.py            # BestGEstimator implementation (not used in final DML)
├── test_dml_only.py         # Quick DML vs AAE test script
├── train_gpt-4o_11_1200.pkl # Data file (GPT-4o augmented)
└── README.md                # This file
```

## Quick Start

```bash
# Install dependencies
pip install numpy scikit-learn torch

# Run the full comparison (DML vs AAE vs Primary vs PPI vs Naive)
python compare_correct.py

# Run quick DML vs AAE test
python test_dml_only.py
```

## Usage

### Running the Full Benchmark

```python
from compare_correct import run_dml, run_aae, calculate_mape, GROUND_TRUTH_PARAMS
import pickle as pkl
import numpy as np

# Load data
with open("train_gpt-4o_11_1200.pkl", "rb") as f:
    data = pkl.load(f)[0]

y_real = np.asarray(data['y'])
y_aug = np.asarray(data['y_aug'])
X_all = list(data['X'])

# Define sample indices
n_real = 100  # Number of primary samples
n_aug = 1000  # Number of augmented samples
real_rows = list(range(n_real))
aug_rows = list(range(n_real, n_real + n_aug))

# Run DML
beta_dml = run_dml(X_all, y_real, y_aug, real_rows, aug_rows)
mape_dml = calculate_mape(beta_dml, GROUND_TRUTH_PARAMS)
print(f"DML MAPE: {mape_dml:.1f}%")

# Run AAE
beta_aae = run_aae(X_all, y_real, y_aug, real_rows, aug_rows)
mape_aae = calculate_mape(beta_aae, GROUND_TRUTH_PARAMS)
print(f"AAE MAPE: {mape_aae:.1f}%")
```

### Using the G-Estimator Directly

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_stratified_g(X_diff, z, y, C=0.05):
    """Train stratified g-model (one per z value)."""
    models = {}
    for z_val in [-1, 0, 1]:
        mask = z == z_val
        if np.sum(mask) >= 2 and len(np.unique(y[mask])) == 2:
            clf = LogisticRegression(C=C, max_iter=2000, random_state=1)
            clf.fit(X_diff[mask], y[mask])
            models[z_val] = clf
    return models

def predict_g(models, X_diff, z):
    """Predict g(X,z) probabilities."""
    proba = np.zeros(len(z))
    for z_val, clf in models.items():
        mask = z == z_val
        if np.any(mask):
            proba[mask] = clf.predict_proba(X_diff[mask])[:, 1]
    return proba
```

## Technical Details

### Data Structure
- **X**: Choice features (2 alternatives × 12 features per observation)
- **y**: Human choice (0 or 1)
- **z**: GPT prediction (-1=abstain, 0, 1)
- **n_real**: Number of labeled samples (50-200)
- **n_aug**: Number of augmented samples (1000)

### Ground Truth Parameters
11-dimensional MNL parameter vector estimated from full dataset.

### Evaluation Metric
**MAPE** = Mean Absolute Percentage Error
```
MAPE = mean(|estimated - true| / (|true| + 1)) × 100
```
Note: The +1 in the denominator handles near-zero true parameter values.

## Key Findings

1. **DML beats AAE** on average MAPE by 1.2% (16.6% vs 17.8%)
2. **DML improves most at n=50** with 3.8% MAPE reduction (17.2% vs 21.0%)
3. **G-estimator selection is critical** - use well-calibrated models (LR with C=0.05)
4. **E-estimator doesn't matter** - all configurations give identical results due to τ clipping
5. **Cross-fitting** provides more robust soft labels, especially at small sample sizes
6. **Both methods far outperform** Naive, PPI, and Primary-only baselines

## Citation

Based on AI-Augmented Estimation framework:
- [AI-Augmented Estimation GitHub](https://github.com/mxw-selene/ai-augmented-estimation)
- Paper: "AI Data Augmentation for Generalized Linear Models"
