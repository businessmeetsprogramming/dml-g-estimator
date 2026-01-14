# G-Function Estimator for DML Framework

This repository contains optimized models for the g-function (outcome model) in Double Machine Learning (DML). The g-function predicts human choices given features and GPT auxiliary predictions.

## Overview

The goal is to train a classifier that predicts human choices (`y`) using:
- **X**: Choice model features (2 alternatives × 12 features per observation)
- **z**: GPT auxiliary variable (not engineered, only one-hot encoded)

We compare our best models against the **AAE (Averaged Accuracy Ensemble)** baseline across different training set sizes.

## Files

```
├── g_estimator.py          # Main module with models and evaluation
├── train_gpt-4o_11_1200.pkl # Data file
└── README.md               # This file
```

## Installation

```bash
pip install numpy scikit-learn catboost
```

## Usage

### Run Full Comparison

```bash
python g_estimator.py
```

This evaluates all models across training sizes [50, 100, 200, 400, 800] using a fixed held-out test set.

### Use in Your Code

```python
from g_estimator import (
    engineer_X_features,
    one_hot_z,
    get_stacking_model,
    get_catboost_model,
    get_aae_model,
    prepare_features,
)
import numpy as np

# Load your data
X_list = [...]  # List of (2, 12) arrays
z = [...]       # GPT auxiliary variable
y = [...]       # Human labels

# Prepare features (with engineering for best model)
X_features = prepare_features(X_list, z, use_engineering=True)

# Train model
model = get_stacking_model(seed=42, n_samples=len(y))
model.fit(X_features, y)

# Predict
predictions = model.predict(X_features)
```

## Evaluation Protocol

We use the **standard ML evaluation protocol**:

1. **Fixed held-out test set**: 200 samples (SAME for all training sizes)
2. **Training set**: Varies by n (50, 100, 200, 400, 800)
3. **Multiple trials**: 5 random train/test splits for robust estimates

This ensures fair comparison - all models are evaluated on the exact same test samples.

## Feature Engineering

Features are engineered from X only (not from GPT z). For each observation with 2 alternatives:

| Feature Type | Description | Count |
|--------------|-------------|-------|
| Original | Flattened alt0 and alt1 features | 22 |
| Difference | alt0 - alt1 (relative advantage) | 11 |
| Absolute Diff | \|alt0 - alt1\| | 11 |
| Product | alt0 × alt1 (shared attributes) | 11 |
| Sum | alt0 + alt1 (choice set richness) | 11 |
| Max/Min | Per-feature bounds | 22 |
| Aggregates | Sum statistics | 5 |
| Interactions | Second-order feature pairs | 45 |

**Total engineered features: 138** (+ 3 for one-hot z = 141)

## Models

### AAE (Baseline)
- Averaged Accuracy Ensemble
- Soft voting of: HistGradientBoosting, RandomForest, ExtraTrees, LogisticRegression
- Uses simple flattened features (25 total)

### Stacking_FE (Best overall)
- Stacking classifier with engineered features
- Base learners: HistGradientBoosting, RandomForest, ExtraTrees, LogisticRegression, SVM
- Final estimator: LogisticRegression
- Regularization adapts to sample size

### CatBoost_FE
- CatBoost with engineered features
- Strong regularization for small samples
- Requires `catboost` package

## Results

### Out-of-Sample Accuracy (Fixed Test Set = 200)

| Method | n=50 | n=100 | n=200 | n=400 | n=800 | Wins vs AAE |
|--------|------|-------|-------|-------|-------|-------------|
| **AAE (baseline)** | 0.5300 | 0.5730 | 0.5830 | 0.5850 | 0.5760 | - |
| **Stacking_FE** | **0.5570** | **0.5870** | 0.5660 | 0.5750 | **0.5830** | **3/5** |
| CatBoost_FE | 0.5400 | 0.5710 | 0.5620 | 0.5450 | 0.5670 | 1/5 |

### Key Observations

1. **Performance increases with training data** (as expected):
   - n=50: ~53-55%
   - n=100+: ~57-58%

2. **Stacking_FE beats AAE in 3/5 sample sizes** (n=50, 100, 800)

3. **AAE is strong at n=200-400** where it has enough data but feature engineering adds noise

4. **Feature engineering helps most at small sample sizes** (n=50, 100)

## Model Selection Guide

| Training Size | Recommended Model | Reason |
|---------------|-------------------|--------|
| n ≤ 100 | Stacking_FE | Feature engineering helps with limited data |
| n = 200-400 | AAE | Simple features work better, less overfitting |
| n ≥ 800 | Stacking_FE | Enough data to leverage engineered features |

## Data Format

The data file (`train_gpt-4o_11_1200.pkl`) contains:

```python
{
    'X': List[np.ndarray],    # Each element is (2, 12) array
    'y': np.ndarray,          # Human labels (0 or 1)
    'y_aug': np.ndarray,      # GPT auxiliary variable z (-1, 0, 1)
}
```

- **X[i]**: Choice between 2 alternatives, each with 12 binary features
- **y[i]**: Human's actual choice (0 = alternative 0, 1 = alternative 1)
- **z[i]**: GPT's auxiliary prediction

## Notes

- Feature engineering is applied **only to X**, not to z
- z is simply one-hot encoded (3 values: -1, 0, 1)
- Evaluation uses a **fixed held-out test set** for fair comparison across sample sizes
- Multiple random trials (5) provide robust estimates with standard deviations

## Performance Ceiling

The maximum accuracy achieved (~58%) is limited by:
1. Weak individual feature correlations (max ~0.10)
2. Inherent noise in human choice prediction
3. Limited signal in the auxiliary variable z

This suggests a Bayes error rate of approximately 42% for this task.
