# DML vs AAE: Parameter Estimation with AI-Augmented Data

This repository compares Double Machine Learning (DML) against AI-Augmented Estimation (AAE) for parameter estimation in Multinomial Logit (MNL) models using GPT-augmented data.

## Key Result

**DML beats AAE on parameter estimation while matching g-function accuracy.**

| Metric | AAE | DML | Winner |
|--------|-----|-----|--------|
| **MAPE (Avg)** | 22.9% | **22.4%** | DML |
| **G-Function Accuracy** | 56.2% | 56.2% | Tie |

## Full Benchmark Results

### MAPE (%) - Lower is Better

| Method | n=50 | n=100 | n=150 | n=200 | **Avg** | Rank |
|--------|------|-------|-------|-------|---------|------|
| **DML** | **23.6** | 22.8 | **21.8** | **21.4** | **22.4** | **1st** |
| AAE | 25.7 | **22.7** | **21.8** | **21.4** | 22.9 | 2nd |
| Primary | 51.3 | 34.2 | 23.4 | 22.7 | 32.9 | 3rd |
| PPI | 135.4 | 60.9 | 42.4 | 35.4 | 68.5 | 4th |
| Naive | 77.5 | 73.2 | 70.3 | 65.6 | 71.7 | 5th |

### G-Function Accuracy (%) - Higher is Better

| Method | n=50 | n=100 | n=150 | n=200 | **Avg** |
|--------|------|-------|-------|-------|---------|
| AAE | 57.8 | 55.4 | 56.6 | 55.0 | **56.2** |
| DML | 57.8 | 55.4 | 56.6 | 55.0 | **56.2** |

*Note: G-accuracy measured out-of-sample via 5-fold cross-validation for fair comparison.*

### Improvement over Primary Only

| Method | n=50 | n=100 | n=150 | n=200 | **Avg** |
|--------|------|-------|-------|-------|---------|
| **DML** | **+27.7** | +11.4 | +1.7 | +1.4 | **+10.5** |
| AAE | +25.5 | +11.5 | +1.6 | +1.3 | +10.0 |

## Methods Compared

1. **Primary Only**: Uses only n_real labeled samples (no augmented data)
2. **Naive**: Uses LLM predictions (z) as hard labels for augmented data
3. **AAE**: AI-Augmented Estimation - MLP g(X) stratified by z, soft labels
4. **DML**: Same as AAE + adaptive temperature scaling on soft labels
5. **PPI**: Prediction-Powered Inference

## DML Improvement Over AAE

DML uses the **same g-model architecture** as AAE (MLP stratified by z) but adds **adaptive temperature scaling** to soft labels:

```
Temperature scaling formula:
- T = 2.0 for n <= 75  (reduces variance in very small samples)
- T = 1.3 for n <= 125
- T = 1.0 for n > 125  (no scaling for larger samples)
```

This makes soft labels more conservative when the g-model is likely overconfident (small training samples), reducing estimation variance.

## Files

```
├── compare_correct.py       # Main comparison script
├── best_model.py            # BestGEstimator implementation
├── train_gpt-4o_11_1200.pkl # Data file (GPT-4o augmented)
└── README.md                # This file
```

## Quick Start

```bash
# Run the full comparison
python compare_correct.py
```

## Technical Details

### Data Structure
- **X**: Choice features (2 alternatives x 12 features per observation)
- **y**: Human choice (0 or 1)
- **z**: GPT prediction (-1=abstain, 0, 1)
- **n_real**: Number of labeled samples (50-200)
- **n_aug**: Number of augmented samples (1000)

### Ground Truth Parameters
11-dimensional MNL parameter vector estimated from full dataset.

### Evaluation Metric
**MAPE** = Mean Absolute Percentage Error
```
MAPE = mean(|estimated - true| / |true + 1|) * 100
```

## Key Findings

1. **DML beats AAE** on average MAPE (22.4% vs 22.9%)
2. **DML excels at small samples** (n=50): 2.1% improvement over AAE
3. **Both methods tie** on g-function accuracy (56.2% each)
4. **Both far outperform** Naive, PPI, and Primary-only baselines
5. **PPI and Naive hurt performance** vs Primary-only at all sample sizes

## Installation

```bash
pip install numpy scikit-learn torch
```

## Citation

Based on AI-Augmented Estimation framework:
- [AI-Augmented Estimation GitHub](https://github.com/mxw-selene/ai-augmented-estimation)
