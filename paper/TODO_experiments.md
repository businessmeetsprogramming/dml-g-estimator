# Experimental Improvements Needed (Cannot Address by Writing)

These issues require new experiments or data and should be addressed before final submission.

## 1. Add Standard Errors and Statistical Tests (High Priority)

**Current Problem**: All results are point estimates only. No confidence intervals or significance tests.

**Required Work**:
- Re-run all 30 trials saving per-trial estimates
- Compute standard errors across trials: `SE = std(MAPE) / sqrt(30)`
- Add to Table 2: e.g., "16.5 ± 0.8"
- Run paired t-tests comparing DML vs each baseline
- Report p-values for key comparisons

**Code Location**: `run_dml.py`, `run_comparison.py`

---

## 2. Clarify or Fix WTP Values in Table 1 (Medium Priority)

**Current Problem**: Table 1 shows specific WTP values ($4.2K, $6.1K, etc.) without citing source.

**Options**:
a) If these are from actual Wang et al. (2024) study: add citation
b) If illustrative: mark as "Illustrative values" in caption
c) If need real values: extract from actual conjoint data

**Note**: May need to check `train_gpt-4o_11_1200.pkl` for actual ground truth params and their economic interpretation.

---

## 3. Add Sensitivity Analysis Varying AI Accuracy (Medium Priority)

**Current Problem**: Only tested with 57% AI accuracy. Reviewers will ask about other accuracy levels.

**Required Work**:
- Simulate AI predictions with varying accuracy (40%, 50%, 60%, 70%, 80%)
- Can do this by randomly flipping labels with probability (1-accuracy)
- Plot MAPE vs AI accuracy for DML vs PPI
- This would powerfully show DML's robustness and PPI's sensitivity

**Implementation**:
```python
def simulate_ai_accuracy(y_true, target_accuracy):
    """Simulate AI predictions with given accuracy"""
    n = len(y_true)
    correct = np.random.rand(n) < target_accuracy
    y_pred = np.where(correct, y_true, 1 - y_true)
    return y_pred
```

---

## 4. Add PPI-Favorable Experiment (Low Priority)

**Current Problem**: PPI looks terrible (51-394% MAPE). Reviewers may think comparison is unfair.

**Required Work**:
- Find or simulate a setting where AI accuracy is high (>80%)
- Show that PPI works well there, while DML still performs comparably
- This demonstrates we're not cherry-picking scenarios

---

## 5. Additional Datasets (Low Priority, for camera-ready)

**Current Problem**: Only one dataset (sports car conjoint).

**Possible Extensions**:
- COVID vaccine conjoint from Wang et al. (2024)
- Simulated data with known ground truth
- Other preference elicitation tasks

---

## Priority Order for Experiments

1. **Standard errors** - Critical for any stats paper
2. **WTP clarification** - Easy fix, just need to check data
3. **AI accuracy sensitivity** - Compelling figure, moderate effort
4. **PPI-favorable case** - Addresses reviewer concern, moderate effort
5. **New datasets** - Nice-to-have for camera-ready
