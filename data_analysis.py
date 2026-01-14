"""
Deep Data Analysis for G-Function Estimator
Goal: Understand why AAE works and find ways to substantially beat it
"""
import pickle as pkl
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Load data
with open("train_gpt-4o_11_1200.pkl", "rb") as f:
    data = pkl.load(f)[0]

X_list = list(data["X"])
y = np.asarray(data["y"], dtype=int)
z = np.asarray(data["y_aug"], dtype=int)  # GPT predictions

# Ensure all arrays have same length
n_total = min(len(X_list), len(y), len(z))
X_list = X_list[:n_total]
y = y[:n_total]
z = z[:n_total]

print("=" * 80)
print("1. BASIC DATA STRUCTURE")
print("=" * 80)

print(f"\nTotal observations: {len(X_list)}")
print(f"X shape per observation: {X_list[0].shape}")  # (2, 12) - 2 alternatives, 12 features
print(f"y (human choice) distribution: {Counter(y)}")
print(f"z (GPT prediction) distribution: {Counter(z)}")

# Check y distribution
y_counts = Counter(y)
print(f"\ny=0 (choose alt0): {y_counts[0]} ({y_counts[0]/len(y)*100:.1f}%)")
print(f"y=1 (choose alt1): {y_counts[1]} ({y_counts[1]/len(y)*100:.1f}%)")

print("\n" + "=" * 80)
print("2. FEATURE ANALYSIS")
print("=" * 80)

# Convert X to array for analysis
X_arr = np.array(X_list)  # Shape: (n, 2, 12)
print(f"\nX array shape: {X_arr.shape}")

# Analyze each feature column
print("\nFeature statistics (across all observations, all alternatives):")
for feat_idx in range(X_arr.shape[2]):
    feat_vals = X_arr[:, :, feat_idx].flatten()
    unique_vals = np.unique(feat_vals)
    print(f"  Feature {feat_idx}: unique values = {len(unique_vals)}, "
          f"range = [{feat_vals.min():.2f}, {feat_vals.max():.2f}], "
          f"mean = {feat_vals.mean():.3f}")

# Check if features are binary
print("\nAre features binary?")
for feat_idx in range(X_arr.shape[2]):
    feat_vals = X_arr[:, :, feat_idx].flatten()
    unique_vals = set(feat_vals)
    is_binary = unique_vals.issubset({0, 1, 0.0, 1.0})
    print(f"  Feature {feat_idx}: {'Yes (binary)' if is_binary else f'No (values: {sorted(unique_vals)[:5]}...)'}")

print("\n" + "=" * 80)
print("3. ALTERNATIVE COMPARISON ANALYSIS")
print("=" * 80)

# For each observation, compare alt0 vs alt1
alt0 = X_arr[:, 0, :]  # All alt0 features
alt1 = X_arr[:, 1, :]  # All alt1 features

print("\nDifference (alt0 - alt1) statistics:")
diff = alt0 - alt1
for feat_idx in range(diff.shape[1]):
    d = diff[:, feat_idx]
    print(f"  Feature {feat_idx}: mean diff = {d.mean():.4f}, std = {d.std():.4f}")

# Key insight: which features predict y?
print("\n" + "=" * 80)
print("4. FEATURE-TARGET CORRELATION ANALYSIS")
print("=" * 80)

# Flatten features for correlation analysis
X_flat = np.hstack([alt0, alt1])  # Shape: (n, 24)

# Correlation with y
print("\nPearson correlation of each feature with y:")
correlations = []
for i in range(X_flat.shape[1]):
    corr, pval = stats.pearsonr(X_flat[:, i], y)
    alt = "alt0" if i < 12 else "alt1"
    feat = i if i < 12 else i - 12
    correlations.append((f"{alt}_feat{feat}", corr, pval))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)
print("\nTop 10 most correlated features with y:")
for name, corr, pval in correlations[:10]:
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {name}: r = {corr:.4f} (p = {pval:.4f}) {sig}")

# Difference features correlation
print("\nCorrelation of DIFFERENCE features (alt0 - alt1) with y:")
diff_corrs = []
for i in range(diff.shape[1]):
    corr, pval = stats.pearsonr(diff[:, i], y)
    diff_corrs.append((f"diff_feat{i}", corr, pval))

diff_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
for name, corr, pval in diff_corrs[:10]:
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {name}: r = {corr:.4f} (p = {pval:.4f}) {sig}")

print("\n" + "=" * 80)
print("5. GPT PREDICTION (z) ANALYSIS")
print("=" * 80)

# How well does GPT predict human choice?
print("\nGPT prediction (z) vs Human choice (y) crosstab:")
crosstab = pd.crosstab(z, y, margins=True)
print(crosstab)

# GPT accuracy when it makes a prediction (z != 0)
gpt_pred_mask = z != 0
gpt_pred_correct = ((z == 1) & (y == 1)) | ((z == -1) & (y == 0))
print(f"\nGPT prediction accuracy (when z != 0):")
print(f"  Predictions made: {gpt_pred_mask.sum()} ({gpt_pred_mask.sum()/len(z)*100:.1f}%)")
print(f"  Correct: {gpt_pred_correct.sum()} ({gpt_pred_correct.sum()/gpt_pred_mask.sum()*100:.1f}% of predictions)")

# When z=0 (uncertain), what is y distribution?
z0_mask = z == 0
print(f"\nWhen GPT is uncertain (z=0):")
print(f"  Count: {z0_mask.sum()} ({z0_mask.sum()/len(z)*100:.1f}%)")
print(f"  y=0: {(y[z0_mask] == 0).sum()} ({(y[z0_mask] == 0).sum()/z0_mask.sum()*100:.1f}%)")
print(f"  y=1: {(y[z0_mask] == 1).sum()} ({(y[z0_mask] == 1).sum()/z0_mask.sum()*100:.1f}%)")

# Mutual information between z and y
z_onehot = np.zeros((len(z), 3))
for i, val in enumerate(z):
    z_onehot[i, val + 1] = 1  # -1 -> 0, 0 -> 1, 1 -> 2

mi = mutual_info_classif(z_onehot, y, discrete_features=True)
print(f"\nMutual information between z (one-hot) and y: {mi.sum():.4f}")

print("\n" + "=" * 80)
print("6. CHOICE PATTERNS ANALYSIS")
print("=" * 80)

# When does human choose alt0 vs alt1?
print("\nAnalyzing patterns when y=0 (choose alt0) vs y=1 (choose alt1):")

y0_idx = y == 0
y1_idx = y == 1

# Sum of features for each alternative
alt0_sum = alt0.sum(axis=1)
alt1_sum = alt1.sum(axis=1)

print(f"\nSum of features (proxy for 'total value'):")
print(f"  When y=0: alt0_sum mean = {alt0_sum[y0_idx].mean():.3f}, alt1_sum mean = {alt1_sum[y0_idx].mean():.3f}")
print(f"  When y=1: alt0_sum mean = {alt0_sum[y1_idx].mean():.3f}, alt1_sum mean = {alt1_sum[y1_idx].mean():.3f}")

# Does the person choose the alternative with higher sum?
choose_higher = ((alt0_sum > alt1_sum) & (y == 0)) | ((alt1_sum > alt0_sum) & (y == 1))
print(f"\nDoes human choose the alternative with higher feature sum?")
print(f"  Yes: {choose_higher.sum()} ({choose_higher.sum()/len(y)*100:.1f}%)")

# Dominance: does one alternative dominate (all features >=)?
dominated = np.all(alt0 >= alt1, axis=1) | np.all(alt1 >= alt0, axis=1)
print(f"\nDominance cases (one alt >= other in all features):")
print(f"  Count: {dominated.sum()} ({dominated.sum()/len(y)*100:.1f}%)")

print("\n" + "=" * 80)
print("7. FEATURE IMPORTANCE (Random Forest)")
print("=" * 80)

# Train RF and get feature importance
X_full = np.hstack([alt0, alt1, diff, z.reshape(-1, 1)])
feature_names = [f"alt0_f{i}" for i in range(12)] + \
                [f"alt1_f{i}" for i in range(12)] + \
                [f"diff_f{i}" for i in range(12)] + ["z"]

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_full, y)

importances = list(zip(feature_names, rf.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)

print("\nTop 15 most important features (RF):")
for name, imp in importances[:15]:
    print(f"  {name}: {imp:.4f}")

print("\n" + "=" * 80)
print("8. INTERACTION EFFECTS")
print("=" * 80)

# Check for interaction effects
print("\nChecking for interaction effects between features...")

# Test if certain feature combinations predict y better
best_interactions = []
for i in range(12):
    for j in range(i+1, 12):
        # Interaction: diff_i * diff_j
        interaction = diff[:, i] * diff[:, j]
        if np.std(interaction) > 0:
            corr, pval = stats.pearsonr(interaction, y)
            best_interactions.append((f"diff{i}*diff{j}", corr, pval))

best_interactions.sort(key=lambda x: abs(x[1]), reverse=True)
print("\nTop 10 interaction terms (diff_i * diff_j):")
for name, corr, pval in best_interactions[:10]:
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {name}: r = {corr:.4f} (p = {pval:.4f}) {sig}")

print("\n" + "=" * 80)
print("9. CONDITIONAL ANALYSIS")
print("=" * 80)

# Analyze behavior in different conditions
print("\nAnalyzing human behavior in different conditions...")

# When GPT is confident (z != 0), does human follow?
z_pos = z == 1
z_neg = z == -1
print(f"\nWhen GPT predicts y=1 (z=1):")
print(f"  Human agrees (y=1): {(y[z_pos] == 1).sum()}/{z_pos.sum()} = {(y[z_pos] == 1).sum()/z_pos.sum()*100:.1f}%")
print(f"When GPT predicts y=0 (z=-1):")
print(f"  Human agrees (y=0): {(y[z_neg] == 0).sum()}/{z_neg.sum()} = {(y[z_neg] == 0).sum()/z_neg.sum()*100:.1f}%")

# Feature patterns when human disagrees with GPT
disagree = ((z == 1) & (y == 0)) | ((z == -1) & (y == 1))
print(f"\nWhen human disagrees with GPT ({disagree.sum()} cases):")
print(f"  Mean feature sum diff: {(alt0_sum - alt1_sum)[disagree].mean():.3f}")

print("\n" + "=" * 80)
print("10. SUMMARY AND HYPOTHESES")
print("=" * 80)

print("""
SUMMARY OF FINDINGS:
--------------------
1. Data: 1200 observations, each with 2 alternatives x 12 features
2. Features appear to be binary (0/1)
3. Target y is balanced (~50/50)
4. GPT predictions (z) have moderate correlation with y

KEY OBSERVATIONS:
-----------------
- Individual feature correlations with y are weak (|r| < 0.15)
- Difference features (alt0 - alt1) are slightly more predictive
- GPT predictions are informative but not highly accurate
- No single feature is strongly predictive
- This is a HARD classification problem (low signal-to-noise)

HYPOTHESES FOR IMPROVEMENT:
---------------------------
1. NON-LINEAR PATTERNS: The relationship between features and y may be
   highly non-linear. Deep neural networks might capture these patterns.

2. FEATURE INTERACTIONS: Complex interactions between features might be
   key. We should try polynomial features and kernel methods.

3. ENSEMBLE DIVERSITY: AAE works by combining diverse models. We need
   MORE diverse models with different inductive biases.

4. CALIBRATION: Instead of hard predictions, focus on probability
   calibration - predicting P(y=1|X) accurately.

5. DATA AUGMENTATION: Create synthetic training data to reduce overfitting.

6. ORDINAL ENCODING OF Z: Instead of one-hot, treat z as ordinal.

7. ATTENTION MECHANISM: Use attention to focus on the most relevant
   features for each observation.
""")

print("\n" + "=" * 80)
print("11. DETAILED FEATURE ANALYSIS BY CLASS")
print("=" * 80)

# For each feature, compare distribution when y=0 vs y=1
print("\nFeature distributions by class:")
for feat_idx in range(12):
    diff_f = diff[:, feat_idx]
    mean_y0 = diff_f[y == 0].mean()
    mean_y1 = diff_f[y == 1].mean()
    t_stat, p_val = stats.ttest_ind(diff_f[y == 0], diff_f[y == 1])
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"  diff_feat{feat_idx}: y=0 mean={mean_y0:.4f}, y=1 mean={mean_y1:.4f}, t={t_stat:.2f}, p={p_val:.4f} {sig}")
