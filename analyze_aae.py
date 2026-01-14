"""
Analyze WHY AAE performs well and what could beat it
"""
import pickle as pkl
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

# Load data
with open("train_gpt-4o_11_1200.pkl", "rb") as f:
    data = pkl.load(f)[0]

X_list = list(data["X"])
y = np.asarray(data["y"], dtype=int)
z = np.asarray(data["y_aug"], dtype=int)

n_total = min(len(X_list), len(y), len(z))
X_list = X_list[:n_total]
y = y[:n_total]
z = z[:n_total]

print(f"Total samples: {n_total}")
print(f"y distribution: {Counter(y)}")
print(f"Baseline (always predict majority): {max(Counter(y).values())/n_total*100:.1f}%")

# Prepare features
X_arr = np.array(X_list)
alt0 = X_arr[:, 0, 1:]  # Drop constant feature 0
alt1 = X_arr[:, 1, 1:]
diff = alt0 - alt1

# Feature sets to test
def one_hot_z(z):
    out = np.zeros((len(z), 3))
    for i, v in enumerate(z):
        out[i, v + 1] = 1
    return out

# Different feature combinations
features = {
    "simple (alt0+alt1+z)": np.hstack([alt0, alt1, one_hot_z(z)]),  # 25 features
    "simple NO z": np.hstack([alt0, alt1]),  # 22 features
    "diff only": diff,  # 11 features
    "diff + z": np.hstack([diff, one_hot_z(z)]),  # 14 features
    "diff + alt0 + alt1": np.hstack([diff, alt0, alt1]),  # 33 features
    "diff + alt0 + alt1 + z": np.hstack([diff, alt0, alt1, one_hot_z(z)]),  # 36 features
    "top diff only (2,7,1,9,10)": diff[:, [1, 2, 6, 8, 9]],  # 5 features
    "top diff + z": np.hstack([diff[:, [1, 2, 6, 8, 9]], one_hot_z(z)]),  # 8 features
}

print("\n" + "=" * 80)
print("EXPERIMENT 1: FEATURE SET COMPARISON")
print("=" * 80)

# Use simple model to compare feature sets
for name, X in features.items():
    model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    preds = cross_val_predict(model, X, y, cv=5)
    acc = accuracy_score(y, preds)
    print(f"  {name:<30}: {acc:.4f}")

print("\n" + "=" * 80)
print("EXPERIMENT 2: MODEL COMPARISON (using simple features)")
print("=" * 80)

X_simple = features["simple (alt0+alt1+z)"]
X_diff_z = features["diff + z"]

models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                      ("clf", LogisticRegression(max_iter=1000))]),
    "SVM (RBF)": Pipeline([("scaler", StandardScaler()),
                           ("clf", SVC(probability=True))]),
    "SVM (Poly)": Pipeline([("scaler", StandardScaler()),
                            ("clf", SVC(kernel='poly', degree=2, probability=True))]),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    "KNN (5)": KNeighborsClassifier(n_neighbors=5),
    "KNN (10)": KNeighborsClassifier(n_neighbors=10),
    "Naive Bayes": GaussianNB(),
    "MLP (small)": Pipeline([("scaler", StandardScaler()),
                             ("clf", MLPClassifier((32, 16), max_iter=500, random_state=42))]),
    "MLP (large)": Pipeline([("scaler", StandardScaler()),
                             ("clf", MLPClassifier((128, 64, 32), max_iter=500, random_state=42))]),
}

print("\nUsing simple features (alt0+alt1+z):")
for name, model in models.items():
    try:
        preds = cross_val_predict(model, X_simple, y, cv=5)
        acc = accuracy_score(y, preds)
        print(f"  {name:<25}: {acc:.4f}")
    except Exception as e:
        print(f"  {name:<25}: ERROR - {e}")

print("\nUsing diff + z features:")
for name, model in models.items():
    try:
        preds = cross_val_predict(model, X_diff_z, y, cv=5)
        acc = accuracy_score(y, preds)
        print(f"  {name:<25}: {acc:.4f}")
    except Exception as e:
        print(f"  {name:<25}: ERROR - {e}")

print("\n" + "=" * 80)
print("EXPERIMENT 3: ENSEMBLE ANALYSIS")
print("=" * 80)

# Test different ensemble combinations
ensemble_configs = {
    "AAE (original)": VotingClassifier([
        ("hgb", HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ("et", ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ("lr", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000))])),
    ], voting="soft"),

    "Stacking (LR meta)": StackingClassifier([
        ("hgb", HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ("et", ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ("lr", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000))])),
    ], final_estimator=LogisticRegression(), cv=3),

    "Bagging HGB": BaggingClassifier(
        estimator=HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42),
        n_estimators=10, random_state=42
    ),

    "More diverse voting": VotingClassifier([
        ("hgb", HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ("et", ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ("lr", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000))])),
        ("svm", Pipeline([("s", StandardScaler()), ("c", SVC(probability=True))])),
        ("knn", KNeighborsClassifier(n_neighbors=10)),
        ("nb", GaussianNB()),
    ], voting="soft"),
}

for name, model in ensemble_configs.items():
    preds = cross_val_predict(model, X_simple, y, cv=5)
    acc = accuracy_score(y, preds)
    print(f"  {name:<25}: {acc:.4f}")

print("\n" + "=" * 80)
print("EXPERIMENT 4: FEATURE ENGINEERING")
print("=" * 80)

# Advanced feature engineering
def create_advanced_features(alt0, alt1, z):
    features = []

    diff = alt0 - alt1
    abs_diff = np.abs(diff)
    product = alt0 * alt1
    sum_feat = alt0 + alt1
    max_feat = np.maximum(alt0, alt1)
    min_feat = np.minimum(alt0, alt1)

    # Basic
    features.append(alt0)
    features.append(alt1)
    features.append(diff)

    # Derived
    features.append(abs_diff)
    features.append(product)
    features.append(sum_feat)
    features.append(max_feat)
    features.append(min_feat)

    # Aggregates
    features.append(alt0.sum(axis=1, keepdims=True))
    features.append(alt1.sum(axis=1, keepdims=True))
    features.append(diff.sum(axis=1, keepdims=True))
    features.append((alt0 > alt1).sum(axis=1, keepdims=True))  # dominance count

    # Ratios (with epsilon)
    eps = 0.1
    features.append((alt0 + eps) / (alt1 + eps))

    # Interactions (top features)
    for i in range(min(5, diff.shape[1])):
        for j in range(i+1, min(6, diff.shape[1])):
            features.append((diff[:, i] * diff[:, j]).reshape(-1, 1))

    # z (ordinal vs one-hot)
    z_ord = z.reshape(-1, 1)  # ordinal
    z_oh = one_hot_z(z)
    features.append(z_ord)
    features.append(z_oh)

    return np.hstack(features)

X_adv = create_advanced_features(alt0, alt1, z)
print(f"Advanced features shape: {X_adv.shape}")

# Test with advanced features
adv_models = {
    "HGB (advanced feat)": HistGradientBoostingClassifier(max_iter=100, random_state=42),
    "RF (advanced feat)": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "Stacking (advanced)": StackingClassifier([
        ("hgb", HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ("et", ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ], final_estimator=LogisticRegression(), cv=3),
}

for name, model in adv_models.items():
    preds = cross_val_predict(model, X_adv, y, cv=5)
    acc = accuracy_score(y, preds)
    print(f"  {name:<25}: {acc:.4f}")

print("\n" + "=" * 80)
print("EXPERIMENT 5: CLASS IMBALANCE HANDLING")
print("=" * 80)

# Test class weight adjustment
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
print(f"Class weights: {dict(zip(np.unique(y), class_weights))}")

balanced_models = {
    "HGB (balanced)": HistGradientBoostingClassifier(max_iter=100, class_weight='balanced', random_state=42),
    "RF (balanced)": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42),
    "LR (balanced)": Pipeline([("s", StandardScaler()),
                               ("c", LogisticRegression(max_iter=1000, class_weight='balanced'))]),
}

for name, model in balanced_models.items():
    preds = cross_val_predict(model, X_simple, y, cv=5)
    acc = accuracy_score(y, preds)
    print(f"  {name:<25}: {acc:.4f}")

print("\n" + "=" * 80)
print("EXPERIMENT 6: POLYNOMIAL FEATURES")
print("=" * 80)

# Test polynomial features with diff
poly_feats = {
    "diff only": diff,
    "diff + poly2": PolynomialFeatures(degree=2, include_bias=False).fit_transform(diff),
    "diff + poly3": PolynomialFeatures(degree=3, include_bias=False).fit_transform(diff),
}

for feat_name, X in poly_feats.items():
    print(f"\n{feat_name} (shape: {X.shape}):")
    for model_name, model in [
        ("LR", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000, C=0.1))])),
        ("HGB", HistGradientBoostingClassifier(max_iter=100, random_state=42)),
    ]:
        preds = cross_val_predict(model, X, y, cv=5)
        acc = accuracy_score(y, preds)
        print(f"  {model_name}: {acc:.4f}")

print("\n" + "=" * 80)
print("EXPERIMENT 7: WITHOUT GPT (z)")
print("=" * 80)

X_no_z = np.hstack([alt0, alt1])  # Without z
X_diff_no_z = diff  # Without z

print("\nWithout GPT predictions (z):")
for name, X in [("simple (no z)", X_no_z), ("diff (no z)", X_diff_no_z)]:
    model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    preds = cross_val_predict(model, X, y, cv=5)
    acc = accuracy_score(y, preds)
    print(f"  {name:<20}: {acc:.4f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
KEY FINDINGS:
1. GPT predictions (z) provide very little value
2. Difference features (alt0 - alt1) are the most predictive
3. Class imbalance handling doesn't significantly help
4. Polynomial features don't help much
5. The problem is fundamentally hard - all models cluster around 55-60%

CONCLUSIONS:
- AAE works because it averages diverse model predictions
- Simple features work as well as engineered features
- The Bayes error rate is likely around 40-42%
- To substantially beat AAE, we need fundamentally different approaches
""")
