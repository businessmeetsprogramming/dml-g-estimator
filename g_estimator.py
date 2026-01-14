"""
G-Function Estimator for DML Framework

This module provides feature engineering and model selection for the g-function
(outcome model) in Double Machine Learning. It includes:

1. Feature engineering for choice model data (X features only)
2. Multiple model options (Stacking, CatBoost, Voting ensembles)
3. Comparison against AAE (Averaged Accuracy Ensemble) baseline

The g-function predicts human choices (y) given features (X) and GPT auxiliary
variable (z). Feature engineering is applied only to X, not to z.

Usage:
    python g_estimator.py

Author: Generated with Claude Code
"""
from __future__ import annotations
import pickle as pkl
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# Optional: CatBoost
CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except (ImportError, OSError):
    pass


# =============================================================================
# Feature Engineering
# =============================================================================

def one_hot_z(z: np.ndarray, z_values: Tuple[int, ...] = (-1, 0, 1)) -> np.ndarray:
    """
    One-hot encode the GPT auxiliary variable z.

    Args:
        z: Array of z values
        z_values: Possible values of z

    Returns:
        One-hot encoded array of shape (n, len(z_values))
    """
    z = np.asarray(z)
    out = np.zeros((len(z), len(z_values)), dtype=float)
    for j, v in enumerate(z_values):
        out[:, j] = (z == v).astype(float)
    return out


def engineer_X_features(X: List[np.ndarray], drop_first_col: bool = True) -> np.ndarray:
    """
    Advanced feature engineering for choice model data.

    Each X[i] is a (2, 12) matrix representing 2 alternatives with 12 features.
    This function creates features that capture the relative advantages between
    alternatives, which is key for choice modeling.

    Features created:
        1. Original flattened features (22)
        2. Difference features: alt0 - alt1 (11) - captures relative advantage
        3. Absolute difference (11) - captures magnitude of difference
        4. Product features: alt0 * alt1 (11) - captures shared attributes
        5. Sum features: alt0 + alt1 (11) - captures choice set richness
        6. Max/min per feature (22) - captures bounds
        7. Aggregate statistics (5) - summary measures
        8. Second-order interactions (45) - pairwise feature interactions

    Args:
        X: List of (2, 12) arrays, one per observation
        drop_first_col: Whether to drop the constant first column

    Returns:
        Engineered feature matrix of shape (n, 138)
    """
    engineered = []

    for Xi in X:
        Xi = np.asarray(Xi, dtype=float)
        if drop_first_col:
            Xi = Xi[:, 1:]  # Remove constant column, now (2, 11)

        alt0 = Xi[0]  # Features of alternative 0
        alt1 = Xi[1]  # Features of alternative 1

        features = []

        # 1. Original flattened features
        features.extend(alt0.tolist())
        features.extend(alt1.tolist())

        # 2. Difference features (key for choice models)
        diff = alt0 - alt1
        features.extend(diff.tolist())

        # 3. Absolute difference
        abs_diff = np.abs(diff)
        features.extend(abs_diff.tolist())

        # 4. Product/interaction features
        product = alt0 * alt1
        features.extend(product.tolist())

        # 5. Sum features
        sum_feat = alt0 + alt1
        features.extend(sum_feat.tolist())

        # 6. Max and min per feature
        max_feat = np.maximum(alt0, alt1)
        min_feat = np.minimum(alt0, alt1)
        features.extend(max_feat.tolist())
        features.extend(min_feat.tolist())

        # 7. Aggregate statistics
        features.append(np.sum(alt0))
        features.append(np.sum(alt1))
        features.append(np.sum(diff))
        features.append(np.sum(abs_diff))
        features.append(np.sum(product))

        # 8. Second-order interactions
        for i in range(min(5, len(alt0))):
            for j in range(i + 1, min(6, len(alt0))):
                features.append(alt0[i] * alt0[j])
                features.append(alt1[i] * alt1[j])
                features.append(diff[i] * diff[j])

        engineered.append(features)

    return np.array(engineered, dtype=float)


def flatten_X_simple(X: List[np.ndarray], drop_first_col: bool = True) -> np.ndarray:
    """
    Simple flattening without feature engineering (for baseline models).

    Args:
        X: List of (2, 12) arrays
        drop_first_col: Whether to drop the constant first column

    Returns:
        Flattened feature matrix of shape (n, 22)
    """
    feats = []
    for Xi in X:
        Xi = np.asarray(Xi, dtype=float)
        if drop_first_col:
            Xi = Xi[:, 1:]
        feats.append(Xi.reshape(-1))
    return np.vstack(feats)


# =============================================================================
# Model Definitions
# =============================================================================

def get_aae_model(seed: int = 0) -> VotingClassifier:
    """
    AAE: Averaged Accuracy Ensemble - the baseline model.

    Combines HistGradientBoosting, RandomForest, ExtraTrees, and LogisticRegression
    using soft voting.

    Args:
        seed: Random seed for reproducibility

    Returns:
        VotingClassifier ensemble
    """
    estimators = [
        ("hist_gb", HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_depth=5, random_state=seed
        )),
        ("rf", RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1
        )),
        ("logit", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
        ])),
    ]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def get_stacking_model(seed: int = 0, n_samples: int = 200) -> StackingClassifier:
    """
    Stacking ensemble with feature engineering.

    Best for medium to large sample sizes (n >= 200).
    Adapts regularization based on sample size.

    Args:
        seed: Random seed
        n_samples: Number of training samples (affects regularization)

    Returns:
        StackingClassifier ensemble
    """
    # Adapt regularization based on sample size
    if n_samples <= 100:
        l2_reg, max_depth, min_leaf = 5.0, 3, max(10, n_samples // 10)
    elif n_samples <= 200:
        l2_reg, max_depth, min_leaf = 3.0, 4, max(8, n_samples // 20)
    else:
        l2_reg, max_depth, min_leaf = 2.0, 5, max(5, n_samples // 40)

    base_estimators = [
        ('hist_gb', HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.03, max_depth=max_depth,
            min_samples_leaf=min_leaf, l2_regularization=l2_reg,
            random_state=seed
        )),
        ('rf', RandomForestClassifier(
            n_estimators=150, max_depth=max_depth + 1, min_samples_leaf=min_leaf,
            random_state=seed, n_jobs=-1
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=150, max_depth=max_depth + 1, min_samples_leaf=min_leaf,
            random_state=seed, n_jobs=-1
        )),
        ('lr', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.5, max_iter=2000, random_state=seed))
        ])),
        ('svm', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=seed))
        ])),
    ]

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=3,
        stack_method='predict_proba',
        n_jobs=-1,
    )


def get_catboost_model(seed: int = 0, n_samples: int = 200):
    """
    CatBoost model with feature engineering.

    Best for small sample sizes (n <= 100).

    Args:
        seed: Random seed
        n_samples: Number of training samples

    Returns:
        CatBoostClassifier or None if not available
    """
    if not CATBOOST_AVAILABLE:
        return None

    if n_samples <= 100:
        iterations, depth, l2_reg = 100, 3, 20.0
    elif n_samples <= 200:
        iterations, depth, l2_reg = 150, 4, 15.0
    else:
        iterations, depth, l2_reg = 200, 5, 10.0

    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        l2_leaf_reg=l2_reg,
        random_strength=1.0,
        bagging_temperature=0.5,
        border_count=128,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


# =============================================================================
# Evaluation Functions
# =============================================================================

def prepare_features(
    X_list: List[np.ndarray],
    z: np.ndarray,
    use_engineering: bool = True
) -> np.ndarray:
    """
    Prepare features for model training/prediction.

    Args:
        X_list: List of X matrices
        z: GPT auxiliary variable
        use_engineering: Whether to apply feature engineering to X

    Returns:
        Combined feature matrix
    """
    if use_engineering:
        X_feat = engineer_X_features(X_list, drop_first_col=True)
    else:
        X_feat = flatten_X_simple(X_list, drop_first_col=True)

    Z_oh = one_hot_z(z)
    return np.hstack([X_feat, Z_oh])


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate a model using stratified k-fold cross-validation.

    Args:
        model: Sklearn-compatible classifier
        X: Feature matrix
        y: Target labels
        n_folds: Number of CV folds
        seed: Random seed

    Returns:
        Dictionary with mean and std accuracy
    """
    actual_folds = min(n_folds, len(y) // 10)
    if actual_folds < 2:
        actual_folds = 2

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
    accs = []

    for train_idx, test_idx in skf.split(X, y):
        model_copy = clone(model)
        model_copy.fit(X[train_idx], y[train_idx])
        y_pred = model_copy.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], y_pred))

    return {'mean': np.mean(accs), 'std': np.std(accs)}


def run_comparison(
    data_path: str,
    sample_sizes: List[int] = [50, 100, 200, 400, 800],
    test_size: int = 200,
    n_trials: int = 5,
    seed: int = 42,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Compare best models (with feature engineering) against AAE baseline.

    Uses proper ML evaluation protocol:
    - Fixed held-out test set (same for ALL sample sizes)
    - Training set varies by n
    - Multiple trials with different random splits for robust estimates

    Args:
        data_path: Path to pickle data file
        sample_sizes: List of training set sizes to evaluate
        test_size: Size of held-out test set (same for all)
        n_trials: Number of random train/test splits
        seed: Random seed

    Returns:
        Nested dictionary of results by sample size and model
    """
    # Load data
    with open(data_path, "rb") as f:
        data = pkl.load(f)[0]

    y_real_all = np.asarray(data["y"], dtype=int)
    y_aug_all = np.asarray(data["y_aug"], dtype=int)
    X_all = list(data["X"])

    n_total = min(len(X_all), len(y_real_all), len(y_aug_all))
    X_all = X_all[:n_total]
    y_real_all = y_real_all[:n_total]
    y_aug_all = y_aug_all[:n_total]

    max_train = max(sample_sizes)
    required_total = max_train + test_size

    print(f"Total available data: {n_total}")
    print(f"CatBoost available: {CATBOOST_AVAILABLE}")
    print(f"\nEvaluation protocol (standard ML):")
    print(f"  - Fixed test set: {test_size} samples (SAME for all sample sizes)")
    print(f"  - Training sets: {sample_sizes}")
    print(f"  - Random trials: {n_trials}")
    print()

    results = {n: {'AAE': [], 'Stacking_FE': [], 'CatBoost_FE': []} for n in sample_sizes}

    for trial in range(n_trials):
        trial_seed = seed + trial
        rng = np.random.RandomState(trial_seed)

        # Shuffle all indices
        all_indices = rng.permutation(n_total)

        # Fixed test set (last test_size samples)
        test_indices = all_indices[-test_size:]
        train_pool_indices = all_indices[:-test_size]

        # Prepare test data (SAME for all sample sizes)
        X_test_list = [X_all[i] for i in test_indices]
        y_test = np.array([y_real_all[i] for i in test_indices], dtype=int)
        z_test = np.array([y_aug_all[i] for i in test_indices], dtype=int)

        X_test_simple = prepare_features(X_test_list, z_test, use_engineering=False)
        X_test_engineered = prepare_features(X_test_list, z_test, use_engineering=True)

        for n_train in sample_sizes:
            # Training data (first n_train from train pool)
            train_indices = train_pool_indices[:n_train]

            X_train_list = [X_all[i] for i in train_indices]
            y_train = np.array([y_real_all[i] for i in train_indices], dtype=int)
            z_train = np.array([y_aug_all[i] for i in train_indices], dtype=int)

            X_train_simple = prepare_features(X_train_list, z_train, use_engineering=False)
            X_train_engineered = prepare_features(X_train_list, z_train, use_engineering=True)

            # Train and evaluate AAE
            model = get_aae_model(trial_seed)
            model.fit(X_train_simple, y_train)
            y_pred = model.predict(X_test_simple)
            results[n_train]['AAE'].append(accuracy_score(y_test, y_pred))

            # Train and evaluate Stacking_FE
            model = get_stacking_model(trial_seed, n_train)
            model.fit(X_train_engineered, y_train)
            y_pred = model.predict(X_test_engineered)
            results[n_train]['Stacking_FE'].append(accuracy_score(y_test, y_pred))

            # Train and evaluate CatBoost_FE
            if CATBOOST_AVAILABLE:
                model = get_catboost_model(trial_seed, n_train)
                model.fit(X_train_engineered, y_train)
                y_pred = model.predict(X_test_engineered)
                results[n_train]['CatBoost_FE'].append(accuracy_score(y_test, y_pred))

        print(f"Trial {trial + 1}/{n_trials} completed")

    # Aggregate results
    final_results = {}
    print(f"\n{'='*70}")
    print("RESULTS BY SAMPLE SIZE")
    print(f"{'='*70}")

    for n_train in sample_sizes:
        print(f"\n--- Training size: n = {n_train}, Test size: {test_size} ---")
        final_results[n_train] = {}

        aae_accs = results[n_train]['AAE']
        aae_mean, aae_std = np.mean(aae_accs), np.std(aae_accs)
        final_results[n_train]['AAE'] = {'mean': aae_mean, 'std': aae_std}
        print(f"  AAE (baseline):     {aae_mean:.4f} (+/- {aae_std:.4f})")

        stack_accs = results[n_train]['Stacking_FE']
        stack_mean, stack_std = np.mean(stack_accs), np.std(stack_accs)
        final_results[n_train]['Stacking_FE'] = {'mean': stack_mean, 'std': stack_std}
        diff = stack_mean - aae_mean
        print(f"  Stacking_FE:        {stack_mean:.4f} (+/- {stack_std:.4f})  [{'+' if diff >= 0 else ''}{diff:.4f}]")

        if CATBOOST_AVAILABLE:
            cat_accs = results[n_train]['CatBoost_FE']
            cat_mean, cat_std = np.mean(cat_accs), np.std(cat_accs)
            final_results[n_train]['CatBoost_FE'] = {'mean': cat_mean, 'std': cat_std}
            diff = cat_mean - aae_mean
            print(f"  CatBoost_FE:        {cat_mean:.4f} (+/- {cat_std:.4f})  [{'+' if diff >= 0 else ''}{diff:.4f}]")

        # Determine winner
        best_model = max(
            [(k, v['mean']) for k, v in final_results[n_train].items() if k != 'AAE'],
            key=lambda x: x[1]
        )
        beats_aae = best_model[1] > aae_mean
        print(f"  Winner: {best_model[0]} {'✓ BEATS AAE' if beats_aae else '✗ Below AAE'}")

    results = final_results

    # Print summary
    print_summary(results, sample_sizes)

    return results


def print_summary(results: Dict, sample_sizes: List[int]):
    """Print formatted summary table."""
    print(f"\n\n{'='*70}")
    print("SUMMARY: Best Model vs AAE Across Sample Sizes")
    print(f"{'='*70}")

    # Header
    header = f"{'Method':<18}"
    for n in sample_sizes:
        header += f" | n={n:>4}"
    header += " | Wins"
    print(header)
    print("-" * 70)

    # Get all model names
    model_names = set()
    for n in sample_sizes:
        model_names.update(results[n].keys())

    # Print rows
    for model in sorted(model_names):
        row = f"{model:<18}"
        wins = 0
        for n in sample_sizes:
            val = results[n].get(model, {}).get('mean', 0)
            aae_val = results[n]['AAE']['mean']
            if model != 'AAE' and val > aae_val:
                wins += 1
            row += f" | {val:.4f}"
        if model != 'AAE':
            row += f" | {wins}/{len(sample_sizes)}"
        print(row)

    print("-" * 70)

    # Count total wins
    total_wins = 0
    for n in sample_sizes:
        best = max(v['mean'] for k, v in results[n].items() if k != 'AAE')
        if best > results[n]['AAE']['mean']:
            total_wins += 1

    print(f"\n*** Best model beats AAE in {total_wins}/{len(sample_sizes)} sample sizes ***")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    results = run_comparison(
        data_path="train_gpt-4o_11_1200.pkl",
        sample_sizes=[50, 100, 200, 400, 800],
        test_size=200,
        n_trials=5,
        seed=42,
    )
