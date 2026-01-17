#!/usr/bin/env python3
"""
Census Data Benchmark: DML vs PPI/PPI++
========================================

Replicates Section 6.2 from the AAE-vs-PPI paper using census healthcare data.
Uses exact same seeds as the GitHub implementation for reproducibility.

Data: California census 2019 - predicting health insurance from income
Task: Logistic regression coefficient estimation

Usage:
    python run_census_benchmark.py
    python run_census_benchmark.py --num_trials 50 --save_results
"""

import os
import argparse
import pickle as pkl
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import torch
from torch.optim import Adam
import warnings
warnings.filterwarnings("ignore")

# Optional PPI dependency
try:
    from ppi_py import ppi_logistic_pointestimate, logistic
    PPI_AVAILABLE = True
except ImportError:
    PPI_AVAILABLE = False
    print("Warning: ppi_py not available. PPI/PPI++ will be skipped.")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_mape(estimated, true):
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs(estimated - true) / (np.abs(true) + 1e-10)) * 100


def optimize_logistic_soft_targets(X, tau, n_epochs=20000, lr=1e-2, seed=0):
    """
    Optimize logistic regression with soft targets using MLE loss.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Feature matrix
    tau : ndarray of shape (n,)
        Soft targets (probabilities)

    Returns
    -------
    ndarray of shape (d,)
        Estimated β parameters
    """
    from scipy.optimize import minimize

    # Use scipy L-BFGS for better convergence
    def neg_loglik(beta):
        u = X @ beta
        # Clip for numerical stability
        u = np.clip(u, -500, 500)
        return np.mean(-tau * u + np.log(1 + np.exp(u)))

    def neg_loglik_grad(beta):
        u = X @ beta
        u = np.clip(u, -500, 500)
        p = 1 / (1 + np.exp(-u))
        return X.T @ (p - tau) / len(tau)

    result = minimize(
        neg_loglik,
        np.zeros(X.shape[1]),
        method='L-BFGS-B',
        jac=neg_loglik_grad,
        options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-12}
    )
    return result.x


# =============================================================================
# DML FOR CENSUS DATA (Standard Logistic Regression Format)
# =============================================================================

def train_g_stratified_census(X, z, y, C=0.05):
    """Train stratified g(X,z) = E[y|X,z] models for census data."""
    g_models = {}

    for z_val in np.unique(z):
        mask = (z == z_val)
        n_samples = np.sum(mask)
        n_classes = len(np.unique(y[mask]))

        if n_samples >= 2 and n_classes == 2:
            clf = LogisticRegression(C=C, max_iter=2000, random_state=1)
            clf.fit(X[mask], y[mask])
            g_models[z_val] = clf
        else:
            g_models[z_val] = None

    return g_models


def predict_g_census(X, z, g_models, fallback_prob):
    """Predict g(X,z) = P(y=1|X,z) using stratified models."""
    n = len(z)
    g_proba = np.zeros(n)

    for i in range(n):
        z_i = z[i]
        if z_i in g_models and g_models[z_i] is not None:
            g_proba[i] = g_models[z_i].predict_proba(X[i:i+1])[0, 1]
        else:
            g_proba[i] = fallback_prob

    return g_proba


def run_dml_census(X_labeled, y_labeled, z_labeled, X_unlabeled, z_unlabeled, n_folds=5):
    """
    DML estimator for census data (standard logistic regression format).
    Uses binary z with stratified g-models.
    """
    from sklearn.model_selection import KFold

    n_p = len(y_labeled)
    n_a = len(z_unlabeled)
    n_total = n_p + n_a

    e_constant = n_p / n_total
    class_prior = np.mean(y_labeled)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    g_proba_p = np.zeros(n_p)
    g_proba_a = np.zeros(n_a)
    a_fold_counts = np.zeros(n_a)

    for train_idx, val_idx in kf.split(X_labeled):
        g_models = train_g_stratified_census(
            X_labeled[train_idx],
            z_labeled[train_idx],
            y_labeled[train_idx]
        )

        g_proba_p[val_idx] = predict_g_census(
            X_labeled[val_idx],
            z_labeled[val_idx],
            g_models,
            class_prior
        )

        g_proba_a += predict_g_census(X_unlabeled, z_unlabeled, g_models, class_prior)
        a_fold_counts += 1

    g_proba_a = g_proba_a / a_fold_counts

    tau_p = g_proba_p * (1 - 1/e_constant) + y_labeled / e_constant
    tau_p = np.clip(tau_p, 0.0, 1.0)
    tau_a = g_proba_a

    X_all = np.vstack([X_labeled, X_unlabeled])
    tau_all = np.concatenate([tau_p, tau_a])

    return optimize_logistic_soft_targets(X_all, tau_all)


def run_dml_direct_census(X_labeled, y_labeled, z_labeled, X_unlabeled, z_unlabeled):
    """
    DML with g = z directly (no model training).

    When AI predictions (z) are well-calibrated, z is already E[Y|X,z].
    """
    n_p = len(y_labeled)
    n_a = len(z_unlabeled)
    n_total = n_p + n_a

    e_constant = n_p / n_total

    # Use z directly as g predictions
    g_proba_p = z_labeled
    g_proba_a = z_unlabeled

    # DML-adjusted targets
    tau_p = g_proba_p * (1 - 1/e_constant) + y_labeled / e_constant
    tau_p = np.clip(tau_p, 0.0, 1.0)
    tau_a = g_proba_a

    X_all = np.vstack([X_labeled, X_unlabeled])
    tau_all = np.concatenate([tau_p, tau_a])

    return optimize_logistic_soft_targets(X_all, tau_all)


def run_dml_learned_census(X_labeled, y_labeled, z_labeled, X_unlabeled, z_unlabeled, n_folds=5):
    """
    DML with learned g(X, z) - train a model using both X and continuous z as features.

    This learns g(X, z) = E[Y|X, z] by treating z as a continuous feature
    rather than stratifying by z or using z directly.
    """
    from sklearn.model_selection import KFold

    n_p = len(y_labeled)
    n_a = len(z_unlabeled)
    n_total = n_p + n_a

    e_constant = n_p / n_total

    # Augment X with z as a feature: [X, z]
    X_aug_labeled = np.column_stack([X_labeled, z_labeled])
    X_aug_unlabeled = np.column_stack([X_unlabeled, z_unlabeled])

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    g_proba_p = np.zeros(n_p)
    g_proba_a = np.zeros(n_a)
    a_fold_counts = np.zeros(n_a)

    for train_idx, val_idx in kf.split(X_aug_labeled):
        # Train g(X, z) on augmented features
        X_train = X_aug_labeled[train_idx]
        y_train = y_labeled[train_idx]

        if len(np.unique(y_train)) == 2:
            clf = LogisticRegression(
                penalty=None,
                solver='lbfgs',
                max_iter=10000,
                tol=1e-15,
                fit_intercept=True
            )
            clf.fit(X_train, y_train)

            # Predict on validation fold (out-of-fold)
            g_proba_p[val_idx] = clf.predict_proba(X_aug_labeled[val_idx])[:, 1]

            # Predict on augmented data
            g_proba_a += clf.predict_proba(X_aug_unlabeled)[:, 1]
            a_fold_counts += 1
        else:
            # Fallback to class prior
            class_prior = np.mean(y_train)
            g_proba_p[val_idx] = class_prior
            g_proba_a += class_prior
            a_fold_counts += 1

    g_proba_a = g_proba_a / a_fold_counts

    # DML-adjusted targets
    tau_p = g_proba_p * (1 - 1/e_constant) + y_labeled / e_constant
    tau_p = np.clip(tau_p, 0.0, 1.0)
    tau_a = g_proba_a

    X_all = np.vstack([X_labeled, X_unlabeled])
    tau_all = np.concatenate([tau_p, tau_a])

    return optimize_logistic_soft_targets(X_all, tau_all)


def run_dml_quantile_census(X_labeled, y_labeled, yhat_labeled, X_unlabeled, yhat_unlabeled, n_bins=5, n_folds=5):
    """
    DML-Quantile: Use quantile-based binning of Yhat instead of binary z.

    This preserves more information from continuous predictions while
    still allowing stratified g-model training.
    """
    from sklearn.model_selection import KFold

    n_p = len(y_labeled)
    n_a = len(yhat_unlabeled)
    n_total = n_p + n_a

    e_constant = n_p / n_total
    class_prior = np.mean(y_labeled)

    # Create quantile bins from labeled data
    quantiles = np.percentile(yhat_labeled, np.linspace(0, 100, n_bins + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    z_labeled_binned = np.digitize(yhat_labeled, quantiles) - 1
    z_unlabeled_binned = np.digitize(yhat_unlabeled, quantiles) - 1

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    g_proba_p = np.zeros(n_p)
    g_proba_a = np.zeros(n_a)
    a_fold_counts = np.zeros(n_a)

    for train_idx, val_idx in kf.split(X_labeled):
        # Train stratified g-models on quantile bins
        g_models = {}
        for z_val in range(n_bins):
            mask = (z_labeled_binned[train_idx] == z_val)
            y_subset = y_labeled[train_idx][mask]
            X_subset = X_labeled[train_idx][mask]

            if len(y_subset) >= 2 and len(np.unique(y_subset)) == 2:
                clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000, tol=1e-15)
                clf.fit(X_subset, y_subset)
                g_models[z_val] = clf
            else:
                g_models[z_val] = None

        # Predict on validation fold
        for i in val_idx:
            z_i = z_labeled_binned[i]
            if z_i in g_models and g_models[z_i] is not None:
                g_proba_p[i] = g_models[z_i].predict_proba(X_labeled[i:i+1])[0, 1]
            else:
                g_proba_p[i] = class_prior

        # Predict on augmented data
        for i in range(n_a):
            z_i = z_unlabeled_binned[i]
            if z_i in g_models and g_models[z_i] is not None:
                g_proba_a[i] += g_models[z_i].predict_proba(X_unlabeled[i:i+1])[0, 1]
            else:
                g_proba_a[i] += class_prior
        a_fold_counts += 1

    g_proba_a = g_proba_a / a_fold_counts

    tau_p = g_proba_p * (1 - 1/e_constant) + y_labeled / e_constant
    tau_p = np.clip(tau_p, 0.0, 1.0)
    tau_a = g_proba_a

    X_all = np.vstack([X_labeled, X_unlabeled])
    tau_all = np.concatenate([tau_p, tau_a])

    return optimize_logistic_soft_targets(X_all, tau_all)


def run_dml_noreg_census(X_labeled, y_labeled, z_labeled, X_unlabeled, z_unlabeled, n_folds=5):
    """
    DML-NoReg: Use no regularization in g-model (like AAE does).
    """
    from sklearn.model_selection import KFold

    n_p = len(y_labeled)
    n_a = len(z_unlabeled)
    n_total = n_p + n_a

    e_constant = n_p / n_total
    class_prior = np.mean(y_labeled)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    g_proba_p = np.zeros(n_p)
    g_proba_a = np.zeros(n_a)
    a_fold_counts = np.zeros(n_a)

    for train_idx, val_idx in kf.split(X_labeled):
        # Train with NO regularization (like AAE)
        g_models = {}
        for z_val in np.unique(z_labeled[train_idx]):
            mask = (z_labeled[train_idx] == z_val)
            y_subset = y_labeled[train_idx][mask]
            X_subset = X_labeled[train_idx][mask]

            if len(y_subset) >= 2 and len(np.unique(y_subset)) == 2:
                clf = LogisticRegression(
                    penalty=None,
                    solver='lbfgs',
                    max_iter=10000,
                    tol=1e-15,
                    fit_intercept=False
                )
                clf.fit(X_subset, y_subset)
                g_models[z_val] = clf
            else:
                g_models[z_val] = None

        g_proba_p[val_idx] = predict_g_census(
            X_labeled[val_idx],
            z_labeled[val_idx],
            g_models,
            class_prior
        )

        g_proba_a += predict_g_census(X_unlabeled, z_unlabeled, g_models, class_prior)
        a_fold_counts += 1

    g_proba_a = g_proba_a / a_fold_counts

    tau_p = g_proba_p * (1 - 1/e_constant) + y_labeled / e_constant
    tau_p = np.clip(tau_p, 0.0, 1.0)
    tau_a = g_proba_a

    X_all = np.vstack([X_labeled, X_unlabeled])
    tau_all = np.concatenate([tau_p, tau_a])

    return optimize_logistic_soft_targets(X_all, tau_all)


def run_primary_only_census(X_labeled, y_labeled):
    """Primary-only baseline using standard logistic regression (sklearn)."""
    # Use sklearn for exact solution with hard labels
    clf = LogisticRegression(
        penalty=None,
        solver='lbfgs',
        max_iter=10000,
        tol=1e-15,
        fit_intercept=False
    )
    clf.fit(X_labeled, y_labeled)
    return clf.coef_.squeeze()


def run_naive_census(X_labeled, y_labeled, X_unlabeled, z_unlabeled):
    """Naive baseline: treat AI predictions as hard labels."""
    X_all = np.vstack([X_labeled, X_unlabeled])
    y_all = np.concatenate([y_labeled, z_unlabeled])
    return optimize_logistic_soft_targets(X_all, y_all.astype(float))


def run_aae_census(X_labeled, y_labeled, z_labeled, X_unlabeled, z_unlabeled):
    """
    AAE baseline for census data.
    Uses g(X) models stratified by z to produce soft labels.
    Matches the GitHub implementation exactly.
    """
    class_prior = np.mean(y_labeled)

    # Train g models (LogisticRegression as in GitHub for census)
    g_models = {}
    for z_val in np.unique(z_labeled):
        mask = (z_labeled == z_val)
        if np.sum(mask) >= 2 and len(np.unique(y_labeled[mask])) == 2:
            clf = LogisticRegression(
                penalty=None,
                solver='lbfgs',
                max_iter=10000,
                tol=1e-15,
                fit_intercept=False
            )
            clf.fit(X_labeled[mask], y_labeled[mask])
            g_models[z_val] = clf

    # Primary: hard labels
    tau_p = y_labeled.astype(float)

    # Augmented: soft labels from g
    tau_a = predict_g_census(X_unlabeled, z_unlabeled, g_models, class_prior)

    # Combine
    X_all = np.vstack([X_labeled, X_unlabeled])
    tau_all = np.concatenate([tau_p, tau_a])

    return optimize_logistic_soft_targets(X_all, tau_all, n_epochs=5000, lr=1e-2)


def run_ppi_census(X_labeled, y_labeled, yhat_labeled, X_unlabeled, yhat_unlabeled):
    """PPI with fixed lambda=1."""
    if not PPI_AVAILABLE:
        return None

    optimizer_options = {"ftol": 1e-5, "gtol": 1e-5, "maxls": 10000, "maxiter": 10000}

    try:
        return ppi_logistic_pointestimate(
            X_labeled, y_labeled.astype(float), yhat_labeled.astype(float),
            X_unlabeled, yhat_unlabeled.astype(float),
            lam=1,
            optimizer_options=optimizer_options
        )
    except Exception as e:
        print(f"PPI error: {e}")
        return None


def run_ppi_plusplus_census(X_labeled, y_labeled, yhat_labeled, X_unlabeled, yhat_unlabeled):
    """PPI++ with optimized lambda (auto-tuned)."""
    if not PPI_AVAILABLE:
        return None

    optimizer_options = {"ftol": 1e-5, "gtol": 1e-5, "maxls": 10000, "maxiter": 10000}

    try:
        # No lam parameter = auto-tune (PPI++)
        return ppi_logistic_pointestimate(
            X_labeled, y_labeled.astype(float), yhat_labeled.astype(float),
            X_unlabeled, yhat_unlabeled.astype(float),
            optimizer_options=optimizer_options
        )
    except Exception as e:
        print(f"PPI++ error: {e}")
        return None


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def load_census_data(data_path="data/census/census_healthcare.npz"):
    """Load and preprocess census data exactly as in GitHub."""
    data = np.load(data_path)
    Y_total = data["Y"]
    Yhat_total = data["Yhat"]
    X_total = data["X"].copy()

    # Normalize X exactly as in GitHub
    X_total[:, 0] = (X_total[:, 0] - X_total[:, 0].min()) / (X_total[:, 0].max() - X_total[:, 0].min() + 1e-10)
    X_total[:, 1] = X_total[:, 1] / (X_total[:, 1].max() + 1e-10)

    return Y_total, Yhat_total, X_total


def compute_ground_truth(X_total, Y_total):
    """Compute ground truth β by fitting logistic regression on all data."""
    clf = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=10000,
        tol=1e-15,
        fit_intercept=False
    )
    clf.fit(X_total, Y_total)
    return clf.coef_.squeeze()


def perform_t_test(results_method1, results_method2, true_theta):
    """One-sample t-test comparing MAPE differences between two methods."""
    mape_diffs = []
    for est1, est2 in zip(results_method1, results_method2):
        if est1 is not None and est2 is not None:
            mape1 = np.mean(np.abs((est1 - true_theta) / true_theta))
            mape2 = np.mean(np.abs((est2 - true_theta) / true_theta))
            mape_diffs.append(mape1 - mape2)

    if len(mape_diffs) < 2:
        return None, None, None

    mape_diffs = np.array(mape_diffs)
    t_stat, p_value = stats.ttest_1samp(mape_diffs, 0, alternative='less')
    return t_stat, p_value, np.mean(mape_diffs)


def main():
    parser = argparse.ArgumentParser(description='Census Data Benchmark: DML vs PPI')
    parser.add_argument('--num_trials', type=int, default=50,
                        help='Number of random trials (default: 50)')
    parser.add_argument('--n_unlabeled', type=int, default=2000,
                        help='Number of unlabeled samples (default: 2000)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save results to pickle file')
    parser.add_argument('--data_path', type=str, default='data/census/census_healthcare.npz',
                        help='Path to census data file')
    args = parser.parse_args()

    print("=" * 70)
    print("CENSUS DATA BENCHMARK: DML vs PPI/PPI++")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    Y_total, Yhat_total, X_total = load_census_data(args.data_path)
    n_total = len(Y_total)
    print(f"Total samples: {n_total}")
    print(f"Features: {X_total.shape[1]}")

    # Compute ground truth
    true_theta = compute_ground_truth(X_total, Y_total)
    print(f"Ground truth β: {true_theta}")

    # Binarize Yhat for z (AI predictions)
    Yhat_binary = (Yhat_total > 0.5).astype(int)

    # Experiment configuration (matching GitHub exactly)
    ms = np.array([100, 250, 500, 750, 1000]).astype(int)
    n = args.n_unlabeled
    num_trials = args.num_trials

    print(f"\nExperiment setup:")
    print(f"  - Labeled sample sizes (m): {ms}")
    print(f"  - Unlabeled samples (n): {n}")
    print(f"  - Number of trials: {num_trials}")

    # Methods to compare
    # Note: z = Yhat (AI prediction) - using consistent naming
    methods = ['Primary', 'DML', 'DML-g=z', 'DML-g(X,z)', 'AAE']
    if PPI_AVAILABLE:
        methods.extend(['PPI', 'PPI++'])

    # Storage for results
    results = {m: {method: [] for method in methods} for m in ms}

    # Run experiments
    for m in ms:
        print(f"\n--- m = {m}, n = {n} ---")

        for trial in range(num_trials):
            # EXACT SAME SEED AS GITHUB: np.random.RandomState(trial)
            rng = np.random.RandomState(trial)
            rand_idx = rng.permutation(n_total)

            # Split data
            X_labeled = X_total[rand_idx[:m]]
            Y_labeled = Y_total[rand_idx[:m]]
            Yhat_labeled = Yhat_total[rand_idx[:m]]
            Yhat_labeled_binary = Yhat_binary[rand_idx[:m]]

            X_unlabeled = X_total[rand_idx[m:m+n]]
            Yhat_unlabeled = Yhat_total[rand_idx[m:m+n]]
            Yhat_unlabeled_binary = Yhat_binary[rand_idx[m:m+n]]

            # Run each method
            # z = Yhat (continuous AI prediction)
            z_labeled = Yhat_labeled
            z_unlabeled = Yhat_unlabeled
            z_labeled_binary = Yhat_labeled_binary
            z_unlabeled_binary = Yhat_unlabeled_binary

            for method in methods:
                try:
                    if method == 'Primary':
                        beta = run_primary_only_census(X_labeled, Y_labeled)
                    elif method == 'DML':
                        # Original DML with stratified g by binary z
                        beta = run_dml_census(
                            X_labeled, Y_labeled, z_labeled_binary,
                            X_unlabeled, z_unlabeled_binary
                        )
                    elif method == 'DML-g=z':
                        # DML with g = z directly (no model)
                        beta = run_dml_direct_census(
                            X_labeled, Y_labeled, z_labeled,
                            X_unlabeled, z_unlabeled
                        )
                    elif method == 'DML-g(X,z)':
                        # DML with learned g(X, z) using z as continuous feature
                        beta = run_dml_learned_census(
                            X_labeled, Y_labeled, z_labeled,
                            X_unlabeled, z_unlabeled
                        )
                    elif method == 'AAE':
                        beta = run_aae_census(
                            X_labeled, Y_labeled, z_labeled_binary,
                            X_unlabeled, z_unlabeled_binary
                        )
                    elif method == 'PPI':
                        beta = run_ppi_census(
                            X_labeled, Y_labeled, z_labeled,
                            X_unlabeled, z_unlabeled
                        )
                    elif method == 'PPI++':
                        beta = run_ppi_plusplus_census(
                            X_labeled, Y_labeled, z_labeled,
                            X_unlabeled, z_unlabeled
                        )

                    results[m][method].append(beta)
                except Exception as e:
                    print(f"  Error in {method}, trial {trial}: {e}")
                    results[m][method].append(None)

            if (trial + 1) % 10 == 0:
                print(f"  Completed trial {trial + 1}/{num_trials}")

        # Print intermediate results
        print(f"\n  Results for m={m}:")
        for method in methods:
            valid_results = [r for r in results[m][method] if r is not None]
            if valid_results:
                mapes = [calculate_mape(r, true_theta) for r in valid_results]
                print(f"    {method:<10}: MAPE = {np.mean(mapes):6.2f}% ± {np.std(mapes):5.2f}%")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS: MAPE (%) - Lower is Better")
    print("=" * 70)

    # Header
    print(f"\n{'Method':<12}", end="")
    for m in ms:
        print(f" | m={m:<4}", end="")
    print(" |   Avg")
    print("-" * 75)

    # Results for each method
    avg_mapes = {}
    for method in methods:
        print(f"{method:<12}", end="")
        method_mapes = []
        for m in ms:
            valid_results = [r for r in results[m][method] if r is not None]
            if valid_results:
                mapes = [calculate_mape(r, true_theta) for r in valid_results]
                mean_mape = np.mean(mapes)
                method_mapes.append(mean_mape)
                print(f" | {mean_mape:5.2f}", end="")
            else:
                print(f" |   N/A", end="")

        avg = np.mean(method_mapes) if method_mapes else float('nan')
        avg_mapes[method] = avg
        print(f" | {avg:5.2f}")

    # Bias reduction table (relative to Primary)
    print("\n" + "-" * 75)
    print("BIAS REDUCTION vs Primary (negative = improvement)")
    print("-" * 75)

    print(f"{'Method':<12}", end="")
    for m in ms:
        print(f" | m={m:<4}", end="")
    print("")
    print("-" * 75)

    for method in methods:
        if method == 'Primary':
            continue
        print(f"{method:<12}", end="")
        for m in ms:
            primary_results = [r for r in results[m]['Primary'] if r is not None]
            method_results = [r for r in results[m][method] if r is not None]

            if primary_results and method_results:
                primary_mapes = [calculate_mape(r, true_theta) for r in primary_results]
                method_mapes = [calculate_mape(r, true_theta) for r in method_results]
                diff = np.mean(method_mapes) - np.mean(primary_mapes)
                print(f" | {diff:+5.2f}", end="")
            else:
                print(f" |   N/A", end="")
        print("")

    # Statistical tests: DML vs each method
    print("\n" + "-" * 75)
    print("STATISTICAL TESTS: DML vs Others (one-sample t-test)")
    print("-" * 75)

    for m in ms:
        print(f"\nm = {m}:")
        dml_results = results[m]['DML']

        for method in methods:
            if method == 'DML':
                continue
            method_results = results[m][method]
            t_stat, p_val, mean_diff = perform_t_test(method_results, dml_results, true_theta)

            if t_stat is not None:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"  {method:<10} vs DML: t={t_stat:+6.2f}, p={p_val:.4f} {sig}, mean_diff={mean_diff:+.4f}")

    # Save results
    if args.save_results:
        os.makedirs("res", exist_ok=True)
        res_file = f"res/census_benchmark_{num_trials}trials.pkl"
        with open(res_file, "wb") as f:
            pkl.dump({
                'results': results,
                'true_theta': true_theta,
                'ms': ms,
                'n': n,
                'num_trials': num_trials
            }, f)
        print(f"\nResults saved to {res_file}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if 'DML' in avg_mapes and 'Primary' in avg_mapes:
        dml_improvement = avg_mapes['Primary'] - avg_mapes['DML']
        print(f"  DML improvement over Primary: {dml_improvement:+.2f}%")

    if PPI_AVAILABLE:
        if 'DML' in avg_mapes and 'PPI' in avg_mapes:
            dml_vs_ppi = avg_mapes['PPI'] - avg_mapes['DML']
            print(f"  DML improvement over PPI: {dml_vs_ppi:+.2f}%")
        if 'DML' in avg_mapes and 'PPI++' in avg_mapes:
            dml_vs_ppi_pp = avg_mapes['PPI++'] - avg_mapes['DML']
            print(f"  DML improvement over PPI++: {dml_vs_ppi_pp:+.2f}%")


if __name__ == "__main__":
    main()
