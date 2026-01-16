"""
Double Machine Learning (DML) for AI-Augmented Generalized Linear Models
=========================================================================

This module implements the DML estimator from:
    "AI Data Augmentation for Generalized Linear Models"
    Lu, Wang, Zhang, Zhang (2026)

The DML estimator combines scarce human-labeled data with abundant AI-generated
labels to estimate parameters of a generalized linear model (specifically,
multinomial logit / binary choice models).

Key Components
--------------
1. Score Function (Equation 1.3 in paper):

   ψ(Ξ; e, g; β) = X^T [∇b(Xβ) - g(X,z) + (w/e(X,z))(g(X,z) - y)]

   where:
   - X: feature matrix (difference between alternatives)
   - y: human label (observed only when w=1)
   - w: indicator for whether human label is observed (w=1 for primary, w=0 for augmented)
   - z: AI-generated label/prediction
   - g(X,z) = E[y|X,z]: conditional expectation of y given X and z
   - e(X,z) = P(w=1|X,z): propensity score (probability of being in primary data)
   - β: parameter vector to estimate
   - b(θ): log-partition function (for binary logit: b(θ) = log(1 + exp(θ)))

2. DML Algorithm (Section 2 in paper):

   Step 1: Split data into K folds
   Step 2: For each fold k:
           - Train g on data NOT in fold k (cross-fitting)
           - Compute score/loss using out-of-fold predictions
   Step 3: Estimate β (two variants available)

3. Two Implementation Variants:

   - DML: Single β optimization using all data with cross-fitted predictions
   - DML-2: Fold-wise β estimation, then average (exact PDF algorithm)

   Both produce equivalent results (confirmed empirically).

4. Simplification (Corollary 2.1):

   When w is independent of (X, y, z), we can use constant e = n_primary / n_total
   instead of training a complex e(X,z) model.

Usage
-----
    from dml import run_dml, run_dml2, run_baselines, calculate_mape

    # Run DML
    beta_dml = run_dml(X_all, y_real, y_aug, real_rows, aug_rows)

    # Run DML-2 (exact PDF algorithm)
    beta_dml2 = run_dml2(X_all, y_real, y_aug, real_rows, aug_rows)

    # Compare to ground truth
    mape = calculate_mape(beta_dml, ground_truth)

Authors: Implementation based on Lu, Wang, Zhang, Zhang (2026)
"""

import numpy as np
import torch
from torch.optim import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# CONSTANTS
# =============================================================================

# Ground truth parameters for the conjoint experiment
GROUND_TRUTH_PARAMS = np.array([
    0.36310104, 0.7465673, 0.32377172, -0.21252407, 0.08090729,
    -0.09540857, -0.40639496, -0.15332593, -0.24158926, 0.17760716, -0.04599298
])


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def flatten(X):
    """
    Convert choice data to difference features.

    For binary choice between two alternatives, the MNL model uses the
    difference in utilities: U_1 - U_0 = (X_1 - X_0) @ β

    Parameters
    ----------
    X : ndarray of shape (2, d+1)
        Feature matrix for one observation with 2 alternatives.
        First column is constant (intercept), remaining d columns are features.

    Returns
    -------
    ndarray of shape (d,)
        Difference features: X[1, 1:] - X[0, 1:] (excluding intercept)
    """
    return X[1][1:] - X[0][1:]


def flatten_all(X_list):
    """
    Apply flatten to all observations.

    Parameters
    ----------
    X_list : list of ndarray
        List of feature matrices, each of shape (2, d+1)

    Returns
    -------
    ndarray of shape (n, d)
        Flattened difference features for all observations
    """
    return np.array([flatten(X) for X in X_list])


def calculate_mape(estimated, true):
    """
    Calculate Mean Absolute Percentage Error.

    Uses formula: mean(|estimated - true| / (|true| + 1)) × 100
    The +1 in denominator handles near-zero true values.

    Parameters
    ----------
    estimated : ndarray
        Estimated parameters
    true : ndarray
        True parameters

    Returns
    -------
    float
        MAPE as percentage
    """
    return np.mean(np.abs(estimated - true) / (np.abs(true) + 1)) * 100


# =============================================================================
# G-MODEL: E[y|X,z] ESTIMATION
# =============================================================================

def train_g_stratified(X_flat, z, y, C=0.05):
    """
    Train stratified g(X,z) = E[y|X,z] models.

    Trains separate logistic regression models for each value of z.
    This stratification captures how the relationship between X and y
    may differ depending on the AI prediction z.

    Parameters
    ----------
    X_flat : ndarray of shape (n, d)
        Flattened difference features
    z : ndarray of shape (n,)
        AI predictions (typically -1, 0, or 1)
    y : ndarray of shape (n,)
        Human labels (0 or 1)
    C : float, default=0.05
        Regularization parameter. Lower C = stronger regularization.
        Strong regularization (C=0.05) produces well-calibrated probabilities.

    Returns
    -------
    dict
        Dictionary mapping z values to fitted LogisticRegression models.
        Returns None for z values with insufficient data.
    """
    g_models = {}

    for z_val in np.unique(z):
        mask = (z == z_val)
        n_samples = np.sum(mask)
        n_classes = len(np.unique(y[mask]))

        # Need at least 2 samples and both classes to fit logistic regression
        if n_samples >= 2 and n_classes == 2:
            clf = LogisticRegression(C=C, max_iter=2000, random_state=1)
            clf.fit(X_flat[mask], y[mask])
            g_models[z_val] = clf
        else:
            g_models[z_val] = None

    return g_models


def predict_g(X_flat, z, g_models, fallback_prob):
    """
    Predict g(X,z) = P(y=1|X,z) using stratified models.

    Parameters
    ----------
    X_flat : ndarray of shape (n, d)
        Flattened difference features
    z : ndarray of shape (n,)
        AI predictions
    g_models : dict
        Fitted g models from train_g_stratified
    fallback_prob : float
        Probability to use when no model is available for a z value

    Returns
    -------
    ndarray of shape (n,)
        Predicted probabilities P(y=1|X,z)
    """
    n = len(z)
    g_proba = np.zeros(n)

    for i in range(n):
        z_i = z[i]
        if z_i in g_models and g_models[z_i] is not None:
            g_proba[i] = g_models[z_i].predict_proba(X_flat[i:i+1])[0, 1]
        else:
            g_proba[i] = fallback_prob

    return g_proba


# =============================================================================
# MNL OPTIMIZATION
# =============================================================================

def optimize_mnl_soft_targets(X_flat, tau, n_epochs=5000, lr=5e-3, seed=0):
    """
    Optimize MNL parameters with soft targets using MLE loss.

    The loss function is derived from the DML score equation:
        L = Σ [-τ·Xβ + log(1 + exp(Xβ))]

    This is equivalent to binary cross-entropy with soft target τ,
    where τ represents the DML-adjusted probability.

    Parameters
    ----------
    X_flat : ndarray of shape (n, d)
        Flattened difference features
    tau : ndarray of shape (n,)
        Soft targets (DML-adjusted probabilities), should be in [0, 1]
    n_epochs : int, default=5000
        Number of optimization epochs
    lr : float, default=5e-3
        Learning rate for Adam optimizer
    seed : int, default=0
        Random seed for reproducibility

    Returns
    -------
    ndarray of shape (d,)
        Estimated β parameters
    """
    X_tensor = torch.tensor(X_flat, dtype=torch.float64)
    tau_tensor = torch.tensor(tau, dtype=torch.float64)

    torch.manual_seed(seed)
    d = X_flat.shape[1]
    beta = torch.nn.Parameter(torch.zeros(d, dtype=torch.float64), requires_grad=True)
    optimizer = Adam([beta], lr=lr)

    n = len(tau)

    for _ in range(n_epochs):
        optimizer.zero_grad()

        # MLE loss: -τ·Xβ + log(1 + exp(Xβ))
        utility = X_tensor @ beta
        loss = torch.sum(-tau_tensor * utility + torch.log(1 + torch.exp(utility))) / n

        loss.backward()
        optimizer.step()

    return beta.detach().numpy()


# =============================================================================
# DML IMPLEMENTATION (Single β)
# =============================================================================

def run_dml(X_all, y_real, y_aug, real_rows, aug_rows, n_folds=5):
    """
    DML estimator with single β optimization.

    This implementation:
    1. Uses cross-fitting for g (trains on K-1 folds, predicts on held-out fold)
    2. Uses constant e = n_primary / n_total (Corollary 2.1)
    3. Optimizes single β on all data using cross-fitted g predictions

    The DML-adjusted targets are:
    - Primary (w=1): τ = g(1 - 1/e) + y/e
    - Augmented (w=0): τ = g

    Parameters
    ----------
    X_all : list of ndarray
        All feature matrices
    y_real : ndarray
        Human labels for all observations (only used for primary)
    y_aug : ndarray
        AI labels (z) for all observations
    real_rows : list of int
        Indices of primary (human-labeled) observations
    aug_rows : list of int
        Indices of augmented (AI-labeled only) observations
    n_folds : int, default=5
        Number of cross-validation folds

    Returns
    -------
    ndarray of shape (d,)
        Estimated β parameters
    """
    # Extract data subsets
    X_p = [X_all[i] for i in real_rows]
    y_p = np.array([y_real[i] for i in real_rows])
    z_p = np.array([y_aug[i] for i in real_rows])

    X_a = [X_all[i] for i in aug_rows]
    z_a = np.array([y_aug[i] for i in aug_rows])

    n_p, n_a = len(y_p), len(z_a)
    n_total = n_p + n_a

    # Constant e = P(w=1) = n_primary / n_total (Corollary 2.1)
    e_constant = n_p / n_total

    # Class prior as fallback for g predictions
    class_prior = np.mean(y_p)

    # Flatten features
    X_p_flat = flatten_all(X_p)
    X_a_flat = flatten_all(X_a)

    # Cross-fitting setup
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    # Storage for cross-fitted predictions
    g_proba_p = np.zeros(n_p)  # Out-of-fold predictions for primary
    g_proba_a = np.zeros(n_a)  # Averaged predictions for augmented
    a_fold_counts = np.zeros(n_a)

    # Cross-fitting loop
    for train_idx, val_idx in kf.split(X_p_flat):
        # Train g on training fold (out-of-fold for val_idx)
        g_models = train_g_stratified(
            X_p_flat[train_idx],
            z_p[train_idx],
            y_p[train_idx]
        )

        # Predict on validation fold (out-of-fold for primary)
        g_proba_p[val_idx] = predict_g(
            X_p_flat[val_idx],
            z_p[val_idx],
            g_models,
            class_prior
        )

        # Predict on all augmented data (always out-of-sample for g)
        # Accumulate predictions from each fold model
        g_proba_a += predict_g(X_a_flat, z_a, g_models, class_prior)
        a_fold_counts += 1

    # Average augmented predictions across folds
    g_proba_a = g_proba_a / a_fold_counts

    # Compute DML-adjusted targets
    # Primary (w=1): τ = g(1 - 1/e) + y/e
    tau_p = g_proba_p * (1 - 1/e_constant) + y_p / e_constant
    tau_p = np.clip(tau_p, 0.0, 1.0)

    # Augmented (w=0): τ = g
    tau_a = g_proba_a

    # Combine all data
    X_all_flat = np.vstack([X_p_flat, X_a_flat])
    tau_all = np.concatenate([tau_p, tau_a])

    # Optimize single β on all data
    beta = optimize_mnl_soft_targets(X_all_flat, tau_all)

    return beta


# =============================================================================
# DML-2 IMPLEMENTATION (Fold-wise β, then average)
# =============================================================================

def run_dml2(X_all, y_real, y_aug, real_rows, aug_rows, n_folds=5):
    """
    DML-2 estimator: Exact PDF algorithm with fold-wise β averaging.

    This implementation follows the paper exactly:
    1. Split primary data into K folds
    2. For each fold k:
       - Train g on folds ≠ k
       - Compute β_k using fold k's primary data + all augmented data
    3. Return β̂ = (1/K) Σ β̂_k

    This produces equivalent results to run_dml (empirically verified).

    Parameters
    ----------
    X_all : list of ndarray
        All feature matrices
    y_real : ndarray
        Human labels for all observations
    y_aug : ndarray
        AI labels (z) for all observations
    real_rows : list of int
        Indices of primary observations
    aug_rows : list of int
        Indices of augmented observations
    n_folds : int, default=5
        Number of cross-validation folds

    Returns
    -------
    ndarray of shape (d,)
        Estimated β parameters (averaged across folds)
    """
    # Extract data subsets
    X_p = [X_all[i] for i in real_rows]
    y_p = np.array([y_real[i] for i in real_rows])
    z_p = np.array([y_aug[i] for i in real_rows])

    X_a = [X_all[i] for i in aug_rows]
    z_a = np.array([y_aug[i] for i in aug_rows])

    n_p, n_a = len(y_p), len(z_a)
    n_total = n_p + n_a

    # Constant e (Corollary 2.1)
    e_constant = n_p / n_total

    class_prior = np.mean(y_p)

    X_p_flat = flatten_all(X_p)
    X_a_flat = flatten_all(X_a)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    # Store β estimates from each fold
    beta_folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_p_flat)):
        # Train g on training fold
        g_models = train_g_stratified(
            X_p_flat[train_idx],
            z_p[train_idx],
            y_p[train_idx]
        )

        # Predict on validation fold (primary)
        g_proba_val = predict_g(
            X_p_flat[val_idx],
            z_p[val_idx],
            g_models,
            class_prior
        )

        # Predict on all augmented
        g_proba_aug = predict_g(X_a_flat, z_a, g_models, class_prior)

        # Compute DML targets for this fold
        y_val = y_p[val_idx]
        tau_val = g_proba_val * (1 - 1/e_constant) + y_val / e_constant
        tau_val = np.clip(tau_val, 0.0, 1.0)
        tau_aug = g_proba_aug

        # Combine fold's primary + all augmented
        X_fold = np.vstack([X_p_flat[val_idx], X_a_flat])
        tau_fold = np.concatenate([tau_val, tau_aug])

        # Optimize β_k for this fold
        beta_k = optimize_mnl_soft_targets(
            X_fold, tau_fold,
            n_epochs=3000,  # Fewer epochs since less data per fold
            seed=fold_idx
        )
        beta_folds.append(beta_k)

    # Average β across folds (Step 3 of PDF algorithm)
    beta_avg = np.mean(beta_folds, axis=0)

    return beta_avg


# =============================================================================
# BASELINE METHODS
# =============================================================================

def run_primary_only(X_all, y_real, real_rows):
    """
    Primary Only baseline: Uses only human-labeled data.

    This is the standard MLE estimator using only primary data,
    ignoring all augmented (AI-labeled) data.

    Parameters
    ----------
    X_all : list of ndarray
        All feature matrices
    y_real : ndarray
        Human labels
    real_rows : list of int
        Indices of primary observations

    Returns
    -------
    ndarray of shape (d,)
        Estimated β parameters
    """
    X_p = [X_all[i] for i in real_rows]
    y_p = np.array([y_real[i] for i in real_rows])

    X_p_flat = flatten_all(X_p)

    # Standard MLE: τ = y (hard labels)
    return optimize_mnl_soft_targets(X_p_flat, y_p.astype(float))


def run_naive(X_all, y_real, y_aug, real_rows, aug_rows):
    """
    Naive baseline: Uses AI labels (z) as hard labels for augmented data.

    This method treats AI predictions as ground truth, which can introduce
    bias if AI predictions are systematically wrong.

    Parameters
    ----------
    X_all : list of ndarray
        All feature matrices
    y_real : ndarray
        Human labels
    y_aug : ndarray
        AI labels (z)
    real_rows : list of int
        Indices of primary observations
    aug_rows : list of int
        Indices of augmented observations

    Returns
    -------
    ndarray of shape (d,)
        Estimated β parameters
    """
    # Primary: use human labels
    X_p = [X_all[i] for i in real_rows]
    y_p = np.array([y_real[i] for i in real_rows])

    # Augmented: use AI labels as hard labels (filter invalid z=-1)
    X_a, y_a = [], []
    for i in aug_rows:
        z = y_aug[i]
        if z in [0, 1]:  # Only use valid binary predictions
            X_a.append(X_all[i])
            y_a.append(z)

    # Combine
    X_all_subset = X_a + X_p
    y_all = np.array(y_a + list(y_p))

    X_flat = flatten_all(X_all_subset)

    return optimize_mnl_soft_targets(X_flat, y_all.astype(float))


def run_aae(X_all, y_real, y_aug, real_rows, aug_rows):
    """
    AAE (AI-Augmented Estimation) baseline.

    Uses g(X) models stratified by z to produce soft labels for augmented data.
    This is the predecessor to DML - it uses soft labels but lacks the
    debiasing correction term (w/e)(g-y).

    Parameters
    ----------
    X_all : list of ndarray
        All feature matrices
    y_real : ndarray
        Human labels
    y_aug : ndarray
        AI labels (z)
    real_rows : list of int
        Indices of primary observations
    aug_rows : list of int
        Indices of augmented observations

    Returns
    -------
    ndarray of shape (d,)
        Estimated β parameters
    """
    X_p = [X_all[i] for i in real_rows]
    y_p = np.array([y_real[i] for i in real_rows])
    z_p = np.array([y_aug[i] for i in real_rows])

    X_a = [X_all[i] for i in aug_rows]
    z_a = np.array([y_aug[i] for i in aug_rows])

    X_p_flat = flatten_all(X_p)
    X_a_flat = flatten_all(X_a)

    class_prior = np.mean(y_p)

    # Train g models (MLP as in original AAE)
    g_models = {}
    for z_val in np.unique(z_p):
        mask = (z_p == z_val)
        if np.sum(mask) >= 2 and len(np.unique(y_p[mask])) == 2:
            clf = MLPClassifier(
                hidden_layer_sizes=(10, 5),
                activation='logistic',
                solver='adam',
                alpha=1e-4,
                max_iter=500,
                random_state=1
            )
            clf.fit(X_p_flat[mask], y_p[mask])
            g_models[z_val] = clf

    # Primary: hard labels
    tau_p = y_p.astype(float)

    # Augmented: soft labels from g
    tau_a = predict_g(X_a_flat, z_a, g_models, class_prior)

    # Combine
    X_flat = np.vstack([X_a_flat, X_p_flat])
    tau = np.concatenate([tau_a, tau_p])

    return optimize_mnl_soft_targets(X_flat, tau)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_all_methods(X_all, y_real, y_aug, real_rows, aug_rows):
    """
    Run all estimation methods and return results.

    Parameters
    ----------
    X_all : list of ndarray
        All feature matrices
    y_real : ndarray
        Human labels
    y_aug : ndarray
        AI labels (z)
    real_rows : list of int
        Indices of primary observations
    aug_rows : list of int
        Indices of augmented observations

    Returns
    -------
    dict
        Dictionary mapping method names to estimated β arrays
    """
    results = {}

    results['Primary'] = run_primary_only(X_all, y_real, real_rows)
    results['Naive'] = run_naive(X_all, y_real, y_aug, real_rows, aug_rows)
    results['AAE'] = run_aae(X_all, y_real, y_aug, real_rows, aug_rows)
    results['DML'] = run_dml(X_all, y_real, y_aug, real_rows, aug_rows)
    results['DML-2'] = run_dml2(X_all, y_real, y_aug, real_rows, aug_rows)

    return results
