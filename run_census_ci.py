#!/usr/bin/env python3
"""
Census Data Benchmark with Confidence Intervals
================================================

Uses analytical confidence intervals instead of simulation-based MAPE.
Compares DML, PPI, PPI++ using their native CI methods.

Usage:
    python run_census_ci.py
    python run_census_ci.py --m 500 --n 2000
"""

import numpy as np
import argparse
from scipy import stats
from sklearn.linear_model import LogisticRegression
from ppi_py.datasets import load_dataset
from ppi_py import ppi_logistic_pointestimate, ppi_logistic_ci, logistic
import warnings
warnings.filterwarnings("ignore")


def calculate_mape(estimated, true):
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs(estimated - true) / (np.abs(true) + 1e-10)) * 100


def get_logistic_se(X, y, beta):
    """
    Get standard errors for logistic regression coefficients.
    Uses the Fisher information matrix.
    """
    u = X @ beta
    p = 1 / (1 + np.exp(-np.clip(u, -500, 500)))
    W = np.diag(p * (1 - p))
    try:
        fisher_info = X.T @ W @ X
        cov_matrix = np.linalg.inv(fisher_info)
        se = np.sqrt(np.diag(cov_matrix))
        return se
    except np.linalg.LinAlgError:
        return np.full(len(beta), np.nan)


def run_dml_direct_with_se(X_labeled, y_labeled, z_labeled, X_unlabeled, z_unlabeled):
    """
    DML with g=z directly, returning point estimate and standard errors.

    Uses influence function for asymptotic variance estimation.
    """
    from scipy.optimize import minimize

    n_p = len(y_labeled)
    n_a = len(z_unlabeled)
    n_total = n_p + n_a
    e_constant = n_p / n_total

    # DML targets
    tau_p = z_labeled * (1 - 1/e_constant) + y_labeled / e_constant
    tau_p = np.clip(tau_p, 0.0, 1.0)
    tau_a = z_unlabeled

    X_all = np.vstack([X_labeled, X_unlabeled])
    tau_all = np.concatenate([tau_p, tau_a])

    # Optimize
    def neg_loglik(beta):
        u = X_all @ beta
        u = np.clip(u, -500, 500)
        return np.mean(-tau_all * u + np.log(1 + np.exp(u)))

    def neg_loglik_grad(beta):
        u = X_all @ beta
        u = np.clip(u, -500, 500)
        p = 1 / (1 + np.exp(-u))
        return X_all.T @ (p - tau_all) / len(tau_all)

    result = minimize(
        neg_loglik, np.zeros(X_all.shape[1]),
        method='L-BFGS-B', jac=neg_loglik_grad,
        options={'maxiter': 10000, 'ftol': 1e-12}
    )
    beta = result.x

    # Standard errors via sandwich estimator
    u = X_all @ beta
    u = np.clip(u, -500, 500)
    p = 1 / (1 + np.exp(-u))
    residuals = tau_all - p

    # Influence function: psi_i = (X_i * residual_i)
    psi = X_all * residuals[:, np.newaxis]

    # Meat: E[psi psi']
    meat = psi.T @ psi / n_total

    # Bread: E[d psi / d beta] = -E[X X' p(1-p)]
    W = p * (1 - p)
    bread = X_all.T @ (X_all * W[:, np.newaxis]) / n_total

    try:
        bread_inv = np.linalg.inv(bread)
        cov = bread_inv @ meat @ bread_inv.T / n_total
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)

    return beta, se


def main():
    parser = argparse.ArgumentParser(description='Census CI-based Benchmark')
    parser.add_argument('--m', type=int, default=100, help='Number of labeled samples')
    parser.add_argument('--n', type=int, default=2000, help='Number of unlabeled samples')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    print("=" * 70)
    print("CENSUS DATA BENCHMARK WITH CONFIDENCE INTERVALS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = load_dataset('data/census/', 'census_healthcare')
    Y_total = data["Y"]
    Yhat_total = data["Yhat"]
    X_total = data["X"].copy()

    # Normalize
    X_total[:, 0] = (X_total[:, 0] - X_total[:, 0].min()) / (X_total[:, 0].max() - X_total[:, 0].min())
    X_total[:, 1] = X_total[:, 1] / X_total[:, 1].max()

    n_total = len(Y_total)

    # Ground truth
    true_theta = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=10000, tol=1e-15, fit_intercept=False
    ).fit(X_total, Y_total).coef_.squeeze()

    print(f"Total samples: {n_total}")
    print(f"Ground truth β: {true_theta}")
    print(f"\nExperiment: m={args.m}, n={args.n}, seed={args.seed}")

    # Sample data
    rng = np.random.RandomState(args.seed)
    rand_idx = rng.permutation(n_total)

    X_lab = X_total[rand_idx[:args.m]]
    Y_lab = Y_total[rand_idx[:args.m]]
    Yhat_lab = Yhat_total[rand_idx[:args.m]]

    X_unlab = X_total[rand_idx[args.m:args.m + args.n]]
    Yhat_unlab = Yhat_total[rand_idx[args.m:args.m + args.n]]

    optimizer_options = {"ftol": 1e-5, "gtol": 1e-5, "maxls": 10000, "maxiter": 10000}

    # =========================================================================
    # Method 1: Primary (Human-data-only)
    # =========================================================================
    print("\n" + "-" * 70)
    print("1. PRIMARY (Human-data-only)")
    print("-" * 70)

    primary_beta = logistic(X_lab, Y_lab)
    primary_se = get_logistic_se(X_lab, Y_lab, primary_beta)
    primary_ci = (
        primary_beta - stats.norm.ppf(1 - args.alpha/2) * primary_se,
        primary_beta + stats.norm.ppf(1 - args.alpha/2) * primary_se
    )

    print(f"Point estimate: {primary_beta}")
    print(f"Standard errors: {primary_se}")
    print(f"95% CI: [{primary_ci[0]}, {primary_ci[1]}]")
    print(f"MAPE: {calculate_mape(primary_beta, true_theta):.2f}%")
    print(f"True β in CI: {np.all((true_theta >= primary_ci[0]) & (true_theta <= primary_ci[1]))}")

    # =========================================================================
    # Method 2: PPI (λ=1)
    # =========================================================================
    print("\n" + "-" * 70)
    print("2. PPI (λ=1)")
    print("-" * 70)

    try:
        ppi_ci_lower, ppi_ci_upper = ppi_logistic_ci(
            X_lab, Y_lab.astype(float), Yhat_lab.astype(float),
            X_unlab, Yhat_unlab.astype(float),
            lam=1.0,
            alpha=args.alpha,
            optimizer_options=optimizer_options
        )
        ppi_beta = (ppi_ci_lower + ppi_ci_upper) / 2  # Point estimate from CI midpoint
        ppi_se = (ppi_ci_upper - ppi_ci_lower) / (2 * stats.norm.ppf(1 - args.alpha/2))

        print(f"Point estimate: {ppi_beta}")
        print(f"Standard errors: {ppi_se}")
        print(f"95% CI: [{ppi_ci_lower}, {ppi_ci_upper}]")
        print(f"MAPE: {calculate_mape(ppi_beta, true_theta):.2f}%")
        print(f"True β in CI: {np.all((true_theta >= ppi_ci_lower) & (true_theta <= ppi_ci_upper))}")
    except Exception as e:
        print(f"Error: {e}")
        ppi_beta = None
        ppi_se = None

    # =========================================================================
    # Method 3: PPI++ (auto λ)
    # =========================================================================
    print("\n" + "-" * 70)
    print("3. PPI++ (auto λ)")
    print("-" * 70)

    try:
        ppi_pp_ci_lower, ppi_pp_ci_upper = ppi_logistic_ci(
            X_lab, Y_lab.astype(float), Yhat_lab.astype(float),
            X_unlab, Yhat_unlab.astype(float),
            alpha=args.alpha,
            optimizer_options=optimizer_options
        )
        ppi_pp_beta = (ppi_pp_ci_lower + ppi_pp_ci_upper) / 2
        ppi_pp_se = (ppi_pp_ci_upper - ppi_pp_ci_lower) / (2 * stats.norm.ppf(1 - args.alpha/2))

        print(f"Point estimate: {ppi_pp_beta}")
        print(f"Standard errors: {ppi_pp_se}")
        print(f"95% CI: [{ppi_pp_ci_lower}, {ppi_pp_ci_upper}]")
        print(f"MAPE: {calculate_mape(ppi_pp_beta, true_theta):.2f}%")
        print(f"True β in CI: {np.all((true_theta >= ppi_pp_ci_lower) & (true_theta <= ppi_pp_ci_upper))}")
    except Exception as e:
        print(f"Error: {e}")
        ppi_pp_beta = None
        ppi_pp_se = None

    # =========================================================================
    # Method 4: DML (g=z)
    # =========================================================================
    print("\n" + "-" * 70)
    print("4. DML (g=z)")
    print("-" * 70)

    dml_beta, dml_se = run_dml_direct_with_se(X_lab, Y_lab, Yhat_lab, X_unlab, Yhat_unlab)
    dml_ci = (
        dml_beta - stats.norm.ppf(1 - args.alpha/2) * dml_se,
        dml_beta + stats.norm.ppf(1 - args.alpha/2) * dml_se
    )

    print(f"Point estimate: {dml_beta}")
    print(f"Standard errors: {dml_se}")
    print(f"95% CI: [{dml_ci[0]}, {dml_ci[1]}]")
    print(f"MAPE: {calculate_mape(dml_beta, true_theta):.2f}%")
    print(f"True β in CI: {np.all((true_theta >= dml_ci[0]) & (true_theta <= dml_ci[1]))}")

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<15} | {'β[0]':>12} | {'β[1]':>12} | {'MAPE':>8} | {'CI Width[0]':>12} | {'CI Width[1]':>12}")
    print("-" * 80)

    methods = [
        ("Primary", primary_beta, primary_se),
        ("DML (g=z)", dml_beta, dml_se),
    ]
    if ppi_beta is not None:
        methods.append(("PPI", ppi_beta, ppi_se))
    if ppi_pp_beta is not None:
        methods.append(("PPI++", ppi_pp_beta, ppi_pp_se))

    for name, beta, se in methods:
        mape = calculate_mape(beta, true_theta)
        ci_width = 2 * stats.norm.ppf(1 - args.alpha/2) * se
        print(f"{name:<15} | {beta[0]:12.4f} | {beta[1]:12.6f} | {mape:7.2f}% | {ci_width[0]:12.4f} | {ci_width[1]:12.6f}")

    print(f"\nTrue β:          | {true_theta[0]:12.4f} | {true_theta[1]:12.6f}")

    # =========================================================================
    # Bias comparison
    # =========================================================================
    print("\n" + "-" * 70)
    print("BIAS FROM TRUE β")
    print("-" * 70)

    print(f"{'Method':<15} | {'Bias[0]':>12} | {'Bias[1]':>12} | {'Relative Bias[0]':>16} | {'Relative Bias[1]':>16}")
    print("-" * 80)

    for name, beta, _ in methods:
        bias = beta - true_theta
        rel_bias = bias / true_theta * 100
        print(f"{name:<15} | {bias[0]:+12.4f} | {bias[1]:+12.6f} | {rel_bias[0]:+15.2f}% | {rel_bias[1]:+15.2f}%")


if __name__ == "__main__":
    main()
