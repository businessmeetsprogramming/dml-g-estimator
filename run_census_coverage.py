#!/usr/bin/env python3
"""
Census Data Benchmark: Coverage Probability
============================================

Computes coverage probability of confidence intervals across multiple trials.
A 95% CI should cover the true parameter ~95% of the time.

Usage:
    python run_census_coverage.py
    python run_census_coverage.py --num_trials 100 --m 500
"""

import numpy as np
import argparse
from scipy import stats
from sklearn.linear_model import LogisticRegression
from ppi_py.datasets import load_dataset
from ppi_py import ppi_logistic_ci, logistic
import warnings
warnings.filterwarnings("ignore")


def calculate_mape(estimated, true):
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs(estimated - true) / (np.abs(true) + 1e-10)) * 100


def get_logistic_se(X, y, beta):
    """Get standard errors for logistic regression coefficients."""
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
    """DML with g=z directly, returning point estimate and standard errors."""
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

    psi = X_all * residuals[:, np.newaxis]
    meat = psi.T @ psi / n_total

    W = p * (1 - p)
    bread = X_all.T @ (X_all * W[:, np.newaxis]) / n_total

    try:
        bread_inv = np.linalg.inv(bread)
        cov = bread_inv @ meat @ bread_inv.T / n_total
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)

    return beta, se


def run_single_trial(X_total, Y_total, Yhat_total, true_theta, m, n, seed, alpha=0.05):
    """Run a single trial and return coverage indicators and metrics for each method."""

    rng = np.random.RandomState(seed)
    rand_idx = rng.permutation(len(Y_total))

    X_lab = X_total[rand_idx[:m]]
    Y_lab = Y_total[rand_idx[:m]]
    Yhat_lab = Yhat_total[rand_idx[:m]]

    X_unlab = X_total[rand_idx[m:m + n]]
    Yhat_unlab = Yhat_total[rand_idx[m:m + n]]

    z_crit = stats.norm.ppf(1 - alpha/2)
    optimizer_options = {"ftol": 1e-5, "gtol": 1e-5, "maxls": 10000, "maxiter": 10000}

    results = {}

    # Primary
    try:
        primary_beta = logistic(X_lab, Y_lab)
        primary_se = get_logistic_se(X_lab, Y_lab, primary_beta)
        primary_ci_lower = primary_beta - z_crit * primary_se
        primary_ci_upper = primary_beta + z_crit * primary_se

        # Coverage: check if true_theta is within CI for each coefficient
        primary_coverage = (true_theta >= primary_ci_lower) & (true_theta <= primary_ci_upper)

        results['Primary'] = {
            'beta': primary_beta,
            'se': primary_se,
            'ci_lower': primary_ci_lower,
            'ci_upper': primary_ci_upper,
            'coverage': primary_coverage,  # Boolean array for each coefficient
            'mape': calculate_mape(primary_beta, true_theta),
            'ci_width': 2 * z_crit * primary_se
        }
    except Exception as e:
        results['Primary'] = None

    # DML (g=z)
    try:
        dml_beta, dml_se = run_dml_direct_with_se(X_lab, Y_lab, Yhat_lab, X_unlab, Yhat_unlab)
        dml_ci_lower = dml_beta - z_crit * dml_se
        dml_ci_upper = dml_beta + z_crit * dml_se

        dml_coverage = (true_theta >= dml_ci_lower) & (true_theta <= dml_ci_upper)

        results['DML'] = {
            'beta': dml_beta,
            'se': dml_se,
            'ci_lower': dml_ci_lower,
            'ci_upper': dml_ci_upper,
            'coverage': dml_coverage,
            'mape': calculate_mape(dml_beta, true_theta),
            'ci_width': 2 * z_crit * dml_se
        }
    except Exception as e:
        results['DML'] = None

    # PPI (λ=1)
    try:
        ppi_ci_lower, ppi_ci_upper = ppi_logistic_ci(
            X_lab, Y_lab.astype(float), Yhat_lab.astype(float),
            X_unlab, Yhat_unlab.astype(float),
            lam=1.0, alpha=alpha, optimizer_options=optimizer_options
        )
        ppi_beta = (ppi_ci_lower + ppi_ci_upper) / 2
        ppi_se = (ppi_ci_upper - ppi_ci_lower) / (2 * z_crit)

        ppi_coverage = (true_theta >= ppi_ci_lower) & (true_theta <= ppi_ci_upper)

        results['PPI'] = {
            'beta': ppi_beta,
            'se': ppi_se,
            'ci_lower': ppi_ci_lower,
            'ci_upper': ppi_ci_upper,
            'coverage': ppi_coverage,
            'mape': calculate_mape(ppi_beta, true_theta),
            'ci_width': ppi_ci_upper - ppi_ci_lower
        }
    except Exception as e:
        results['PPI'] = None

    # PPI++ (auto λ)
    try:
        ppi_pp_ci_lower, ppi_pp_ci_upper = ppi_logistic_ci(
            X_lab, Y_lab.astype(float), Yhat_lab.astype(float),
            X_unlab, Yhat_unlab.astype(float),
            alpha=alpha, optimizer_options=optimizer_options
        )
        ppi_pp_beta = (ppi_pp_ci_lower + ppi_pp_ci_upper) / 2
        ppi_pp_se = (ppi_pp_ci_upper - ppi_pp_ci_lower) / (2 * z_crit)

        ppi_pp_coverage = (true_theta >= ppi_pp_ci_lower) & (true_theta <= ppi_pp_ci_upper)

        results['PPI++'] = {
            'beta': ppi_pp_beta,
            'se': ppi_pp_se,
            'ci_lower': ppi_pp_ci_lower,
            'ci_upper': ppi_pp_ci_upper,
            'coverage': ppi_pp_coverage,
            'mape': calculate_mape(ppi_pp_beta, true_theta),
            'ci_width': ppi_pp_ci_upper - ppi_pp_ci_lower
        }
    except Exception as e:
        results['PPI++'] = None

    return results


def main():
    parser = argparse.ArgumentParser(description='Census Coverage Probability Benchmark')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--m', type=int, default=None, help='Number of labeled samples (if None, run all)')
    parser.add_argument('--n', type=int, default=2000, help='Number of unlabeled samples')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--save_results', action='store_true', help='Save results to file')
    args = parser.parse_args()

    print("=" * 80)
    print("CENSUS DATA BENCHMARK: COVERAGE PROBABILITY")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    data = load_dataset('data/census/', 'census_healthcare')
    Y_total = data["Y"]
    Yhat_total = data["Yhat"]
    X_total = data["X"].copy()

    # Normalize
    X_total[:, 0] = (X_total[:, 0] - X_total[:, 0].min()) / (X_total[:, 0].max() - X_total[:, 0].min())
    X_total[:, 1] = X_total[:, 1] / X_total[:, 1].max()

    # Ground truth
    true_theta = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=10000, tol=1e-15, fit_intercept=False
    ).fit(X_total, Y_total).coef_.squeeze()

    print(f"Total samples: {len(Y_total)}")
    print(f"Ground truth β: {true_theta}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Confidence level: {100*(1-args.alpha):.0f}%")

    # Sample sizes to test
    if args.m is not None:
        m_values = [args.m]
    else:
        m_values = [100, 250, 500, 750, 1000]

    methods = ['Primary', 'DML', 'PPI', 'PPI++']
    n_coef = len(true_theta)

    # Store results
    all_results = {m: {method: {'coverage': [], 'mape': [], 'ci_width': []}
                       for method in methods} for m in m_values}

    for m in m_values:
        print(f"\n{'='*80}")
        print(f"Running m = {m}, n = {args.n}")
        print("=" * 80)

        for trial in range(args.num_trials):
            if (trial + 1) % 20 == 0:
                print(f"  Completed trial {trial + 1}/{args.num_trials}")

            trial_results = run_single_trial(
                X_total, Y_total, Yhat_total, true_theta,
                m, args.n, seed=trial, alpha=args.alpha
            )

            for method in methods:
                if trial_results[method] is not None:
                    all_results[m][method]['coverage'].append(trial_results[method]['coverage'])
                    all_results[m][method]['mape'].append(trial_results[method]['mape'])
                    all_results[m][method]['ci_width'].append(trial_results[method]['ci_width'])

        # Print results for this m
        print(f"\n  Results for m={m}:")
        print(f"  {'Method':<10} | {'Cov(β₀)':>8} | {'Cov(β₁)':>8} | {'Cov(all)':>8} | {'MAPE':>8} | {'CI Width(β₀)':>12} | {'CI Width(β₁)':>12}")
        print("  " + "-" * 85)

        for method in methods:
            if len(all_results[m][method]['coverage']) > 0:
                coverage_arr = np.array(all_results[m][method]['coverage'])
                mape_arr = np.array(all_results[m][method]['mape'])
                ci_width_arr = np.array(all_results[m][method]['ci_width'])

                # Coverage probability for each coefficient
                cov_prob = coverage_arr.mean(axis=0)
                # Coverage probability for all coefficients jointly
                cov_all = coverage_arr.all(axis=1).mean()

                avg_mape = mape_arr.mean()
                avg_ci_width = ci_width_arr.mean(axis=0)

                print(f"  {method:<10} | {cov_prob[0]*100:7.1f}% | {cov_prob[1]*100:7.1f}% | {cov_all*100:7.1f}% | {avg_mape:7.1f}% | {avg_ci_width[0]:12.2f} | {avg_ci_width[1]:12.4f}")
            else:
                print(f"  {method:<10} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>12} | {'N/A':>12}")

    # Final summary table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: COVERAGE PROBABILITY (%)")
    print("=" * 80)
    print(f"\nExpected coverage for {100*(1-args.alpha):.0f}% CI: {100*(1-args.alpha):.0f}%")
    print(f"Number of trials: {args.num_trials}")

    print(f"\n{'Method':<10} |", end="")
    for m in m_values:
        print(f" m={m:>4} |", end="")
    print("   Avg")
    print("-" * (14 + 9 * len(m_values) + 8))

    for method in methods:
        print(f"{method:<10} |", end="")
        method_coverages = []
        for m in m_values:
            if len(all_results[m][method]['coverage']) > 0:
                coverage_arr = np.array(all_results[m][method]['coverage'])
                cov_all = coverage_arr.all(axis=1).mean() * 100
                method_coverages.append(cov_all)
                print(f" {cov_all:5.1f}% |", end="")
            else:
                print(f"   N/A |", end="")
        if method_coverages:
            print(f" {np.mean(method_coverages):5.1f}%")
        else:
            print("   N/A")

    # Per-coefficient coverage
    print(f"\n{'='*80}")
    print("COVERAGE PROBABILITY BY COEFFICIENT (%)")
    print("=" * 80)

    for coef_idx in range(n_coef):
        print(f"\nβ[{coef_idx}] (true = {true_theta[coef_idx]:.4f}):")
        print(f"{'Method':<10} |", end="")
        for m in m_values:
            print(f" m={m:>4} |", end="")
        print("   Avg")
        print("-" * (14 + 9 * len(m_values) + 8))

        for method in methods:
            print(f"{method:<10} |", end="")
            method_coverages = []
            for m in m_values:
                if len(all_results[m][method]['coverage']) > 0:
                    coverage_arr = np.array(all_results[m][method]['coverage'])
                    cov = coverage_arr[:, coef_idx].mean() * 100
                    method_coverages.append(cov)
                    print(f" {cov:5.1f}% |", end="")
                else:
                    print(f"   N/A |", end="")
            if method_coverages:
                print(f" {np.mean(method_coverages):5.1f}%")
            else:
                print("   N/A")

    # MAPE summary
    print(f"\n{'='*80}")
    print("MAPE (%) - Lower is Better")
    print("=" * 80)

    print(f"\n{'Method':<10} |", end="")
    for m in m_values:
        print(f" m={m:>4} |", end="")
    print("   Avg")
    print("-" * (14 + 9 * len(m_values) + 8))

    for method in methods:
        print(f"{method:<10} |", end="")
        method_mapes = []
        for m in m_values:
            if len(all_results[m][method]['mape']) > 0:
                avg_mape = np.mean(all_results[m][method]['mape'])
                method_mapes.append(avg_mape)
                print(f" {avg_mape:5.1f}% |", end="")
            else:
                print(f"   N/A |", end="")
        if method_mapes:
            print(f" {np.mean(method_mapes):5.1f}%")
        else:
            print("   N/A")

    # CI Width summary
    print(f"\n{'='*80}")
    print("AVERAGE CI WIDTH FOR β₀")
    print("=" * 80)

    print(f"\n{'Method':<10} |", end="")
    for m in m_values:
        print(f" m={m:>4} |", end="")
    print("   Avg")
    print("-" * (14 + 9 * len(m_values) + 8))

    for method in methods:
        print(f"{method:<10} |", end="")
        method_widths = []
        for m in m_values:
            if len(all_results[m][method]['ci_width']) > 0:
                avg_width = np.mean([w[0] for w in all_results[m][method]['ci_width']])
                method_widths.append(avg_width)
                print(f" {avg_width:5.1f} |", end="")
            else:
                print(f"   N/A |", end="")
        if method_widths:
            print(f" {np.mean(method_widths):5.1f}")
        else:
            print("   N/A")

    # Save results
    if args.save_results:
        import pickle
        save_path = f'res/census_coverage_{args.num_trials}trials.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump({
                'all_results': all_results,
                'true_theta': true_theta,
                'm_values': m_values,
                'n': args.n,
                'alpha': args.alpha,
                'num_trials': args.num_trials
            }, f)
        print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
