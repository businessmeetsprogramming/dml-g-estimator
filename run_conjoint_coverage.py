#!/usr/bin/env python3
"""
Conjoint Data Benchmark: Coverage Probability
==============================================

Computes coverage probability of confidence intervals across multiple trials.
A 95% CI should cover the true parameter ~95% of the time.

Usage:
    python run_conjoint_coverage.py
    python run_conjoint_coverage.py --num_trials 30 --n_real 100
"""

import numpy as np
import pickle as pkl
import argparse
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

# Import DML functions from main module
from dml import run_dml, flatten_all

# Ground truth parameters
GROUND_TRUTH_PARAMS = np.array([
    0.36310104, 0.7465673, 0.32377172, -0.21252407, 0.08090729,
    -0.09540857, -0.40639496, -0.15332593, -0.24158926, 0.17760716, -0.04599298
])


def calculate_mape(estimated, true):
    """Calculate Mean Absolute Percentage Error.
    Uses formula: mean(|estimated - true| / (|true| + 1)) × 100
    The +1 in denominator handles near-zero true values.
    """
    return np.mean(np.abs(estimated - true) / (np.abs(true) + 1)) * 100


def run_dml_with_se(X_all, y_real, y_aug, real_rows, aug_rows, n_folds=5, seed=0):
    """
    Run DML with proper sandwich SE estimation using UNCLIPPED tau values.

    The key insight: clipping tau to [0,1] destroys variance information.
    For SE computation, we use unclipped tau to capture the true uncertainty.

    Returns beta and standard errors.
    """
    from dml import train_g_stratified, predict_g

    # Extract data subsets
    X_p = [X_all[i] for i in real_rows]
    y_p = np.array([y_real[i] for i in real_rows])
    z_p = np.array([y_aug[i] for i in real_rows])

    X_a = [X_all[i] for i in aug_rows]
    z_a = np.array([y_aug[i] for i in aug_rows])

    n_p, n_a = len(y_p), len(z_a)
    n_total = n_p + n_a
    e_constant = n_p / n_total

    # Class prior as fallback
    class_prior = np.mean(y_p)

    # Flatten features
    X_p_flat = flatten_all(X_p)
    X_a_flat = flatten_all(X_a)

    # Cross-fitting to get g predictions
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    g_proba_p = np.zeros(n_p)
    g_proba_a = np.zeros(n_a)
    a_fold_counts = np.zeros(n_a)

    for train_idx, val_idx in kf.split(X_p_flat):
        g_models = train_g_stratified(
            X_p_flat[train_idx],
            z_p[train_idx],
            y_p[train_idx]
        )
        g_proba_p[val_idx] = predict_g(X_p_flat[val_idx], z_p[val_idx], g_models, class_prior)
        g_proba_a += predict_g(X_a_flat, z_a, g_models, class_prior)
        a_fold_counts += 1

    g_proba_a = g_proba_a / a_fold_counts

    # Compute DML targets - UNCLIPPED for SE computation
    tau_p_unclipped = g_proba_p * (1 - 1/e_constant) + y_p / e_constant
    tau_a = g_proba_a
    tau_unclipped = np.concatenate([tau_p_unclipped, tau_a])

    # Get point estimate using the actual DML implementation (which clips internally)
    beta = run_dml(X_all, y_real, y_aug, real_rows, aug_rows, n_folds=n_folds)

    # Combined features
    X_combined = np.vstack([X_p_flat, X_a_flat])

    # Compute predictions and residuals using UNCLIPPED tau
    u = X_combined @ beta
    u = np.clip(u, -500, 500)
    p = 1 / (1 + np.exp(-u))
    residuals = tau_unclipped - p

    # Sandwich estimator with UNCLIPPED residuals
    psi = X_combined * residuals[:, np.newaxis]
    meat = psi.T @ psi / n_total

    W = p * (1 - p)
    bread = X_combined.T @ (X_combined * W[:, np.newaxis]) / n_total

    try:
        bread_inv = np.linalg.inv(bread)
        cov = bread_inv @ meat @ bread_inv.T / n_total
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)

    return beta, se


def run_primary_with_se(X_all, y_real, real_rows, seed=0):
    """
    Run Primary-only estimator with standard errors.
    """
    X_flat = flatten_all(X_all)
    X_real = X_flat[real_rows]
    y_real_arr = np.array([y_real[i] for i in real_rows])

    clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=10000, fit_intercept=False)
    clf.fit(X_real, y_real_arr)
    beta = clf.coef_.squeeze()

    # Standard errors via Fisher Information
    u = X_real @ beta
    p = 1 / (1 + np.exp(-np.clip(u, -500, 500)))
    W = np.diag(p * (1 - p))

    try:
        fisher_info = X_real.T @ W @ X_real
        cov_matrix = np.linalg.inv(fisher_info)
        se = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)

    return beta, se


def run_ppi_with_ci(X_all, y_real, y_aug, real_rows, aug_rows, alpha=0.05):
    """
    Run PPI++ with confidence intervals using ppi_py.
    """
    try:
        from ppi_py import ppi_logistic_ci
    except ImportError:
        return None, None, None, None

    X_flat = flatten_all(X_all)
    X_real = X_flat[real_rows]
    y_real_arr = np.array([y_real[i] for i in real_rows]).astype(float)
    z_real = np.array([y_aug[i] for i in real_rows]).astype(float)

    X_aug = X_flat[aug_rows]
    z_aug = np.array([y_aug[i] for i in aug_rows]).astype(float)

    optimizer_options = {"ftol": 1e-5, "gtol": 1e-5, "maxls": 10000, "maxiter": 10000}

    try:
        ci_lower, ci_upper = ppi_logistic_ci(
            X_real, y_real_arr, z_real,
            X_aug, z_aug,
            alpha=alpha,
            optimizer_options=optimizer_options
        )
        beta = (ci_lower + ci_upper) / 2
        z_crit = stats.norm.ppf(1 - alpha/2)
        se = (ci_upper - ci_lower) / (2 * z_crit)
        return beta, se, ci_lower, ci_upper
    except Exception as e:
        return None, None, None, None


def run_single_trial(data, n_real, n_aug, seed, alpha=0.05):
    """Run a single trial and return coverage indicators."""

    y_real = np.asarray(data['y'])
    y_aug = np.asarray(data['y_aug'])
    X_all = list(data['X'])

    # Truncate to minimum length (matching run_comparison.py)
    n_total = min(len(y_real), len(y_aug), len(X_all))
    y_real = y_real[:n_total]
    y_aug = y_aug[:n_total]
    X_all = X_all[:n_total]

    # Sample by participant (5 observations each) - matching run_comparison.py
    rng = np.random.RandomState(seed)
    n_max = min(n_real + n_aug, n_total)
    n_participants = n_max // 5

    participants = rng.choice(n_total // 5, size=n_participants, replace=False)
    rows = []
    for p in participants:
        rows.extend(range(p * 5, min((p + 1) * 5, n_total)))
    rows = rows[:n_max]

    real_rows = rows[:n_real]
    aug_rows = rows[n_real:n_max]

    z_crit = stats.norm.ppf(1 - alpha/2)
    true_theta = GROUND_TRUTH_PARAMS

    results = {}

    # DML with sandwich SE using unclipped tau
    try:
        dml_beta, dml_se = run_dml_with_se(X_all, y_real, y_aug, real_rows, aug_rows, seed=seed)
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

    # Primary
    try:
        primary_beta, primary_se = run_primary_with_se(X_all, y_real, real_rows, seed=seed)
        primary_ci_lower = primary_beta - z_crit * primary_se
        primary_ci_upper = primary_beta + z_crit * primary_se
        primary_coverage = (true_theta >= primary_ci_lower) & (true_theta <= primary_ci_upper)

        results['Primary'] = {
            'beta': primary_beta,
            'se': primary_se,
            'ci_lower': primary_ci_lower,
            'ci_upper': primary_ci_upper,
            'coverage': primary_coverage,
            'mape': calculate_mape(primary_beta, true_theta),
            'ci_width': 2 * z_crit * primary_se
        }
    except Exception as e:
        results['Primary'] = None

    # PPI++
    ppi_beta, ppi_se, ppi_ci_lower, ppi_ci_upper = run_ppi_with_ci(
        X_all, y_real, y_aug, real_rows, aug_rows, alpha=alpha
    )
    if ppi_beta is not None:
        ppi_coverage = (true_theta >= ppi_ci_lower) & (true_theta <= ppi_ci_upper)
        results['PPI++'] = {
            'beta': ppi_beta,
            'se': ppi_se,
            'ci_lower': ppi_ci_lower,
            'ci_upper': ppi_ci_upper,
            'coverage': ppi_coverage,
            'mape': calculate_mape(ppi_beta, true_theta),
            'ci_width': ppi_ci_upper - ppi_ci_lower
        }
    else:
        results['PPI++'] = None

    return results


def main():
    parser = argparse.ArgumentParser(description='Conjoint Coverage Probability Benchmark')
    parser.add_argument('--num_trials', type=int, default=30, help='Number of trials')
    parser.add_argument('--n_real', type=int, default=None, help='Number of labeled samples (if None, run all)')
    parser.add_argument('--n_aug', type=int, default=1000, help='Number of augmented samples')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--save_results', action='store_true', help='Save results to file')
    args = parser.parse_args()

    print("=" * 80)
    print("CONJOINT DATA BENCHMARK: COVERAGE PROBABILITY")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    with open("train_gpt-4o_11_1200.pkl", "rb") as f:
        data = pkl.load(f)[0]

    print(f"Ground truth β: {GROUND_TRUTH_PARAMS}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Augmented samples: {args.n_aug}")
    print(f"Confidence level: {100*(1-args.alpha):.0f}%")

    # Sample sizes
    if args.n_real is not None:
        n_values = [args.n_real]
    else:
        n_values = [50, 100, 150, 200]

    methods = ['DML', 'Primary', 'PPI++']
    n_coef = len(GROUND_TRUTH_PARAMS)

    # Store results
    all_results = {n: {method: {'coverage': [], 'mape': [], 'ci_width': []}
                       for method in methods} for n in n_values}

    for n_real in n_values:
        print(f"\n{'='*80}")
        print(f"Running n_real = {n_real}, n_aug = {args.n_aug}")
        print("=" * 80)

        for trial in range(args.num_trials):
            if (trial + 1) % 10 == 0:
                print(f"  Completed trial {trial + 1}/{args.num_trials}")

            trial_results = run_single_trial(data, n_real, args.n_aug, seed=trial, alpha=args.alpha)

            for method in methods:
                if trial_results[method] is not None:
                    all_results[n_real][method]['coverage'].append(trial_results[method]['coverage'])
                    all_results[n_real][method]['mape'].append(trial_results[method]['mape'])
                    all_results[n_real][method]['ci_width'].append(trial_results[method]['ci_width'])

        # Print results
        print(f"\n  Results for n={n_real}:")
        print(f"  {'Method':<10} | {'Cov(avg)':>8} | {'Cov(all)':>8} | {'MAPE':>8} | {'CI Width(avg)':>12}")
        print("  " + "-" * 60)

        for method in methods:
            if len(all_results[n_real][method]['coverage']) > 0:
                coverage_arr = np.array(all_results[n_real][method]['coverage'])
                mape_arr = np.array(all_results[n_real][method]['mape'])
                ci_width_arr = np.array(all_results[n_real][method]['ci_width'])

                cov_avg = coverage_arr.mean()  # Average across all coefficients and trials
                cov_all = coverage_arr.all(axis=1).mean()  # Joint coverage
                avg_mape = mape_arr.mean()
                avg_ci_width = ci_width_arr.mean()

                print(f"  {method:<10} | {cov_avg*100:7.1f}% | {cov_all*100:7.1f}% | {avg_mape:7.1f}% | {avg_ci_width:12.4f}")
            else:
                print(f"  {method:<10} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>12}")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: COVERAGE PROBABILITY (%) - Average across 11 coefficients")
    print("=" * 80)
    print(f"\nExpected coverage for {100*(1-args.alpha):.0f}% CI: {100*(1-args.alpha):.0f}%")

    print(f"\n{'Method':<10} |", end="")
    for n in n_values:
        print(f" n={n:>3} |", end="")
    print("   Avg")
    print("-" * (14 + 8 * len(n_values) + 8))

    for method in methods:
        print(f"{method:<10} |", end="")
        method_coverages = []
        for n in n_values:
            if len(all_results[n][method]['coverage']) > 0:
                coverage_arr = np.array(all_results[n][method]['coverage'])
                cov_avg = coverage_arr.mean() * 100
                method_coverages.append(cov_avg)
                print(f" {cov_avg:5.1f}% |", end="")
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
    for n in n_values:
        print(f" n={n:>3} |", end="")
    print("   Avg")
    print("-" * (14 + 8 * len(n_values) + 8))

    for method in methods:
        print(f"{method:<10} |", end="")
        method_mapes = []
        for n in n_values:
            if len(all_results[n][method]['mape']) > 0:
                avg_mape = np.mean(all_results[n][method]['mape'])
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
    print("AVERAGE CI WIDTH (across 11 coefficients)")
    print("=" * 80)

    print(f"\n{'Method':<10} |", end="")
    for n in n_values:
        print(f" n={n:>3} |", end="")
    print("   Avg")
    print("-" * (14 + 8 * len(n_values) + 8))

    for method in methods:
        print(f"{method:<10} |", end="")
        method_widths = []
        for n in n_values:
            if len(all_results[n][method]['ci_width']) > 0:
                avg_width = np.mean(all_results[n][method]['ci_width'])
                method_widths.append(avg_width)
                print(f" {avg_width:5.2f} |", end="")
            else:
                print(f"   N/A |", end="")
        if method_widths:
            print(f" {np.mean(method_widths):5.2f}")
        else:
            print("   N/A")

    # Save results
    if args.save_results:
        import os
        os.makedirs('res', exist_ok=True)
        save_path = f'res/conjoint_coverage_{args.num_trials}trials.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump({
                'all_results': all_results,
                'true_theta': GROUND_TRUTH_PARAMS,
                'n_values': n_values,
                'n_aug': args.n_aug,
                'alpha': args.alpha,
                'num_trials': args.num_trials
            }, f)
        print(f"\nResults saved to {save_path}")

    return all_results


if __name__ == "__main__":
    main()
