#!/usr/bin/env python3
"""
Run DML Comparison
==================

Simple script to compare all estimation methods on the conjoint data.

Usage:
    python run_comparison.py [--n_trials 20] [--n_aug 1000]

Example:
    python run_comparison.py
    python run_comparison.py --n_trials 10 --n_aug 500
"""

import pickle as pkl
import numpy as np
import argparse
import gc
from dml import (
    run_dml, run_dml2, run_primary_only, run_naive, run_aae, run_ppi,
    calculate_mape, GROUND_TRUTH_PARAMS, PPI_AVAILABLE
)


def main():
    parser = argparse.ArgumentParser(description='Compare DML methods')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of random trials (default: 20)')
    parser.add_argument('--n_aug', type=int, default=1000,
                        help='Number of augmented samples (default: 1000)')
    parser.add_argument('--data_file', type=str, default='train_gpt-4o_11_1200.pkl',
                        help='Path to data file')
    args = parser.parse_args()

    print("=" * 70)
    print("DML METHOD COMPARISON")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {args.data_file}...")
    with open(args.data_file, "rb") as f:
        data = pkl.load(f)[0]

    y_real = np.asarray(data['y'])
    y_aug = np.asarray(data['y_aug'])
    X_all = list(data['X'])

    n_total = min(len(y_real), len(y_aug), len(X_all))
    y_real = y_real[:n_total]
    y_aug = y_aug[:n_total]
    X_all = X_all[:n_total]

    print(f"Total samples: {n_total}")
    print(f"Trials per condition: {args.n_trials}")
    print(f"Augmented samples: {args.n_aug}")

    # Sample sizes to test
    n_real_values = [50, 100, 150, 200]

    # Methods to compare
    methods = ['Primary', 'Naive', 'AAE', 'DML', 'DML-2']
    if PPI_AVAILABLE:
        methods.append('PPI')
    results = {m: {n: [] for n in n_real_values} for m in methods}

    # Run experiments
    for n_real in n_real_values:
        print(f"\n--- n_real = {n_real}, n_aug = {args.n_aug} ---")

        for trial in range(args.n_trials):
            # Random sampling by participants (5 observations each)
            rng = np.random.RandomState(trial)
            n_max = min(n_real + args.n_aug, n_total)
            n_participants = n_max // 5

            participants = rng.choice(n_total // 5, size=n_participants, replace=False)
            rows = []
            for p in participants:
                rows.extend(range(p * 5, min((p + 1) * 5, n_total)))
            rows = rows[:n_max]

            real_rows = rows[:n_real]
            aug_rows = rows[n_real:n_max]

            # Run each method
            for method in methods:
                try:
                    if method == 'Primary':
                        beta = run_primary_only(X_all, y_real, real_rows)
                    elif method == 'Naive':
                        beta = run_naive(X_all, y_real, y_aug, real_rows, aug_rows)
                    elif method == 'AAE':
                        beta = run_aae(X_all, y_real, y_aug, real_rows, aug_rows)
                    elif method == 'DML':
                        beta = run_dml(X_all, y_real, y_aug, real_rows, aug_rows)
                    elif method == 'DML-2':
                        beta = run_dml2(X_all, y_real, y_aug, real_rows, aug_rows)
                    elif method == 'PPI':
                        beta = run_ppi(X_all, y_real, y_aug, real_rows, aug_rows)
                        if beta is None:
                            continue

                    mape = calculate_mape(beta, GROUND_TRUTH_PARAMS)
                    if mape < 1000:  # Filter extreme values for PPI
                        results[method][n_real].append(mape)
                except Exception as e:
                    print(f"  Error in {method}, trial {trial}: {e}")

            gc.collect()

        # Print progress
        for method in methods:
            if results[method][n_real]:
                m = np.mean(results[method][n_real])
                s = np.std(results[method][n_real])
                print(f"  {method:<10}: {m:5.1f}% ± {s:4.1f}%")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS: MAPE (%) - Lower is Better")
    print("=" * 70)

    # Header
    print(f"\n{'Method':<12}", end="")
    for n in n_real_values:
        print(f" | n={n:<4}", end="")
    print(" |   Avg")
    print("-" * 65)

    # Results for each method
    for method in methods:
        print(f"{method:<12}", end="")
        mapes = []
        for n in n_real_values:
            if results[method][n]:
                m = np.mean(results[method][n])
                mapes.append(m)
                print(f" | {m:5.1f}", end="")
            else:
                print(f" |   N/A", end="")
        avg = np.mean(mapes) if mapes else float('nan')
        print(f" | {avg:5.1f}")

    # DML vs DML-2 comparison
    print("\n" + "-" * 65)
    print("DML vs DML-2 Difference:")
    print("-" * 65)

    for n in n_real_values:
        dml = np.mean(results['DML'][n]) if results['DML'][n] else float('nan')
        dml2 = np.mean(results['DML-2'][n]) if results['DML-2'][n] else float('nan')
        diff = dml - dml2
        print(f"  n={n}: DML={dml:.2f}%, DML-2={dml2:.2f}%, Diff={diff:+.2f}%")

    # Overall comparison
    dml_avg = np.mean([np.mean(results['DML'][n]) for n in n_real_values])
    dml2_avg = np.mean([np.mean(results['DML-2'][n]) for n in n_real_values])
    aae_avg = np.mean([np.mean(results['AAE'][n]) for n in n_real_values])

    print(f"\n  Average: DML={dml_avg:.2f}%, DML-2={dml2_avg:.2f}%, Diff={dml_avg-dml2_avg:+.2f}%")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
    - DML and DML-2 difference: {abs(dml_avg - dml2_avg):.2f}% (should be < 1%)
    - DML improvement over AAE: {aae_avg - dml_avg:+.2f}%
    - Both DML variants are asymptotically equivalent
    """)


if __name__ == "__main__":
    main()
