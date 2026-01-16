#!/usr/bin/env python3
"""
Run DML and save results in standardized format.

Saves .pkl files compatible with the analysis pipeline.

Usage:
    python run_dml.py --n_trials 30 --n_real 50 --method gpt-4o
    python run_dml.py --n_trials 30 --n_real 100 --method gpt-4o
"""

import os
import argparse
import pickle as pkl
import numpy as np
from dml import run_dml, run_dml2, run_primary_only


def main(dx: int, n_samples: int, n_trials: int, n_real: int, method: str,
         n_folds: int, clip_eps: float, use_dml2: bool = False):
    """
    Run DML experiments and save results.

    Saves results for:
    - n_aug = 0 (Primary Only baseline)
    - n_aug = n_max_aug (Full DML with augmented data)

    Output format matches the standard analysis pipeline.
    """
    os.makedirs("res2", exist_ok=True)

    # Load data file
    data_file = f"data/train_{method}_{dx}_1200.pkl"
    if not os.path.exists(data_file):
        # Fallback to current directory
        data_file = f"train_{method}_{dx}_1200.pkl"

    print(f"Loading data from {data_file}...")
    with open(data_file, "rb") as f:
        data = pkl.load(f)[0]

    y_real = np.asarray(data["y"])
    y_aug = np.asarray(data["y_aug"])
    X_all = list(data["X"])

    # Configuration matching debias.py
    n_max_aug = 1000
    step_aug = 1000  # Only save at 0 and n_max_aug
    n_max = n_real + n_max_aug

    # Global RNG stream (not per-trial) to match debias.py
    rng = np.random.RandomState(0)

    results = {
        "n_real_list": [],
        "n_aug_list": [],
        "sample_id_list": [],
        "params_list": [],
    }

    dml_func = run_dml2 if use_dml2 else run_dml
    dml_name = "DML-2" if use_dml2 else "DML"

    print(f"\nRunning {n_trials} trials with n_real={n_real}, n_max_aug={n_max_aug}")
    print(f"Method: {dml_name}")
    print("-" * 50)

    for sid in range(n_trials):
        # Match debias.py sampling path exactly
        # Sample by participants (5 observations each)
        participants = rng.choice(
            int(n_max / 5),
            size=int((n_real + n_max_aug) / 5),
            replace=False,
        )
        rows = []
        for j in participants:
            rows += list(range(j * 5, (j * 5) + 5))
        rows = np.asarray(rows, dtype=int)

        real_r = rows[:n_real]

        # Primary-only entry (n_aug=0)
        # Uses the SAME real_r sample as the DML runs
        params_primary = run_primary_only(X_all, y_real, real_r.tolist())
        results["params_list"].append(params_primary)
        results["n_real_list"].append(n_real)
        results["n_aug_list"].append(0)
        results["sample_id_list"].append(sid)

        # DML for n_aug in {step_aug, 2*step_aug, ..., n_max_aug}
        for n_aug in np.arange(step_aug, n_max_aug + step_aug, step_aug):
            aug_r = rows[n_real:n_real + n_aug]

            beta = dml_func(
                X_all, y_real, y_aug,
                real_r.tolist(), aug_r.tolist(),
                n_folds=n_folds,
            )

            results["params_list"].append(beta)
            results["n_real_list"].append(n_real)
            results["n_aug_list"].append(len(aug_r))
            results["sample_id_list"].append(sid)

        if (sid + 1) % 10 == 0:
            print(f"  Completed trial {sid + 1}/{n_trials}")

    # Save results
    dml_suffix = "dml2" if use_dml2 else "dml"
    res_file = f"res2/{dml_suffix}_{method}_{n_real}_{n_max_aug}_{n_trials}.pkl"
    with open(res_file, "wb") as f:
        pkl.dump(results, f, -1)

    print(f"\nSaved: {res_file}")
    print(f"  - {n_trials} trials")
    print(f"  - {len(results['params_list'])} total parameter estimates")
    print(f"  - n_aug values: 0, {n_max_aug}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run DML experiments and save results")
    p.add_argument("--dx", type=int, default=11,
                   help="Number of features (default: 11)")
    p.add_argument("--n_samples", type=int, default=1200,
                   help="Total samples in data file (default: 1200)")
    p.add_argument("--n_trials", type=int, required=True,
                   help="Number of random trials")
    p.add_argument("--n_real", type=int, required=True,
                   help="Number of primary (human-labeled) samples")
    p.add_argument("--method", type=str, required=True,
                   help="AI method name (e.g., gpt-4o)")
    p.add_argument("--n_folds", type=int, default=5,
                   help="Number of cross-fitting folds (default: 5)")
    p.add_argument("--clip_epsilon", type=float, default=0.02,
                   help="Clipping epsilon (default: 0.02, currently unused)")
    p.add_argument("--dml2", action="store_true",
                   help="Use DML-2 (fold-wise beta averaging) instead of DML")

    args = p.parse_args()

    main(
        dx=args.dx,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        n_real=args.n_real,
        method=args.method,
        n_folds=args.n_folds,
        clip_eps=args.clip_epsilon,
        use_dml2=args.dml2,
    )
