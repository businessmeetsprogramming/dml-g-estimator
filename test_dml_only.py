"""
Quick DML vs AAE test script
"""
import pickle as pkl
import numpy as np
from compare_correct import run_dml, run_aae, calculate_mape, GROUND_TRUTH_PARAMS
import gc

def main():
    print("=" * 60)
    print("DML vs AAE TEST")
    print("=" * 60)

    # Load data
    with open("train_gpt-4o_11_1200.pkl", "rb") as f:
        data = pkl.load(f)[0]

    y_real = np.asarray(data['y'])
    y_aug = np.asarray(data['y_aug'])
    X_all = list(data['X'])

    n_total = min(len(y_real), len(y_aug), len(X_all))
    y_real = y_real[:n_total]
    y_aug = y_aug[:n_total]
    X_all = X_all[:n_total]

    n_real_values = [50, 100, 150, 200]
    n_aug = 1000
    n_trials = 20  # More trials for better accuracy

    dml_results = {n: [] for n in n_real_values}
    aae_results = {n: [] for n in n_real_values}

    for n_real in n_real_values:
        print(f"\n--- n_real = {n_real} ---")

        for trial in range(n_trials):
            rng = np.random.RandomState(trial)
            n_max = min(n_real + n_aug, n_total)

            participants = rng.choice(int(n_total/5), size=int(n_max/5), replace=False)
            rows = []
            for j in participants:
                rows += list(range(j*5, min((j+1)*5, n_total)))
            rows = rows[:n_max]

            real_rows = rows[:n_real]
            aug_rows = rows[n_real:n_max]

            # Run DML
            try:
                beta = run_dml(X_all, y_real, y_aug, real_rows, aug_rows)
                dml_results[n_real].append(calculate_mape(beta, GROUND_TRUTH_PARAMS))
            except Exception as e:
                print(f"  DML Trial {trial} error: {e}")

            # Run AAE
            try:
                beta = run_aae(X_all, y_real, y_aug, real_rows, aug_rows)
                aae_results[n_real].append(calculate_mape(beta, GROUND_TRUTH_PARAMS))
            except Exception as e:
                print(f"  AAE Trial {trial} error: {e}")

            gc.collect()

        dml_mape = np.mean(dml_results[n_real])
        aae_mape = np.mean(aae_results[n_real])
        diff = dml_mape - aae_mape
        print(f"  DML: {dml_mape:.1f}% | AAE: {aae_mape:.1f}% | Diff: {diff:+.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'n':<6} | {'DML':<10} | {'AAE':<10} | {'Diff':<10}")
    print("-" * 45)

    dml_avg = []
    aae_avg = []
    for n in n_real_values:
        dml = np.mean(dml_results[n])
        aae = np.mean(aae_results[n])
        dml_avg.append(dml)
        aae_avg.append(aae)
        print(f"{n:<6} | {dml:<10.1f} | {aae:<10.1f} | {dml-aae:+10.1f}")

    print("-" * 45)
    print(f"{'Avg':<6} | {np.mean(dml_avg):<10.1f} | {np.mean(aae_avg):<10.1f} | {np.mean(dml_avg)-np.mean(aae_avg):+10.1f}")

if __name__ == "__main__":
    main()
