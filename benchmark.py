"""
Benchmark: BestGEstimator vs AAE
================================
Compare the optimized BestGEstimator with the original AAE baseline.
"""
import pickle as pkl
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from best_model import BestGEstimator, prepare_features, get_diff_features, one_hot_z


# =============================================================================
# AAE BASELINE MODEL
# =============================================================================

def prepare_aae_features(X_list, z):
    """AAE features: alt0 + alt1 + z (one-hot)."""
    X_arr = np.array(X_list)
    alt0 = X_arr[:, 0, 1:]  # Drop constant feature 0
    alt1 = X_arr[:, 1, 1:]
    z_oh = np.zeros((len(z), 3))
    for i, v in enumerate(z):
        z_oh[i, v + 1] = 1
    return np.hstack([alt0, alt1, z_oh])


def get_aae_model(seed=42):
    """Original AAE baseline model."""
    return VotingClassifier([
        ("hgb", HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=seed)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1)),
        ("et", ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1)),
        ("lr", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000, random_state=seed))])),
    ], voting="soft", n_jobs=-1)


# =============================================================================
# BENCHMARK
# =============================================================================

def run_benchmark(data_path="train_gpt-4o_11_1200.pkl",
                  sample_sizes=[50, 100, 200, 400, 800],
                  n_splits=30, test_size=200, seed=42):
    """
    Run benchmark comparing BestGEstimator vs AAE.
    """
    print("Loading data...")
    with open(data_path, "rb") as f:
        data = pkl.load(f)[0]

    X_list = list(data["X"])
    y_all = np.asarray(data["y"], dtype=int)
    z_all = np.asarray(data["y_aug"], dtype=int)

    n_total = min(len(X_list), len(y_all), len(z_all))
    X_list = X_list[:n_total]
    y_all = y_all[:n_total]
    z_all = z_all[:n_total]

    print(f"Total samples: {n_total}")
    print(f"Test size: {test_size}, Splits: {n_splits}")

    results = {
        "AAE": {n: [] for n in sample_sizes},
        "BestGEstimator": {n: [] for n in sample_sizes}
    }

    for i in range(n_splits):
        split_seed = seed + i * 17
        rng = np.random.RandomState(split_seed)

        indices = rng.permutation(n_total)
        test_idx = indices[-test_size:]
        train_pool = indices[:-test_size]

        # Test data
        X_test = [X_list[j] for j in test_idx]
        z_test = z_all[test_idx]
        y_test = y_all[test_idx]

        # AAE test features
        X_test_aae = prepare_aae_features(X_test, z_test)

        for n_train in sample_sizes:
            train_idx = train_pool[:n_train]

            X_train = [X_list[j] for j in train_idx]
            z_train = z_all[train_idx]
            y_train = y_all[train_idx]

            # --- AAE Baseline ---
            X_train_aae = prepare_aae_features(X_train, z_train)
            aae_model = get_aae_model(split_seed)
            aae_model.fit(X_train_aae, y_train)
            aae_pred = aae_model.predict(X_test_aae)
            results["AAE"][n_train].append(accuracy_score(y_test, aae_pred))

            # --- BestGEstimator ---
            best_model = BestGEstimator(cleanlab_pct=30, seed=split_seed)
            best_model.fit(X_train, z_train, y_train)
            best_pred = best_model.predict(X_test, z_test)
            results["BestGEstimator"][n_train].append(accuracy_score(y_test, best_pred))

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_splits} splits")

    return results


def print_results(results, sample_sizes=[50, 100, 200, 400, 800]):
    """Print benchmark results in a nice table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS: BestGEstimator vs AAE")
    print("=" * 80)

    print("\n| Model           | n=50   | n=100  | n=200  | n=400  | n=800  | Avg    |")
    print("|-----------------|--------|--------|--------|--------|--------|--------|")

    for model_name in ["AAE", "BestGEstimator"]:
        row = f"| {model_name:<15} "
        means = []
        for n in sample_sizes:
            m = np.mean(results[model_name][n])
            s = np.std(results[model_name][n])
            means.append(m)
            row += f"| {m*100:.2f}% "
        avg = np.mean(means)
        row += f"| {avg*100:.2f}% |"
        print(row)

    # Improvements
    print("\n--- Improvement (BestGEstimator - AAE) ---")
    print("\n| Metric          | n=50   | n=100  | n=200  | n=400  | n=800  | Avg    |")
    print("|-----------------|--------|--------|--------|--------|--------|--------|")

    row = "| Improvement     "
    improvements = []
    for n in sample_sizes:
        best_mean = np.mean(results["BestGEstimator"][n])
        aae_mean = np.mean(results["AAE"][n])
        imp = best_mean - aae_mean
        improvements.append(imp)
        sign = "+" if imp >= 0 else ""
        row += f"| {sign}{imp*100:.2f}% "
    avg_imp = np.mean(improvements)
    sign = "+" if avg_imp >= 0 else ""
    row += f"| {sign}{avg_imp*100:.2f}% |"
    print(row)

    # Statistical significance
    print("\n--- Statistical Summary ---")
    wins = 0
    for n in sample_sizes:
        best_mean = np.mean(results["BestGEstimator"][n])
        aae_mean = np.mean(results["AAE"][n])
        best_std = np.std(results["BestGEstimator"][n])
        aae_std = np.std(results["AAE"][n])

        if best_mean > aae_mean:
            wins += 1
            status = "WIN"
        else:
            status = "LOSS"

        print(f"  n={n}: BestG {best_mean*100:.2f}% (±{best_std*100:.2f}) vs AAE {aae_mean*100:.2f}% (±{aae_std*100:.2f}) [{status}]")

    print(f"\n*** BestGEstimator wins in {wins}/{len(sample_sizes)} sample sizes ***")

    return improvements


if __name__ == "__main__":
    results = run_benchmark()
    print_results(results)
