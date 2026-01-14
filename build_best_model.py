"""
Build the BEST model based on data analysis findings:
1. Use DIFFERENCE features (alt0 - alt1) - much more predictive!
2. Use best models: Naive Bayes, Logistic Regression, RF, ET
3. Create super-ensemble with diff features
"""
import pickle as pkl
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

# Load data
with open("train_gpt-4o_11_1200.pkl", "rb") as f:
    data = pkl.load(f)[0]

X_list = list(data["X"])
y_all = np.asarray(data["y"], dtype=int)
z_all = np.asarray(data["y_aug"], dtype=int)

n_total = min(len(X_list), len(y_all), len(z_all))
X_list = X_list[:n_total]
y_all = y_all[:n_total]
z_all = z_all[:n_total]

print(f"Total samples: {n_total}")
print(f"y distribution: {Counter(y_all)}")
print(f"Baseline (majority): {max(Counter(y_all).values())/n_total*100:.1f}%")

# Prepare features
def prepare_simple_features(X_list, z):
    """Original AAE features: alt0 + alt1 + z (one-hot)"""
    X_arr = np.array(X_list)
    alt0 = X_arr[:, 0, 1:]  # Drop constant feature 0
    alt1 = X_arr[:, 1, 1:]
    z_oh = np.zeros((len(z), 3))
    for i, v in enumerate(z):
        z_oh[i, v + 1] = 1
    return np.hstack([alt0, alt1, z_oh])

def prepare_diff_features(X_list, z):
    """BEST features: diff (alt0 - alt1) + z"""
    X_arr = np.array(X_list)
    alt0 = X_arr[:, 0, 1:]  # Drop constant feature 0
    alt1 = X_arr[:, 1, 1:]
    diff = alt0 - alt1
    z_oh = np.zeros((len(z), 3))
    for i, v in enumerate(z):
        z_oh[i, v + 1] = 1
    return np.hstack([diff, z_oh])

def prepare_top_diff_features(X_list, z):
    """Top diff features only (2,7,1,9,10) + z"""
    X_arr = np.array(X_list)
    alt0 = X_arr[:, 0, 1:]  # Drop constant feature 0
    alt1 = X_arr[:, 1, 1:]
    diff = alt0 - alt1
    # Top features: indices 1, 2, 6, 8, 9 (0-indexed from diff)
    # These correspond to original features 2, 3, 7, 9, 10 (1-indexed)
    top_diff = diff[:, [1, 2, 6, 8, 9]]
    z_oh = np.zeros((len(z), 3))
    for i, v in enumerate(z):
        z_oh[i, v + 1] = 1
    return np.hstack([top_diff, z_oh])

def prepare_diff_full_features(X_list, z):
    """Diff + additional derived features + z"""
    X_arr = np.array(X_list)
    alt0 = X_arr[:, 0, 1:]
    alt1 = X_arr[:, 1, 1:]
    diff = alt0 - alt1
    abs_diff = np.abs(diff)

    # Aggregates
    diff_sum = diff.sum(axis=1, keepdims=True)
    alt0_sum = alt0.sum(axis=1, keepdims=True)
    alt1_sum = alt1.sum(axis=1, keepdims=True)
    dominance = (diff > 0).sum(axis=1, keepdims=True)

    z_oh = np.zeros((len(z), 3))
    for i, v in enumerate(z):
        z_oh[i, v + 1] = 1

    return np.hstack([diff, abs_diff, diff_sum, alt0_sum, alt1_sum, dominance, z_oh])

# Models
def get_aae_model(seed=42):
    """Original AAE - uses simple features"""
    return VotingClassifier([
        ("hgb", HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=seed)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)),
        ("et", ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=seed)),
        ("lr", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000, random_state=seed))])),
    ], voting="soft")

def get_diff_voting_model(seed=42):
    """Voting with BEST models for diff features"""
    return VotingClassifier([
        ("nb", GaussianNB()),
        ("lr", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000, random_state=seed))])),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)),
        ("et", ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=seed)),
    ], voting="soft")

def get_diff_stacking_model(seed=42):
    """Stacking with BEST models for diff features"""
    return StackingClassifier([
        ("nb", GaussianNB()),
        ("lr", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000, random_state=seed))])),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)),
        ("et", ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=seed)),
        ("svm", Pipeline([("s", StandardScaler()), ("c", SVC(probability=True, random_state=seed))])),
    ], final_estimator=LogisticRegression(C=0.5), cv=3)

def get_nb_ensemble(seed=42):
    """Ensemble focused on Naive Bayes variants"""
    return VotingClassifier([
        ("gnb", GaussianNB()),
        ("bnb", BernoulliNB()),
        ("lr", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000, random_state=seed))])),
        ("knn", KNeighborsClassifier(n_neighbors=10)),
    ], voting="soft")

def get_super_ensemble(seed=42):
    """Super ensemble with all best models"""
    return VotingClassifier([
        ("gnb", GaussianNB()),
        ("bnb", BernoulliNB()),
        ("lr1", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))])),
        ("lr2", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=1000, random_state=seed))])),
        ("rf", RandomForestClassifier(n_estimators=150, max_depth=5, random_state=seed)),
        ("et", ExtraTreesClassifier(n_estimators=150, max_depth=5, random_state=seed)),
        ("svm", Pipeline([("s", StandardScaler()), ("c", SVC(probability=True, random_state=seed))])),
        ("knn", KNeighborsClassifier(n_neighbors=10)),
    ], voting="soft", weights=[2.0, 1.5, 2.0, 1.5, 1.5, 1.5, 1.5, 1.0])

def get_calibrated_ensemble(seed=42):
    """Ensemble with calibrated classifiers"""
    return VotingClassifier([
        ("gnb_cal", CalibratedClassifierCV(GaussianNB(), cv=3)),
        ("lr", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000, random_state=seed))])),
        ("rf_cal", CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed), cv=3)),
        ("et_cal", CalibratedClassifierCV(ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=seed), cv=3)),
    ], voting="soft")

# Evaluation
def run_evaluation(sample_sizes=[50, 100, 200, 400, 800], test_size=200, n_trials=10, seed=42):
    """
    Run evaluation with proper protocol:
    - Fixed held-out test set
    - Multiple trials with different random splits
    """
    print(f"\n{'='*90}")
    print("EVALUATION: DIFF FEATURES vs SIMPLE FEATURES (AAE)")
    print(f"{'='*90}")
    print(f"Test size: {test_size}, Trials: {n_trials}")

    # Models with their feature types
    models = [
        ("AAE (simple feat)", "simple", get_aae_model),
        ("NaiveBayes (diff)", "diff", lambda s: GaussianNB()),
        ("LogReg (diff)", "diff", lambda s: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000, random_state=s))])),
        ("RF (diff)", "diff", lambda s: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=s)),
        ("DiffVoting", "diff", get_diff_voting_model),
        ("DiffStacking", "diff", get_diff_stacking_model),
        ("NB_Ensemble", "diff", get_nb_ensemble),
        ("SuperEnsemble", "diff", get_super_ensemble),
        ("Calibrated", "diff", get_calibrated_ensemble),
    ]

    results = {n: {m[0]: [] for m in models} for n in sample_sizes}

    for trial in range(n_trials):
        trial_seed = seed + trial
        rng = np.random.RandomState(trial_seed)

        # Split data
        all_indices = rng.permutation(n_total)
        test_indices = all_indices[-test_size:]
        train_pool = all_indices[:-test_size]

        # Prepare test features
        X_test_list = [X_list[i] for i in test_indices]
        y_test = y_all[test_indices]
        z_test = z_all[test_indices]

        X_test_simple = prepare_simple_features(X_test_list, z_test)
        X_test_diff = prepare_diff_features(X_test_list, z_test)

        for n_train in sample_sizes:
            train_indices = train_pool[:n_train]

            X_train_list = [X_list[i] for i in train_indices]
            y_train = y_all[train_indices]
            z_train = z_all[train_indices]

            X_train_simple = prepare_simple_features(X_train_list, z_train)
            X_train_diff = prepare_diff_features(X_train_list, z_train)

            for model_name, feat_type, model_fn in models:
                try:
                    model = model_fn(trial_seed)

                    if feat_type == "simple":
                        model.fit(X_train_simple, y_train)
                        pred = model.predict(X_test_simple)
                    else:
                        model.fit(X_train_diff, y_train)
                        pred = model.predict(X_test_diff)

                    acc = accuracy_score(y_test, pred)
                    results[n_train][model_name].append(acc)
                except Exception as e:
                    print(f"Error with {model_name} at n={n_train}: {e}")

        print(f"Trial {trial+1}/{n_trials} completed")

    # Print results
    print(f"\n{'='*100}")
    print("RESULTS")
    print(f"{'='*100}")

    for n_train in sample_sizes:
        print(f"\n--- n = {n_train} ---")
        aae_mean = np.mean(results[n_train]["AAE (simple feat)"])

        for model_name, _, _ in models:
            accs = results[n_train][model_name]
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            diff = mean_acc - aae_mean if model_name != "AAE (simple feat)" else 0
            diff_str = f"[{'+' if diff >= 0 else ''}{diff:.4f}]" if diff != 0 else ""
            print(f"  {model_name:<20}: {mean_acc:.4f} (+/- {std_acc:.4f}) {diff_str}")

    # Summary table
    print(f"\n{'='*120}")
    print("SUMMARY TABLE")
    print(f"{'='*120}")

    header = f"{'Model':<20}"
    for n in sample_sizes:
        header += f" | n={n:>4}"
    header += " | Wins vs AAE"
    print(header)
    print("-" * 120)

    final_results = {}
    for model_name, _, _ in models:
        row = f"{model_name:<20}"
        wins = 0
        for n in sample_sizes:
            mean_acc = np.mean(results[n][model_name])
            final_results[(model_name, n)] = mean_acc
            aae_mean = np.mean(results[n]["AAE (simple feat)"])
            if model_name != "AAE (simple feat)" and mean_acc > aae_mean:
                wins += 1
            row += f" | {mean_acc:.4f}"
        if model_name != "AAE (simple feat)":
            row += f" | {wins}/{len(sample_sizes)}"
        print(row)

    # Best model per size
    print(f"\n{'='*80}")
    print("BEST MODEL PER SAMPLE SIZE")
    print(f"{'='*80}")

    total_wins = 0
    for n in sample_sizes:
        aae_acc = final_results[("AAE (simple feat)", n)]
        best_model = None
        best_acc = 0
        for model_name, _, _ in models:
            if model_name != "AAE (simple feat)":
                acc = final_results[(model_name, n)]
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_name

        beats = best_acc > aae_acc
        if beats:
            total_wins += 1
        margin = best_acc - aae_acc
        status = "✓" if beats else "✗"
        print(f"  n={n:>3}: {best_model:<20} ({best_acc:.4f}) vs AAE ({aae_acc:.4f}) [{margin:+.4f}] {status}")

    print(f"\n*** Best models beat AAE in {total_wins}/{len(sample_sizes)} sample sizes ***")

    # Calculate average improvement
    print(f"\n{'='*80}")
    print("AVERAGE IMPROVEMENT OVER AAE")
    print(f"{'='*80}")

    for model_name, _, _ in models:
        if model_name == "AAE (simple feat)":
            continue
        improvements = []
        for n in sample_sizes:
            aae_acc = final_results[("AAE (simple feat)", n)]
            model_acc = final_results[(model_name, n)]
            improvements.append(model_acc - aae_acc)
        avg_imp = np.mean(improvements)
        print(f"  {model_name:<20}: {avg_imp:+.4f} ({avg_imp*100:+.2f}%)")

    return results

if __name__ == "__main__":
    results = run_evaluation(
        sample_sizes=[50, 100, 200, 400, 800],
        test_size=200,
        n_trials=10,
        seed=42
    )
