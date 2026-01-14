"""
EXTREME OPTIMIZATION: Push for maximum accuracy
Based on findings:
1. Diff features are much better than simple features
2. Simple models (NB, LR) work well with diff features
3. Calibration helps
4. Need more diversity in ensemble
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
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

# BEST feature preparation: diff features
def prepare_diff_features(X_list, z):
    X_arr = np.array(X_list)
    alt0 = X_arr[:, 0, 1:]  # Drop constant feature 0
    alt1 = X_arr[:, 1, 1:]
    diff = alt0 - alt1
    z_oh = np.zeros((len(z), 3))
    for i, v in enumerate(z):
        z_oh[i, v + 1] = 1
    return np.hstack([diff, z_oh])

def prepare_simple_features(X_list, z):
    X_arr = np.array(X_list)
    alt0 = X_arr[:, 0, 1:]
    alt1 = X_arr[:, 1, 1:]
    z_oh = np.zeros((len(z), 3))
    for i, v in enumerate(z):
        z_oh[i, v + 1] = 1
    return np.hstack([alt0, alt1, z_oh])

def prepare_enhanced_diff_features(X_list, z):
    """Enhanced diff features with more signal"""
    X_arr = np.array(X_list)
    alt0 = X_arr[:, 0, 1:]
    alt1 = X_arr[:, 1, 1:]
    diff = alt0 - alt1
    abs_diff = np.abs(diff)
    product = alt0 * alt1

    # Sum features
    diff_sum = diff.sum(axis=1, keepdims=True)
    alt0_sum = alt0.sum(axis=1, keepdims=True)
    alt1_sum = alt1.sum(axis=1, keepdims=True)

    # Dominance features
    dom_count = (diff > 0).sum(axis=1, keepdims=True)
    anti_dom_count = (diff < 0).sum(axis=1, keepdims=True)

    # z features
    z_oh = np.zeros((len(z), 3))
    for i, v in enumerate(z):
        z_oh[i, v + 1] = 1
    z_ord = z.reshape(-1, 1)

    # Key interactions (top features: 1, 2, 6, 8, 9)
    # Pairwise products of top diff features
    top_idx = [1, 2, 6, 8, 9]
    interactions = []
    for i in range(len(top_idx)):
        for j in range(i+1, len(top_idx)):
            interactions.append((diff[:, top_idx[i]] * diff[:, top_idx[j]]).reshape(-1, 1))

    return np.hstack([
        diff, abs_diff,  # 22 features
        diff_sum, alt0_sum, alt1_sum, dom_count, anti_dom_count,  # 5 features
        z_oh, z_ord,  # 4 features
    ] + interactions)  # 10 interactions

# AAE baseline
def get_aae(seed=42):
    return VotingClassifier([
        ("hgb", HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=seed)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)),
        ("et", ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=seed)),
        ("lr", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000, random_state=seed))])),
    ], voting="soft")

# MEGA ENSEMBLE: Combine many diverse models
def get_mega_ensemble(seed=42):
    """Maximum diversity ensemble"""
    return VotingClassifier([
        # Naive Bayes variants
        ("gnb", GaussianNB()),
        ("bnb", BernoulliNB()),

        # Linear models
        ("lr1", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))])),
        ("lr2", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.1, max_iter=1000, random_state=seed))])),
        ("lr3", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=10.0, max_iter=1000, random_state=seed))])),
        ("lda", LinearDiscriminantAnalysis()),

        # Tree-based
        ("rf1", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)),
        ("rf2", RandomForestClassifier(n_estimators=200, max_depth=3, random_state=seed+1)),
        ("et", ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=seed)),
        ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)),
        ("hgb", HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=seed)),
        ("ada", AdaBoostClassifier(n_estimators=50, random_state=seed)),

        # Instance-based
        ("knn5", KNeighborsClassifier(n_neighbors=5)),
        ("knn10", KNeighborsClassifier(n_neighbors=10)),
        ("knn20", KNeighborsClassifier(n_neighbors=20)),

        # SVM variants
        ("svm_rbf", Pipeline([("s", StandardScaler()), ("c", SVC(kernel='rbf', probability=True, random_state=seed))])),
        ("svm_lin", Pipeline([("s", StandardScaler()), ("c", SVC(kernel='linear', probability=True, random_state=seed))])),
    ], voting="soft", weights=[
        2.0, 1.5,  # NB (strong with diff)
        2.5, 2.0, 1.5, 1.5,  # Linear (strong with diff)
        1.5, 1.0, 1.5, 1.5, 1.0, 1.0,  # Tree
        1.0, 1.0, 0.8,  # KNN
        1.5, 1.0,  # SVM
    ])

# STACKED MEGA ENSEMBLE
def get_stacked_mega(seed=42):
    """Two-level stacking with diverse base models"""
    base = [
        ("gnb", GaussianNB()),
        ("bnb", BernoulliNB()),
        ("lr", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))])),
        ("lda", LinearDiscriminantAnalysis()),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)),
        ("et", ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=seed)),
        ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)),
        ("svm", Pipeline([("s", StandardScaler()), ("c", SVC(probability=True, random_state=seed))])),
        ("knn", KNeighborsClassifier(n_neighbors=10)),
    ]
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=5,
        stack_method='predict_proba',
    )

# CALIBRATED MEGA ENSEMBLE
def get_calibrated_mega(seed=42):
    """All calibrated classifiers"""
    return VotingClassifier([
        ("gnb", CalibratedClassifierCV(GaussianNB(), cv=3)),
        ("bnb", CalibratedClassifierCV(BernoulliNB(), cv=3)),
        ("lr", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))])),
        ("lda", CalibratedClassifierCV(LinearDiscriminantAnalysis(), cv=3)),
        ("rf", CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed), cv=3)),
        ("et", CalibratedClassifierCV(ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=seed), cv=3)),
        ("gb", CalibratedClassifierCV(GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed), cv=3)),
        ("knn", CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=10), cv=3)),
    ], voting="soft", weights=[2.0, 1.5, 2.5, 1.5, 1.5, 1.5, 1.5, 1.0])

# OPTIMAL SIMPLE
def get_optimal_simple(seed=42):
    """Optimal based on analysis: simple models with diff features"""
    return VotingClassifier([
        ("gnb", GaussianNB()),
        ("lr1", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))])),
        ("lr2", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.5, max_iter=1000, random_state=seed))])),
        ("lda", LinearDiscriminantAnalysis()),
    ], voting="soft", weights=[2.0, 2.5, 2.0, 1.5])

# BAGGED LDA
def get_bagged_lda(seed=42):
    """Bagged LDA - stable linear classifier"""
    return BaggingClassifier(
        estimator=LinearDiscriminantAnalysis(),
        n_estimators=20,
        random_state=seed,
        bootstrap=True,
    )

# BAGGED LOGISTIC
def get_bagged_lr(seed=42):
    """Bagged Logistic Regression"""
    return BaggingClassifier(
        estimator=Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))]),
        n_estimators=20,
        random_state=seed,
        bootstrap=True,
    )

# EVALUATION
def run_evaluation(sample_sizes=[50, 100, 200, 400, 800], test_size=200, n_trials=10, seed=42):
    print(f"\n{'='*100}")
    print("EXTREME OPTIMIZATION: Diff Features + Mega Ensembles")
    print(f"{'='*100}")

    models = [
        ("AAE (baseline)", "simple", get_aae),
        ("MegaEnsemble", "diff", get_mega_ensemble),
        ("StackedMega", "diff", get_stacked_mega),
        ("CalibratedMega", "diff", get_calibrated_mega),
        ("OptimalSimple", "diff", get_optimal_simple),
        ("BaggedLDA", "diff", get_bagged_lda),
        ("BaggedLR", "diff", get_bagged_lr),
        ("EnhancedDiff", "enhanced", get_stacked_mega),  # Use enhanced features
    ]

    results = {n: {m[0]: [] for m in models} for n in sample_sizes}

    for trial in range(n_trials):
        trial_seed = seed + trial
        rng = np.random.RandomState(trial_seed)

        all_indices = rng.permutation(n_total)
        test_indices = all_indices[-test_size:]
        train_pool = all_indices[:-test_size]

        X_test_list = [X_list[i] for i in test_indices]
        y_test = y_all[test_indices]
        z_test = z_all[test_indices]

        X_test_simple = prepare_simple_features(X_test_list, z_test)
        X_test_diff = prepare_diff_features(X_test_list, z_test)
        X_test_enhanced = prepare_enhanced_diff_features(X_test_list, z_test)

        for n_train in sample_sizes:
            train_indices = train_pool[:n_train]

            X_train_list = [X_list[i] for i in train_indices]
            y_train = y_all[train_indices]
            z_train = z_all[train_indices]

            X_train_simple = prepare_simple_features(X_train_list, z_train)
            X_train_diff = prepare_diff_features(X_train_list, z_train)
            X_train_enhanced = prepare_enhanced_diff_features(X_train_list, z_train)

            for model_name, feat_type, model_fn in models:
                try:
                    model = model_fn(trial_seed)

                    if feat_type == "simple":
                        model.fit(X_train_simple, y_train)
                        pred = model.predict(X_test_simple)
                    elif feat_type == "diff":
                        model.fit(X_train_diff, y_train)
                        pred = model.predict(X_test_diff)
                    else:  # enhanced
                        model.fit(X_train_enhanced, y_train)
                        pred = model.predict(X_test_enhanced)

                    results[n_train][model_name].append(accuracy_score(y_test, pred))
                except Exception as e:
                    print(f"Error {model_name} n={n_train}: {e}")

        print(f"Trial {trial+1}/{n_trials}")

    # Results
    print(f"\n{'='*130}")
    print("RESULTS")
    print(f"{'='*130}")

    header = f"{'Model':<20}"
    for n in sample_sizes:
        header += f" | n={n:>4}"
    header += " | Avg Imp | Wins"
    print(header)
    print("-" * 130)

    final_results = {}
    for model_name, _, _ in models:
        row = f"{model_name:<20}"
        improvements = []
        wins = 0
        for n in sample_sizes:
            accs = results[n][model_name]
            mean_acc = np.mean(accs) if accs else 0
            final_results[(model_name, n)] = mean_acc

            aae_mean = np.mean(results[n]["AAE (baseline)"]) if results[n]["AAE (baseline)"] else 0

            if model_name != "AAE (baseline)":
                imp = mean_acc - aae_mean
                improvements.append(imp)
                if mean_acc > aae_mean:
                    wins += 1

            row += f" | {mean_acc:.4f}"

        if model_name != "AAE (baseline)":
            avg_imp = np.mean(improvements) if improvements else 0
            row += f" | {avg_imp:+.4f} | {wins}/{len(sample_sizes)}"
        print(row)

    # Best per size
    print(f"\n{'='*80}")
    print("BEST MODEL PER SAMPLE SIZE")
    print(f"{'='*80}")

    total_wins = 0
    for n in sample_sizes:
        aae_acc = final_results[("AAE (baseline)", n)]
        best = max([(m, final_results[(m, n)]) for m, _, _ in models if m != "AAE (baseline)"],
                   key=lambda x: x[1])
        margin = best[1] - aae_acc
        status = "✓" if best[1] > aae_acc else "✗"
        if best[1] > aae_acc:
            total_wins += 1
        print(f"  n={n:>3}: {best[0]:<20} ({best[1]:.4f}) vs AAE ({aae_acc:.4f}) [{margin:+.4f}] {status}")

    print(f"\n*** Beat AAE in {total_wins}/{len(sample_sizes)} ***")

    return results

if __name__ == "__main__":
    run_evaluation(n_trials=10, seed=42)
