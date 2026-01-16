"""
Compare DML variants:
- DML: Original implementation (single β, cross-fitted nuisance)
- DML-2: Exact PDF algorithm (fold-wise β, then average)

Both use constant e = n_primary / n_total (as per Corollary 2.1 in PDF)
"""
import pickle as pkl
import numpy as np
import torch
from torch.optim import Adam
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from best_model import prepare_features
import gc
import warnings
warnings.filterwarnings("ignore")

GROUND_TRUTH_PARAMS = np.array([
    0.36310104, 0.7465673, 0.32377172, -0.21252407, 0.08090729,
    -0.09540857, -0.40639496, -0.15332593, -0.24158926, 0.17760716, -0.04599298
])

def flatten(X):
    return X[1][1:] - X[0][1:]

def flatten_full(X_list):
    return [flatten(X_list[i]) for i in range(len(X_list))]

def calculate_mape(estimated, true):
    return np.mean(np.abs(estimated - true) / (np.abs(true) + 1)) * 100

# =============================================================================
# DML-2: Exact PDF Algorithm with Fold-wise β Averaging
# =============================================================================

def run_dml2(X_all, y_real, y_aug, real_rows, aug_rows, n_folds=5):
    """
    DML-2: Exact implementation of PDF Algorithm (Section 2)

    Key differences from DML:
    1. Uses CONSTANT e = n_primary / n_total (Corollary 2.1 assumption)
    2. For each fold k:
       - Train g on primary data NOT in fold k
       - Solve for β_k using fold k's primary data + ALL augmented data
    3. Final β = (1/K) Σ β_k (average across folds)

    This follows the PDF exactly:
    - Step 1: Split into K folds
    - Step 2: For each fold, train nuisance on I_k^c, solve β_k on I_k
    - Step 3: β̂ = (1/K) Σ β̂_k
    """
    X_p = [X_all[row] for row in real_rows]
    y_p = np.array([y_real[row] for row in real_rows])
    z_p = np.array([y_aug[row] for row in real_rows])
    X_a = [X_all[row] for row in aug_rows]
    z_a = np.array([y_aug[row] for row in aug_rows])

    n_p = len(y_p)
    n_a = len(z_a)
    n_total = n_p + n_a

    # CONSTANT e = sampling probability (Corollary 2.1)
    # e(X,z) = ρ = P(w=1) = n_primary / n_total
    e_constant = n_p / n_total

    # Class prior for fallback
    class_prior = np.mean(y_p)

    # Prepare flattened features
    X_p_flat = np.array(flatten_full(X_p))
    X_a_flat = np.array(flatten_full(X_a))

    # K-fold split of PRIMARY data only
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    # Store β estimates from each fold
    beta_folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_p_flat)):
        # === STEP 2a: Train g on I_k^c (primary data NOT in fold k) ===
        X_train_g_flat = X_p_flat[train_idx]
        z_train_g = z_p[train_idx]
        y_train_g = y_p[train_idx]

        # Stratified g models (one per z value)
        g_models = {}
        for z_val in [-1, 0, 1]:
            mask = z_train_g == z_val
            if np.sum(mask) >= 2 and len(np.unique(y_train_g[mask])) == 2:
                clf = LogisticRegression(C=0.05, max_iter=2000, random_state=1)
                clf.fit(X_train_g_flat[mask], y_train_g[mask])
                g_models[z_val] = clf

        # === Predict g on fold k's primary data (out-of-fold) ===
        g_proba_fold_primary = np.zeros(len(val_idx))
        for i, idx in enumerate(val_idx):
            z_i = z_p[idx]
            X_i = X_p_flat[idx:idx+1]
            if z_i in g_models and g_models[z_i] is not None:
                g_proba_fold_primary[i] = g_models[z_i].predict_proba(X_i)[0, 1]
            else:
                g_proba_fold_primary[i] = class_prior

        # === Predict g on ALL augmented data (always out-of-sample for g) ===
        g_proba_aug = np.zeros(n_a)
        for j in range(n_a):
            z_j = z_a[j]
            X_j = X_a_flat[j:j+1]
            if z_j in g_models and g_models[z_j] is not None:
                g_proba_aug[j] = g_models[z_j].predict_proba(X_j)[0, 1]
            else:
                g_proba_aug[j] = class_prior

        # === Compute DML targets ===
        # For primary (w=1): τ = g(1 - 1/e) + y/e
        y_fold = y_p[val_idx]
        tau_primary = g_proba_fold_primary * (1 - 1/e_constant) + y_fold / e_constant
        tau_primary = np.clip(tau_primary, 0.0, 1.0)

        # For augmented (w=0): τ = g
        tau_aug = g_proba_aug

        # === STEP 2b: Solve for β_k on fold k data ===
        # Use fold k's primary + ALL augmented
        X_fold_primary = X_p_flat[val_idx]

        X_fold_tensor = torch.tensor(X_fold_primary, dtype=torch.float64)
        X_aug_tensor = torch.tensor(X_a_flat, dtype=torch.float64)
        tau_p_tensor = torch.tensor(tau_primary, dtype=torch.float64)
        tau_a_tensor = torch.tensor(tau_aug, dtype=torch.float64)

        # Initialize and optimize β_k
        torch.manual_seed(fold_idx)
        beta_k = torch.nn.Parameter(torch.zeros(11, dtype=torch.float64), requires_grad=True)
        optimizer = Adam([beta_k], lr=5e-3)

        n_fold_total = len(val_idx) + n_a

        for epoch in range(3000):
            optimizer.zero_grad()

            # MLE loss: -τ*Xβ + log(1 + exp(Xβ))
            utility_p = X_fold_tensor @ beta_k
            loss_p = -tau_p_tensor * utility_p + torch.log(1 + torch.exp(utility_p))

            utility_a = X_aug_tensor @ beta_k
            loss_a = -tau_a_tensor * utility_a + torch.log(1 + torch.exp(utility_a))

            loss = (torch.sum(loss_p) + torch.sum(loss_a)) / n_fold_total

            loss.backward()
            optimizer.step()

        beta_folds.append(beta_k.detach().numpy())

    # === STEP 3: Average β across folds ===
    beta_avg = np.mean(beta_folds, axis=0)

    gc.collect()
    return beta_avg


# =============================================================================
# DML (Original): Single β with Cross-fitted Nuisance
# =============================================================================

def run_dml_original(X_all, y_real, y_aug, real_rows, aug_rows, n_folds=5):
    """
    Original DML: Single β optimization with cross-fitted g predictions.
    Uses constant e = n_primary / n_total.
    """
    X_p = [X_all[row] for row in real_rows]
    y_p = np.array([y_real[row] for row in real_rows])
    z_p = np.array([y_aug[row] for row in real_rows])
    X_a = [X_all[row] for row in aug_rows]
    z_a = np.array([y_aug[row] for row in aug_rows])

    n_p = len(y_p)
    n_a = len(z_a)
    n_total = n_p + n_a

    # Constant e (same as DML-2)
    e_constant = n_p / n_total

    class_prior = np.mean(y_p)

    X_p_flat = np.array(flatten_full(X_p))
    X_a_flat = np.array(flatten_full(X_a))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    # Cross-fitted predictions
    g_proba_primary_oof = np.zeros(n_p)
    g_proba_aug_cv = np.zeros(n_a)
    aug_fold_counts = np.zeros(n_a)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_p_flat)):
        # Train g on train_idx
        X_train_g_flat = X_p_flat[train_idx]
        z_train_g = z_p[train_idx]
        y_train_g = y_p[train_idx]

        g_models = {}
        for z_val in [-1, 0, 1]:
            mask = z_train_g == z_val
            if np.sum(mask) >= 2 and len(np.unique(y_train_g[mask])) == 2:
                clf = LogisticRegression(C=0.05, max_iter=2000, random_state=1)
                clf.fit(X_train_g_flat[mask], y_train_g[mask])
                g_models[z_val] = clf

        # Predict g on val_idx (out-of-fold primary)
        for i in val_idx:
            z_i = z_p[i]
            X_i = X_p_flat[i:i+1]
            if z_i in g_models and g_models[z_i] is not None:
                g_proba_primary_oof[i] = g_models[z_i].predict_proba(X_i)[0, 1]
            else:
                g_proba_primary_oof[i] = class_prior

        # Predict g on ALL augmented (accumulate for averaging)
        for j in range(n_a):
            z_j = z_a[j]
            X_j = X_a_flat[j:j+1]
            if z_j in g_models and g_models[z_j] is not None:
                g_proba_aug_cv[j] += g_models[z_j].predict_proba(X_j)[0, 1]
            else:
                g_proba_aug_cv[j] += class_prior
        aug_fold_counts += 1

    # Average augmented predictions
    g_proba_aug_cv = g_proba_aug_cv / aug_fold_counts

    # Compute DML targets
    tau_primary = g_proba_primary_oof * (1 - 1/e_constant) + y_p / e_constant
    tau_primary = np.clip(tau_primary, 0.0, 1.0)
    tau_aug = g_proba_aug_cv

    # Single β optimization on ALL data
    X_p_tensor = torch.tensor(X_p_flat, dtype=torch.float64)
    X_a_tensor = torch.tensor(X_a_flat, dtype=torch.float64)
    tau_p_tensor = torch.tensor(tau_primary, dtype=torch.float64)
    tau_a_tensor = torch.tensor(tau_aug, dtype=torch.float64)

    torch.manual_seed(0)
    beta = torch.nn.Parameter(torch.zeros(11, dtype=torch.float64), requires_grad=True)
    optimizer = Adam([beta], lr=5e-3)

    for epoch in range(5000):
        optimizer.zero_grad()

        utility_p = X_p_tensor @ beta
        loss_p = -tau_p_tensor * utility_p + torch.log(1 + torch.exp(utility_p))

        utility_a = X_a_tensor @ beta
        loss_a = -tau_a_tensor * utility_a + torch.log(1 + torch.exp(utility_a))

        loss = (torch.sum(loss_p) + torch.sum(loss_a)) / n_total

        loss.backward()
        optimizer.step()

    gc.collect()
    return beta.detach().numpy()


# =============================================================================
# Baseline Methods (from compare_correct.py)
# =============================================================================

def fit_mnl(X, y_weighted, n_epochs=5000, lr=5e-4):
    """Fit MNL model with weighted targets."""
    X_flat = np.array(flatten_full(X))
    x_tensor = torch.tensor(X_flat, dtype=torch.float64)
    y_tensor = torch.tensor(y_weighted, dtype=torch.float64)

    params = torch.nn.Parameter(torch.zeros(11, dtype=torch.float64), requires_grad=True)
    torch.manual_seed(0)
    optimizer = Adam([params], lr=lr)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        utility = x_tensor @ params
        utility_full = torch.stack([torch.zeros_like(utility), utility], dim=1)
        log_prob = utility_full - torch.logsumexp(utility_full, dim=1, keepdim=True)
        loss = -torch.sum(y_tensor * log_prob) / len(X)
        loss.backward()
        optimizer.step()

    return params.detach().numpy()


def run_primary_only(X_all, y_real, real_rows):
    """Primary Only baseline."""
    X_p = [X_all[row] for row in real_rows]
    y_p = np.array([y_real[row] for row in real_rows])
    w_p = np.array([[1-y, y] for y in y_p])
    return fit_mnl(X_p, w_p)


def run_naive(X_all, y_real, y_aug, real_rows, aug_rows):
    """Naive: z as hard labels."""
    X_p = [X_all[row] for row in real_rows]
    y_p = np.array([y_real[row] for row in real_rows])
    w_p = np.array([[1-y, y] for y in y_p])

    X_a = [X_all[row] for row in aug_rows]
    z_a = np.array([y_aug[row] for row in aug_rows])

    # Filter valid z (0 or 1, not -1)
    X_c, w_c = [], []
    for i, z in enumerate(z_a):
        if z in [0, 1]:
            X_c.append(X_a[i])
            w_c.append([1-z, z])
    for i in range(len(X_p)):
        X_c.append(X_p[i])
        w_c.append(w_p[i])

    return fit_mnl(X_c, np.array(w_c))


def run_aae(X_all, y_real, y_aug, real_rows, aug_rows):
    """AAE: g(X) stratified by z, soft labels."""
    X_p = [X_all[row] for row in real_rows]
    y_p = np.array([y_real[row] for row in real_rows])
    z_p = np.array([y_aug[row] for row in real_rows])
    X_a = [X_all[row] for row in aug_rows]
    z_a = np.array([y_aug[row] for row in aug_rows])

    X_p_flat = np.array(flatten_full(X_p))
    X_a_flat = np.array(flatten_full(X_a))

    # Train g stratified by z
    g_models = {}
    for z_val in [-1, 0, 1]:
        mask = z_p == z_val
        if np.sum(mask) >= 2 and len(np.unique(y_p[mask])) == 2:
            clf = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic',
                               solver='adam', alpha=1e-4, max_iter=500, random_state=1)
            clf.fit(X_p_flat[mask], y_p[mask])
            g_models[z_val] = clf

    # Primary: hard labels
    w_p = np.array([[1-y, y] for y in y_p])

    # Augmented: soft labels from g
    w_a = []
    for i in range(len(z_a)):
        z_i = z_a[i]
        if z_i in g_models:
            prob = g_models[z_i].predict_proba([X_a_flat[i]])[0]
            # Ensure [P(y=0), P(y=1)] format
            if len(g_models[z_i].classes_) == 2:
                w_a.append([prob[0], prob[1]])
            else:
                w_a.append([1-prob[0], prob[0]])
        else:
            # Fallback to z as hard label
            if z_i in [0, 1]:
                w_a.append([1-z_i, z_i])
            else:
                w_a.append([0.5, 0.5])
    w_a = np.array(w_a)

    # Combine: augmented + primary
    X_c = X_a + X_p
    w_c = np.vstack([w_a, w_p])

    return fit_mnl(X_c, w_c)


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def main():
    print("=" * 80)
    print("DML VARIANT COMPARISON")
    print("=" * 80)
    print("""
    Comparing:
    - Primary Only: Uses only n_real labeled samples
    - Naive: z as hard labels for augmented data
    - AAE: g(X) stratified by z, soft labels (original benchmark)
    - DML: Single β, cross-fitted g, constant e
    - DML-2: Fold-wise β then average (EXACT PDF algorithm)

    Both DML variants use:
    - Constant e = n_primary / n_total (Corollary 2.1)
    - Stratified Logistic Regression for g(X,z)
    - Same cross-fitting for g

    Key difference:
    - DML: Optimize single β on all data with cross-fitted g predictions
    - DML-2: Optimize β_k on each fold, then average β̂ = (1/K) Σ β̂_k
    """)

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

    print(f"Total samples: {n_total}")

    n_real_values = [50, 100, 150, 200]
    n_aug = 1000
    n_trials = 20

    methods = ['Primary', 'Naive', 'AAE', 'DML', 'DML-2']
    results = {method: {n: [] for n in n_real_values} for method in methods}

    for n_real in n_real_values:
        print(f"\n--- n_real = {n_real}, n_aug = {n_aug} ---")

        for trial in range(n_trials):
            rng = np.random.RandomState(trial)
            n_max = min(n_real + n_aug, n_total)

            # Sample by participants (5 obs each)
            participants = rng.choice(int(n_total/5), size=int(n_max/5), replace=False)
            rows = []
            for j in participants:
                rows += list(range(j*5, min((j+1)*5, n_total)))
            rows = rows[:n_max]

            real_rows = rows[:n_real]
            aug_rows = rows[n_real:n_max]

            # Primary Only
            try:
                beta = run_primary_only(X_all, y_real, real_rows)
                results['Primary'][n_real].append(calculate_mape(beta, GROUND_TRUTH_PARAMS))
            except:
                pass

            # Naive
            try:
                beta = run_naive(X_all, y_real, y_aug, real_rows, aug_rows)
                results['Naive'][n_real].append(calculate_mape(beta, GROUND_TRUTH_PARAMS))
            except:
                pass

            # AAE
            try:
                beta = run_aae(X_all, y_real, y_aug, real_rows, aug_rows)
                results['AAE'][n_real].append(calculate_mape(beta, GROUND_TRUTH_PARAMS))
            except:
                pass

            # DML (original)
            try:
                beta = run_dml_original(X_all, y_real, y_aug, real_rows, aug_rows)
                results['DML'][n_real].append(calculate_mape(beta, GROUND_TRUTH_PARAMS))
            except:
                pass

            # DML-2 (exact PDF)
            try:
                beta = run_dml2(X_all, y_real, y_aug, real_rows, aug_rows)
                results['DML-2'][n_real].append(calculate_mape(beta, GROUND_TRUTH_PARAMS))
            except:
                pass

            gc.collect()

        # Print progress
        print(f"  Trial results (MAPE %):")
        for method in methods:
            if results[method][n_real]:
                m = np.mean(results[method][n_real])
                s = np.std(results[method][n_real])
                print(f"    {method:<10}: {m:5.1f} ± {s:4.1f}")

    # Final summary table
    print("\n" + "=" * 80)
    print("FINAL RESULTS: MAPE (%) - Lower is Better")
    print("=" * 80)

    print(f"\n{'Method':<12}", end="")
    for n in n_real_values:
        print(f" | n={n:<4}", end="")
    print(" |   Avg")
    print("-" * 70)

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
        if mapes:
            print(f" | {np.mean(mapes):5.1f}")
        else:
            print(" |   N/A")

    # DML vs DML-2 comparison
    print("\n" + "=" * 80)
    print("DML vs DML-2 COMPARISON")
    print("=" * 80)

    print(f"\n{'n_real':<8} | {'DML':<12} | {'DML-2':<12} | {'Diff':<10} | {'Winner'}")
    print("-" * 60)

    dml_better = 0
    dml2_better = 0

    for n in n_real_values:
        dml_mape = np.mean(results['DML'][n]) if results['DML'][n] else float('nan')
        dml2_mape = np.mean(results['DML-2'][n]) if results['DML-2'][n] else float('nan')
        diff = dml_mape - dml2_mape

        if diff < -0.5:
            winner = "DML"
            dml_better += 1
        elif diff > 0.5:
            winner = "DML-2"
            dml2_better += 1
        else:
            winner = "~Same"

        print(f"{n:<8} | {dml_mape:10.2f}% | {dml2_mape:10.2f}% | {diff:+8.2f}% | {winner}")

    # Overall
    dml_avg = np.mean([np.mean(results['DML'][n]) for n in n_real_values if results['DML'][n]])
    dml2_avg = np.mean([np.mean(results['DML-2'][n]) for n in n_real_values if results['DML-2'][n]])

    print("-" * 60)
    print(f"{'Average':<8} | {dml_avg:10.2f}% | {dml2_avg:10.2f}% | {dml_avg-dml2_avg:+8.2f}%")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if abs(dml_avg - dml2_avg) < 1.0:
        print("""
    DML and DML-2 produce SIMILAR results (difference < 1% MAPE).

    This confirms that:
    1. Single β optimization ≈ Fold-wise β averaging (asymptotically equivalent)
    2. Using constant e = n_p/n_total is valid (Corollary 2.1)
    3. Cross-fitting for g is the key requirement, not β averaging

    The original implementation is theoretically sound.
        """)
    elif dml_avg < dml2_avg:
        print(f"""
    DML (single β) outperforms DML-2 (averaged β) by {dml2_avg - dml_avg:.2f}% MAPE.
    This is likely due to:
    - Single β uses more data for final optimization
    - Averaging fold-wise β may have higher variance with small folds
        """)
    else:
        print(f"""
    DML-2 (averaged β) outperforms DML (single β) by {dml_avg - dml2_avg:.2f}% MAPE.
    The exact PDF algorithm performs better in this setting.
        """)

    # Improvement over baselines
    print("\n" + "=" * 80)
    print("IMPROVEMENT OVER BASELINES")
    print("=" * 80)

    aae_avg = np.mean([np.mean(results['AAE'][n]) for n in n_real_values if results['AAE'][n]])
    primary_avg = np.mean([np.mean(results['Primary'][n]) for n in n_real_values if results['Primary'][n]])

    print(f"\nDML improvement over AAE:     {aae_avg - dml_avg:+.2f}% MAPE")
    print(f"DML-2 improvement over AAE:   {aae_avg - dml2_avg:+.2f}% MAPE")
    print(f"DML improvement over Primary: {primary_avg - dml_avg:+.2f}% MAPE")
    print(f"DML-2 improvement over Primary: {primary_avg - dml2_avg:+.2f}% MAPE")

    return results


if __name__ == "__main__":
    results = main()
