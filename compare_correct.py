"""
CORRECT Comparison: AAE vs Naive vs Primary Only vs PPI
Based on exact GitHub implementations
"""
import pickle as pkl
import numpy as np
import torch
from torch.optim import Adam
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from best_model import BestGEstimator, prepare_features
import gc
import warnings
warnings.filterwarnings("ignore")

try:
    from ppi_py import ppi_logistic_pointestimate
    PPI_AVAILABLE = True
except ImportError:
    PPI_AVAILABLE = False

GROUND_TRUTH_PARAMS = np.array([
    0.36310104, 0.7465673, 0.32377172, -0.21252407, 0.08090729,
    -0.09540857, -0.40639496, -0.15332593, -0.24158926, 0.17760716, -0.04599298
])

# =============================================================================
# UTILITY FUNCTIONS (from GitHub)
# =============================================================================

def flatten(X):
    return X[1][1:] - X[0][1:]

def flatten_full(X_list):
    return [flatten(X_list[i]) for i in range(len(X_list))]

def prepare_features_with_z(X_list, z):
    """
    Prepare features for g(X, z) model:
    - Difference features (alt0 - alt1)
    - Z one-hot encoding
    - Z × top feature interactions
    """
    X_arr = np.array(X_list)
    alt0 = X_arr[:, 0, 1:]  # Drop constant feature 0
    alt1 = X_arr[:, 1, 1:]
    X_diff = alt0 - alt1

    # One-hot encode z
    z = np.asarray(z)
    z_oh = np.zeros((len(z), 3))
    for i, v in enumerate(z):
        z_oh[i, int(v) + 1] = 1

    # Z × top feature interactions (top 5 features by correlation)
    top_indices = [1, 6, 0, 8, 9]
    z_numeric = z.reshape(-1, 1)
    interactions = np.hstack([z_numeric * X_diff[:, idx:idx+1] for idx in top_indices])

    return np.hstack([X_diff, z_oh, interactions])

def log_sum_exp(inputs, keepdim=False, mask=None):
    if mask is not None:
        max_offset = -1e7 * mask
    else:
        max_offset = 0.
    s, _ = torch.max(inputs + max_offset, dim=-1, keepdim=True)
    inputs_offset = inputs - s
    if mask is not None:
        inputs_offset.masked_fill_(mask.bool(), -float('inf'))
    outputs = s + inputs_offset.exp().sum(dim=-1, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(-1)
    return outputs

def convert_input(X, y_weighted):
    X_diff = [X[i][1][1:] - X[i][0][1:] for i in range(len(X))]
    x_all = torch.tensor(np.array(X_diff), dtype=torch.float64)
    y_all_weighted = torch.tensor(y_weighted, dtype=torch.float64)
    return x_all, y_all_weighted

def neg_ll(params, x_all, y_all_weighted):
    utility_all = torch.sum(params * x_all, 1)
    utility_all = utility_all.unsqueeze(1)
    utility_all = torch.cat((torch.zeros(utility_all.shape[0], 1, dtype=torch.float64), utility_all), 1)
    LL = torch.sum(utility_all * y_all_weighted, 1) - log_sum_exp(utility_all)
    return -torch.sum(LL) / x_all.shape[0]

def fit(X, y_weighted, seed=0, n_epochs=5000, lr=5e-4):
    params = torch.nn.Parameter(torch.ones(X[0].shape[1] - 1, dtype=torch.float64), requires_grad=True)
    torch.manual_seed(seed)
    x_all, y_all_weighted = convert_input(X, y_weighted)
    optimizer = Adam([params], lr=lr)
    for i in range(n_epochs):
        optimizer.zero_grad()
        nll = neg_ll(params, x_all, y_all_weighted)
        nll.backward()
        optimizer.step()
    return params.detach().numpy()

def fit_g(X, y, seed=0):
    if len(np.unique(y)) == 1:
        return 0, y[0]
    X_flat = flatten_full(X)
    clf = MLPClassifier(solver='adam', alpha=1e-4, activation='logistic',
                       hidden_layer_sizes=(10, 5), random_state=1, max_iter=500).fit(X_flat, y)
    return 1, clf

def calculate_mape(estimated, true):
    return np.mean(np.abs((estimated - true) / (true + 1))) * 100

# =============================================================================
# PRIMARY ONLY (n_aug = 0)
# =============================================================================

def run_primary_only(X_all, y_real, real_rows):
    """Primary Only - uses ONLY real labeled data, NO augmented data"""
    X_p = [X_all[row] for row in real_rows]
    y_p = np.array([y_real[row] for row in real_rows])
    w_p = np.array([[int(y == 0), int(y == 1)] for y in y_p])
    return fit(X_p, w_p, seed=0)

# =============================================================================
# NAIVE (from naive.py) - uses z as HARD labels
# =============================================================================

def run_naive(X_all, y_real, y_aug, real_rows, aug_rows):
    """
    Naive method from GitHub naive.py
    Uses z (LLM prediction) as HARD labels for augmented data
    """
    # Primary: hard labels from y_real
    w_p = np.array([[int(y_real[row] == 0), int(y_real[row] == 1)] for row in real_rows])
    X_p = [X_all[row] for row in real_rows]

    # Augmented: HARD labels from z (y_aug)
    # z=0 -> [1,0], z=1 -> [0,1], z=-1 -> [1,0] (since int(-1==0)=0, int(-1==1)=0... wait)
    # Actually: z=-1 -> [0,0] which gets filtered out
    w_a = np.array([[int(y_aug[row] == 0), int(y_aug[row] == 1)] for row in aug_rows])
    X_a = [X_all[row] for row in aug_rows]

    # Filter out invalid weights (where sum != 1, i.e., z=-1 cases)
    X_c, w_c = [], []
    for j in range(len(w_a)):
        if w_a[j][0] + w_a[j][1] != 0:
            w_c.append(w_a[j])
            X_c.append(X_a[j])
    for j in range(len(w_p)):
        if w_p[j][0] + w_p[j][1] != 0:
            w_c.append(w_p[j])
            X_c.append(X_p[j])
    w_c = np.array(w_c)

    return fit(X_c, w_c, seed=0)

# =============================================================================
# AAE (from debias.py) - uses g(X,z) as SOFT labels
# =============================================================================

def run_aae(X_all, y_real, y_aug, real_rows, aug_rows, return_accuracy=False):
    """
    AAE method from GitHub debias.py
    Uses g(X,z) predictions as SOFT labels for augmented data
    """
    # Primary data
    y_p = np.array([y_real[row] for row in real_rows])
    w_p = np.array([[int(y == 0), int(y == 1)] for y in y_p])
    z_p = np.array([y_aug[row] for row in real_rows])
    X_p = [X_all[row] for row in real_rows]

    # Train g stratified by z
    g_function = {}
    for label in [-1, 0, 1]:
        idx = [i for i in range(len(y_p)) if z_p[i] == label]
        if len(idx) > 0:
            y_sub = y_p[idx]
            X_sub = [X_p[i] for i in idx]
            g_function[label] = fit_g(X_sub, y_sub, seed=0)

    # Compute OUT-OF-SAMPLE accuracy using cross-validation (fair comparison)
    g_preds_oof = np.zeros(len(y_p))
    X_p_flat = np.array(flatten_full(X_p))
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    for train_idx, val_idx in kf.split(X_p_flat):
        # Train stratified g models on this fold
        g_fold = {}
        for z_val in [-1, 0, 1]:
            train_z_mask = z_p[train_idx] == z_val
            if np.sum(train_z_mask) < 2:
                continue
            X_train_z = X_p_flat[train_idx][train_z_mask]
            y_train_z = y_p[train_idx][train_z_mask]
            if len(np.unique(y_train_z)) == 1:
                g_fold[z_val] = ('constant', y_train_z[0])
            else:
                clf = MLPClassifier(solver='adam', alpha=1e-4, activation='logistic',
                                   hidden_layer_sizes=(10, 5), random_state=1, max_iter=500)
                clf.fit(X_train_z, y_train_z)
                g_fold[z_val] = ('model', clf)

        # Predict on validation
        for i in val_idx:
            z_val = z_p[i]
            if z_val in g_fold:
                g_type, g_func = g_fold[z_val]
                if g_type == 'constant':
                    g_preds_oof[i] = g_func
                else:
                    g_preds_oof[i] = g_func.predict([X_p_flat[i]])[0]
            else:
                g_preds_oof[i] = 0  # Default

    g_accuracy_oof = np.mean(g_preds_oof == y_p)

    # Augmented data
    z_a = np.array([y_aug[row] for row in aug_rows])
    X_a = [X_all[row] for row in aug_rows]

    # Compute SOFT labels using g
    w_a = []
    for i in range(len(z_a)):
        if z_a[i] in g_function:
            g_type, g_func = g_function[z_a[i]]
            if g_type == 0:
                weights = np.array([int(g_func == 0), int(g_func == 1)])
            else:
                X_flat = flatten(X_a[i])
                classes = g_func.classes_
                weights = g_func.predict_proba([X_flat])[0]
                for j, cl in enumerate([0, 1]):
                    if cl not in classes:
                        weights = np.insert(weights, j, 0)
            w_a.append(weights)
        else:
            w_a.append(np.array([int(z_a[i] == 0), int(z_a[i] == 1)]))
    w_a = np.array(w_a)

    # Combine: Augmented FIRST, then Primary
    X_c = X_a + X_p
    w_c = np.concatenate([w_a, w_p])

    beta = fit(X_c, w_c, seed=0)

    if return_accuracy:
        return beta, g_accuracy_oof  # Return out-of-sample accuracy (fair comparison)
    return beta

# =============================================================================
# DML - Optimized g & e with MLP, Cross-fitting, Debiasing
# =============================================================================

def get_best_g_model(n_samples, seed=0):
    """Get the best g model - EXACT same as AAE: MLP(10,5)."""
    # Use exact same architecture as AAE for fair comparison
    return MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic',
                        solver='adam', alpha=1e-4, max_iter=500, random_state=1)

def get_best_e_model(n_samples, seed=0):
    """Get the best e model based on sample size."""
    if n_samples < 200:
        return MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic',
                            solver='adam', alpha=1e-4, max_iter=500, random_state=seed)
    else:
        return MLPClassifier(hidden_layer_sizes=(20, 10), activation='logistic',
                            solver='adam', alpha=1e-4, max_iter=500, random_state=seed)

def run_dml(X_all, y_real, y_aug, real_rows, aug_rows, n_folds=5, return_accuracy=False):
    """
    DML method - Optimized with multiple improvements over AAE:

    1. ENSEMBLE G-MODEL: Combines stratified MLP (like AAE) + pooled LR with enhanced features
    2. ENHANCED FEATURES: diff + z_onehot + z×feature_interactions for pooled model
    3. ADAPTIVE TEMPERATURE: Calibrates soft label confidence based on sample size
    4. SAMPLE WEIGHTING: Gives more weight to reliable primary data
    5. CROSS-FITTED AUGMENTED: Uses cross-fitting for more robust soft labels
    """
    X_p = [X_all[row] for row in real_rows]
    y_p = np.array([y_real[row] for row in real_rows])
    z_p = np.array([y_aug[row] for row in real_rows])
    X_a = [X_all[row] for row in aug_rows]
    z_a = np.array([y_aug[row] for row in aug_rows])

    n_p = len(y_p)
    n_a = len(z_a)

    # Prepare features
    X_p_flat = np.array(flatten_full(X_p))  # For stratified MLP
    X_a_flat = np.array(flatten_full(X_a))
    X_p_enhanced = prepare_features(X_p, z_p)  # For pooled LR
    X_a_enhanced = prepare_features(X_a, z_a)

    # Scale enhanced features
    scaler = StandardScaler()
    X_all_enhanced = np.vstack([X_p_enhanced, X_a_enhanced])
    X_all_scaled = scaler.fit_transform(X_all_enhanced)
    X_p_scaled = X_all_scaled[:n_p]
    X_a_scaled = X_all_scaled[n_p:]

    # === ENSEMBLE G-MODEL: Stratified MLP + Pooled LR ===
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    # Store cross-fitted predictions for accuracy
    g_preds_oof = np.zeros(n_p)
    g_proba_oof = np.zeros((n_p, 2))

    # Also store cross-fitted predictions for augmented data
    g_proba_aug_cv = np.zeros((n_a, 2))
    aug_fold_counts = np.zeros(n_a)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_p_flat)):
        # === MODEL 1: Stratified MLP (like AAE) ===
        g_stratified = {}
        for z_val in [-1, 0, 1]:
            train_z_mask = z_p[train_idx] == z_val
            if np.sum(train_z_mask) < 2:
                continue
            X_train_z = X_p_flat[train_idx][train_z_mask]
            y_train_z = y_p[train_idx][train_z_mask]
            if len(np.unique(y_train_z)) == 1:
                g_stratified[z_val] = ('constant', y_train_z[0])
            else:
                model = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic',
                                    solver='adam', alpha=1e-4, max_iter=500, random_state=1)
                model.fit(X_train_z, y_train_z)
                g_stratified[z_val] = ('model', model)

        # === MODEL 2: Pooled LR with enhanced features ===
        g_pooled = None
        if len(np.unique(y_p[train_idx])) > 1:
            g_pooled = LogisticRegression(C=1.0, max_iter=2000, random_state=1)
            g_pooled.fit(X_p_scaled[train_idx], y_p[train_idx])

        # === ENSEMBLE: Combine predictions ===
        # Weight: stratified gets more weight at small n, pooled at large n
        stratified_weight = max(0.3, min(0.7, 1.0 - n_p / 300))
        pooled_weight = 1.0 - stratified_weight

        # Predict on validation (primary)
        for i in val_idx:
            z_val = z_p[i]

            # Stratified prediction
            if z_val in g_stratified:
                g_type, g_func = g_stratified[z_val]
                if g_type == 'constant':
                    strat_proba = np.array([1.0 - g_func, float(g_func)])
                else:
                    strat_proba = g_func.predict_proba([X_p_flat[i]])[0]
            else:
                strat_proba = np.array([0.5, 0.5])

            # Pooled prediction
            if g_pooled is not None:
                pool_proba = g_pooled.predict_proba([X_p_scaled[i]])[0]
            else:
                pool_proba = np.array([0.5, 0.5])

            # Ensemble
            g_proba_oof[i] = stratified_weight * strat_proba + pooled_weight * pool_proba
            g_preds_oof[i] = np.argmax(g_proba_oof[i])

        # Predict on augmented (cross-fitted for robustness)
        for i in range(n_a):
            z_val = z_a[i]

            # Stratified prediction
            if z_val in g_stratified:
                g_type, g_func = g_stratified[z_val]
                if g_type == 'constant':
                    strat_proba = np.array([1.0 - g_func, float(g_func)])
                else:
                    strat_proba = g_func.predict_proba([X_a_flat[i]])[0]
            else:
                strat_proba = np.array([0.5, 0.5])

            # Pooled prediction
            if g_pooled is not None:
                pool_proba = g_pooled.predict_proba([X_a_scaled[i]])[0]
            else:
                pool_proba = np.array([0.5, 0.5])

            # Ensemble
            g_proba_aug_cv[i] += stratified_weight * strat_proba + pooled_weight * pool_proba
            aug_fold_counts[i] += 1

    # Average augmented predictions across folds
    g_proba_aug_cv = g_proba_aug_cv / aug_fold_counts[:, np.newaxis]

    g_accuracy = np.mean(g_preds_oof == y_p)

    # === ADAPTIVE TEMPERATURE SCALING ===
    if n_p <= 60:
        temperature = 2.5  # Very high for tiny samples
    elif n_p <= 100:
        temperature = 1.8
    elif n_p <= 150:
        temperature = 1.3
    else:
        temperature = 1.0

    # Apply temperature to augmented soft labels
    w_a = []
    for i in range(n_a):
        proba = g_proba_aug_cv[i]
        if temperature != 1.0:
            log_odds = np.log(np.clip(proba, 1e-10, 1-1e-10))
            scaled_odds = log_odds / temperature
            weights = np.exp(scaled_odds) / np.sum(np.exp(scaled_odds))
        else:
            weights = proba
        w_a.append(weights)
    w_a = np.array(w_a)

    # === PRIMARY: HARD labels (like AAE) ===
    w_p = np.array([[1 - y, y] for y in y_p], dtype=float)

    # === SAMPLE WEIGHTING: Give more weight to primary ===
    # This increases the effective sample size of reliable primary data
    primary_weight = 1.0 + (100 / max(n_p, 50))  # ~3x for n=50, ~2x for n=100, ~1.5x for n=200
    w_p = w_p * primary_weight

    # === Combine: Augmented FIRST, then Primary ===
    X_c = X_a + X_p
    w_c = np.concatenate([w_a, w_p])

    # Normalize weights to sum to n_a + n_p (to keep loss scale consistent)
    w_c = w_c * (n_a + n_p) / np.sum(w_c)

    gc.collect()

    beta = fit(X_c, w_c, seed=0)

    if return_accuracy:
        return beta, g_accuracy
    return beta

# =============================================================================
# PPI
# =============================================================================

def run_ppi(X_all, y_real, y_aug, real_rows, aug_rows):
    """PPI implementation matching GitHub exactly"""
    if not PPI_AVAILABLE:
        return None

    X_p = np.array(flatten_full([X_all[row] for row in real_rows]))
    y_p = np.array([y_real[row] for row in real_rows]).astype(float)
    z_p = np.array([y_aug[row] for row in real_rows]).astype(float)

    X_a = np.array(flatten_full([X_all[row] for row in aug_rows]))
    z_a = np.array([y_aug[row] for row in aug_rows]).astype(float)

    # Optimizer options from GitHub
    optimizer_options = {"ftol": 1e-5, "gtol": 1e-5, "maxiter": 10000}

    try:
        return ppi_logistic_pointestimate(X_p, y_p, z_p, X_a, z_a, lam=1,
                                          optimizer_options=optimizer_options)
    except:
        return None

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("CORRECT COMPARISON: AAE vs Naive vs Primary Only vs PPI vs DML")
    print("=" * 100)
    print("""
    Methods:
    - Primary Only: Uses ONLY n_real samples (no augmented data)
    - Naive: Uses n_real + n_aug, with z as HARD labels for augmented
    - AAE: Uses n_real + n_aug, MLP(10,5) g(X) stratified by z
    - DML: Uses n_real + n_aug, OPTIMIZED ensemble with multiple improvements
    - PPI: Prediction-Powered Inference

    DML optimizations over AAE:
    1. ENSEMBLE G-MODEL: Stratified MLP + Pooled LR with enhanced features
    2. ENHANCED FEATURES: diff + z_onehot + z×feature_interactions
    3. ADAPTIVE TEMPERATURE: Calibrates soft label confidence (T=2.5 to 1.0)
    4. SAMPLE WEIGHTING: More weight on reliable primary data
    5. CROSS-FITTED AUGMENTED: Uses cross-fitting for robust soft labels

    NOTE: Both g-accuracy measured OUT-OF-SAMPLE via 5-fold cross-validation
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
    n_trials = 10

    results = {method: {n: [] for n in n_real_values}
               for method in ['Primary', 'Naive', 'AAE', 'DML', 'PPI']}

    # Track g-function accuracies
    g_accuracies = {'AAE': {n: [] for n in n_real_values},
                    'DML': {n: [] for n in n_real_values}}

    for n_real in n_real_values:
        print(f"\n--- n_real = {n_real}, n_aug = {n_aug} ---")

        for trial in range(n_trials):
            rng = np.random.RandomState(trial)
            n_max = min(n_real + n_aug, n_total)

            # Sample like GitHub: by participants (5 obs each)
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
            except Exception as e:
                pass

            # Naive
            try:
                beta = run_naive(X_all, y_real, y_aug, real_rows, aug_rows)
                results['Naive'][n_real].append(calculate_mape(beta, GROUND_TRUTH_PARAMS))
            except Exception as e:
                pass

            # AAE (with accuracy tracking)
            try:
                beta, acc = run_aae(X_all, y_real, y_aug, real_rows, aug_rows, return_accuracy=True)
                results['AAE'][n_real].append(calculate_mape(beta, GROUND_TRUTH_PARAMS))
                g_accuracies['AAE'][n_real].append(acc)
            except Exception as e:
                pass

            # DML (with accuracy tracking)
            try:
                beta, acc = run_dml(X_all, y_real, y_aug, real_rows, aug_rows, return_accuracy=True)
                results['DML'][n_real].append(calculate_mape(beta, GROUND_TRUTH_PARAMS))
                g_accuracies['DML'][n_real].append(acc)
            except Exception as e:
                pass

            # PPI
            if PPI_AVAILABLE:
                try:
                    beta = run_ppi(X_all, y_real, y_aug, real_rows, aug_rows)
                    if beta is not None:
                        mape = calculate_mape(beta, GROUND_TRUTH_PARAMS)
                        # Filter extreme values (numerical instability)
                        if mape < 1000:
                            results['PPI'][n_real].append(mape)
                except:
                    pass

            # Memory cleanup after each trial
            gc.collect()

        # Print progress
        p = np.mean(results['Primary'][n_real]) if results['Primary'][n_real] else float('nan')
        n = np.mean(results['Naive'][n_real]) if results['Naive'][n_real] else float('nan')
        a = np.mean(results['AAE'][n_real]) if results['AAE'][n_real] else float('nan')
        d = np.mean(results['DML'][n_real]) if results['DML'][n_real] else float('nan')
        pp = np.mean(results['PPI'][n_real]) if results['PPI'][n_real] else float('nan')
        aae_acc = np.mean(g_accuracies['AAE'][n_real]) if g_accuracies['AAE'][n_real] else float('nan')
        dml_acc = np.mean(g_accuracies['DML'][n_real]) if g_accuracies['DML'][n_real] else float('nan')
        print(f"  MAPE - Primary: {p:.1f}%, Naive: {n:.1f}%, AAE: {a:.1f}%, DML: {d:.1f}%, PPI: {pp:.1f}%")
        print(f"  G-Acc - AAE: {aae_acc*100:.1f}%, DML: {dml_acc*100:.1f}%")

    # Final table
    print("\n" + "=" * 100)
    print("FINAL RESULTS: MAPE (%) - Lower is Better")
    print("=" * 100)

    print(f"\n{'Method':<12}", end="")
    for n in n_real_values:
        print(f" | n={n:<4}", end="")
    print(" | Avg")
    print("-" * 85)

    for method in ['Primary', 'Naive', 'AAE', 'DML', 'PPI']:
        print(f"{method:<12}", end="")
        mapes = []
        for n in n_real_values:
            if results[method][n]:
                m = np.mean(results[method][n])
                mapes.append(m)
                print(f" | {m:5.1f}", end="")
            else:
                print(f" |   N/A", end="")
        print(f" | {np.mean(mapes):5.1f}" if mapes else " |   N/A")

    # G-function accuracy table
    print("\n" + "=" * 100)
    print("G-FUNCTION ACCURACY (%) - Higher is Better")
    print("=" * 100)

    print(f"\n{'Method':<12}", end="")
    for n in n_real_values:
        print(f" | n={n:<4}", end="")
    print(" | Avg")
    print("-" * 85)

    for method in ['AAE', 'DML']:
        print(f"{method:<12}", end="")
        accs = []
        for n in n_real_values:
            if g_accuracies[method][n]:
                acc = np.mean(g_accuracies[method][n]) * 100
                accs.append(acc)
                print(f" | {acc:5.1f}", end="")
            else:
                print(f" |   N/A", end="")
        print(f" | {np.mean(accs):5.1f}" if accs else " |   N/A")

    # Improvement table
    print("\n" + "-" * 85)
    print("MAPE Improvement over Primary Only:")
    for method in ['Naive', 'AAE', 'DML', 'PPI']:
        print(f"{method:<12}", end="")
        for n in n_real_values:
            if results[method][n] and results['Primary'][n]:
                imp = np.mean(results['Primary'][n]) - np.mean(results[method][n])
                print(f" | {imp:+5.1f}", end="")
            else:
                print(f" |   N/A", end="")
        print()

    return results, g_accuracies

if __name__ == "__main__":
    results = main()
