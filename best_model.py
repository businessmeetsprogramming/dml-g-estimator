"""
Best G-Function Estimator for DML Framework
============================================
Achieves ~61% accuracy (best found after extensive optimization)

Key components:
1. Difference features (alt0 - alt1)
2. Z one-hot encoding
3. Z × top feature interactions
4. Cleanlab noise filtering
5. Logistic Regression (linear models work best)

Usage:
    from best_model import BestGEstimator, prepare_features

    model = BestGEstimator(cleanlab_pct=30)
    model.fit(X_list_train, z_train, y_train)
    predictions = model.predict(X_list_test, z_test)
"""
import pickle as pkl
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Try to import cleanlab (optional but recommended)
try:
    from cleanlab.rank import get_label_quality_scores
    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False
    print("Warning: cleanlab not available. Install with: pip install cleanlab")


# =============================================================================
# FEATURE PREPARATION
# =============================================================================

def get_diff_features(X_list):
    """
    Extract difference features from X_list.

    Args:
        X_list: List of arrays, each with shape (2, 12) for 2 alternatives

    Returns:
        X_diff: Array of shape (n, 11) - difference features (dropping constant feature 0)
    """
    X_arr = np.array(X_list)
    alt0 = X_arr[:, 0, 1:]  # Drop constant feature 0
    alt1 = X_arr[:, 1, 1:]
    return alt0 - alt1


def one_hot_z(z):
    """One-hot encode z values (-1, 0, 1) to 3 columns."""
    z = np.asarray(z)
    z_oh = np.zeros((len(z), 3))
    for i, v in enumerate(z):
        z_oh[i, int(v) + 1] = 1
    return z_oh


def create_z_interactions(X_diff, z, top_feature_indices=None):
    """
    Create interaction features between z and top diff features.

    Args:
        X_diff: Difference features (n, 11)
        z: Z values (n,)
        top_feature_indices: List of feature indices to interact with z
                            Default: [1, 6, 0, 8, 9] (top 5 by correlation)

    Returns:
        Interaction features (n, len(top_feature_indices))
    """
    if top_feature_indices is None:
        top_feature_indices = [1, 6, 0, 8, 9]  # Top 5 features by |correlation| with y

    z_numeric = np.asarray(z).reshape(-1, 1)
    interactions = []
    for feat_idx in top_feature_indices:
        interaction = z_numeric * X_diff[:, feat_idx:feat_idx+1]
        interactions.append(interaction)

    return np.hstack(interactions) if interactions else np.array([]).reshape(len(X_diff), 0)


def prepare_features(X_list, z, include_z_onehot=True, include_interactions=True,
                    top_k_interactions=5):
    """
    Prepare full feature set for the best model.

    Args:
        X_list: List of X arrays
        z: Z values
        include_z_onehot: Include Z one-hot encoding
        include_interactions: Include Z × top feature interactions
        top_k_interactions: Number of top features to interact with Z

    Returns:
        X_full: Full feature array
    """
    X_diff = get_diff_features(X_list)

    features = [X_diff]

    if include_z_onehot:
        features.append(one_hot_z(z))

    if include_interactions:
        top_indices = [1, 6, 0, 8, 9][:top_k_interactions]
        interactions = create_z_interactions(X_diff, z, top_indices)
        if interactions.shape[1] > 0:
            features.append(interactions)

    return np.hstack(features)


# =============================================================================
# BEST MODEL CLASS
# =============================================================================

class BestGEstimator:
    """
    Best G-function estimator combining all optimization findings.

    Architecture:
    - Features: diff + z_onehot + z×top5_interactions = 19 features
    - Model: Logistic Regression (C=1.0)
    - Preprocessing: StandardScaler
    - Optional: Cleanlab noise filtering

    Achieves ~61% accuracy on held-out test data.
    """

    def __init__(self, cleanlab_pct=30, C=1.0, seed=42):
        """
        Args:
            cleanlab_pct: Percentage of lowest quality samples to remove (0-100)
                         Set to 0 to disable Cleanlab filtering
            C: Regularization parameter for LogisticRegression
            seed: Random seed
        """
        self.cleanlab_pct = cleanlab_pct
        self.C = C
        self.seed = seed
        self.model = None
        self.scaler = StandardScaler()
        self.label_quality = None
        self._is_fitted = False

    def _compute_label_quality(self, X_full, y):
        """Compute label quality scores using Cleanlab."""
        if not CLEANLAB_AVAILABLE or self.cleanlab_pct <= 0:
            return None

        model = Pipeline([
            ("s", StandardScaler()),
            ("c", LogisticRegression(C=1.0, max_iter=2000, random_state=self.seed))
        ])
        pred_probs = cross_val_predict(model, X_full, y, cv=5, method='predict_proba')
        return get_label_quality_scores(y, pred_probs)

    def fit(self, X_list, z, y):
        """
        Fit the model.

        Args:
            X_list: List of X arrays (each shape (2, 12))
            z: Z values array
            y: Target labels

        Returns:
            self
        """
        y = np.asarray(y)
        z = np.asarray(z)

        # Prepare features
        X_full = prepare_features(X_list, z)

        # Compute label quality for Cleanlab filtering
        if CLEANLAB_AVAILABLE and self.cleanlab_pct > 0:
            self.label_quality = self._compute_label_quality(X_full, y)

            # Filter low quality samples
            threshold = np.percentile(self.label_quality, self.cleanlab_pct)
            keep_mask = self.label_quality >= threshold

            # Ensure both classes remain
            if len(np.unique(y[keep_mask])) == 2:
                X_full = X_full[keep_mask]
                y = y[keep_mask]

        # Scale and fit
        X_scaled = self.scaler.fit_transform(X_full)
        self.model = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed)
        self.model.fit(X_scaled, y)

        self._is_fitted = True
        return self

    def predict(self, X_list, z):
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_full = prepare_features(X_list, z)
        X_scaled = self.scaler.transform(X_full)
        return self.model.predict(X_scaled)

    def predict_proba(self, X_list, z):
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_full = prepare_features(X_list, z)
        X_scaled = self.scaler.transform(X_full)
        return self.model.predict_proba(X_scaled)


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_model(model_class, X_list, z_all, y_all,
                   sample_sizes=[50, 100, 200, 400, 800],
                   n_splits=30, test_size=200, seed=42, **model_kwargs):
    """
    Evaluate model across different sample sizes with multiple random splits.

    Returns:
        Dictionary mapping sample size to list of accuracies
    """
    n_total = len(y_all)
    results = {n: [] for n in sample_sizes}

    for i in range(n_splits):
        split_seed = seed + i * 17
        rng = np.random.RandomState(split_seed)

        indices = rng.permutation(n_total)
        test_idx = indices[-test_size:]
        train_pool = indices[:-test_size]

        X_test = [X_list[j] for j in test_idx]
        z_test = z_all[test_idx]
        y_test = y_all[test_idx]

        for n_train in sample_sizes:
            train_idx = train_pool[:n_train]

            X_train = [X_list[j] for j in train_idx]
            z_train = z_all[train_idx]
            y_train = y_all[train_idx]

            model = model_class(seed=split_seed, **model_kwargs)
            model.fit(X_train, z_train, y_train)
            pred = model.predict(X_test, z_test)

            results[n_train].append(accuracy_score(y_test, pred))

    return results


# =============================================================================
# MAIN - Run if executed directly
# =============================================================================

if __name__ == "__main__":
    print("Loading data...")
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

    print("\nEvaluating Best Model...")
    results = evaluate_model(BestGEstimator, X_list, z_all, y_all, cleanlab_pct=30)

    print("\n" + "="*60)
    print("BEST MODEL RESULTS")
    print("="*60)
    print("\n| Sample Size | Mean Accuracy | Std |")
    print("|-------------|---------------|-----|")

    for n in [50, 100, 200, 400, 800]:
        mean = np.mean(results[n])
        std = np.std(results[n])
        print(f"| n={n:<9} | {mean*100:.2f}%         | {std*100:.2f}% |")

    avg = np.mean([np.mean(results[n]) for n in results])
    print(f"\nOverall Average: {avg*100:.2f}%")
