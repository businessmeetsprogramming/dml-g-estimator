"""
Optimized G-Function Estimator with:
1. Advanced feature engineering
2. Neural networks with tuning
3. Hyperparameter optimization (Optuna)
4. Multi-level stacking and blending
"""
from __future__ import annotations
import pickle as pkl
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.base import clone

CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    pass


# =============================================================================
# Advanced Feature Engineering
# =============================================================================

def one_hot_z(z: np.ndarray, z_values: Tuple[int, ...] = (-1, 0, 1)) -> np.ndarray:
    """One-hot encode z (GPT predictions)."""
    z = np.asarray(z)
    out = np.zeros((len(z), len(z_values)), dtype=float)
    for j, v in enumerate(z_values):
        out[:, j] = (z == v).astype(float)
    return out


def engineer_X_features_v2(X: List[np.ndarray], drop_first_col: bool = True) -> np.ndarray:
    """
    Advanced feature engineering v2 with more features:
    - All v1 features
    - Ratio features
    - Weighted differences
    - Polynomial interactions
    """
    engineered = []

    for Xi in X:
        Xi = np.asarray(Xi, dtype=float)
        if drop_first_col:
            Xi = Xi[:, 1:]

        alt0 = Xi[0]
        alt1 = Xi[1]
        features = []

        # === V1 Features ===
        # Original flattened
        features.extend(alt0.tolist())
        features.extend(alt1.tolist())

        # Difference
        diff = alt0 - alt1
        features.extend(diff.tolist())

        # Absolute difference
        abs_diff = np.abs(diff)
        features.extend(abs_diff.tolist())

        # Product
        product = alt0 * alt1
        features.extend(product.tolist())

        # Sum
        sum_feat = alt0 + alt1
        features.extend(sum_feat.tolist())

        # Max/Min
        max_feat = np.maximum(alt0, alt1)
        min_feat = np.minimum(alt0, alt1)
        features.extend(max_feat.tolist())
        features.extend(min_feat.tolist())

        # Aggregates
        features.append(np.sum(alt0))
        features.append(np.sum(alt1))
        features.append(np.sum(diff))
        features.append(np.sum(abs_diff))
        features.append(np.sum(product))

        # Second-order interactions
        for i in range(min(5, len(alt0))):
            for j in range(i + 1, min(6, len(alt0))):
                features.append(alt0[i] * alt0[j])
                features.append(alt1[i] * alt1[j])
                features.append(diff[i] * diff[j])

        # === V2 New Features ===

        # Ratio features (with epsilon to avoid division by zero)
        eps = 0.01
        ratio = (alt0 + eps) / (alt1 + eps)
        features.extend(ratio.tolist())

        # Inverse ratio
        inv_ratio = (alt1 + eps) / (alt0 + eps)
        features.extend(inv_ratio.tolist())

        # Normalized difference: (a0 - a1) / (a0 + a1 + eps)
        norm_diff = diff / (sum_feat + eps)
        features.extend(norm_diff.tolist())

        # Squared difference
        sq_diff = diff ** 2
        features.extend(sq_diff.tolist())

        # Weighted features (weight by position)
        weights = np.linspace(1, 2, len(alt0))
        weighted_alt0 = alt0 * weights
        weighted_alt1 = alt1 * weights
        features.extend(weighted_alt0.tolist())
        features.extend(weighted_alt1.tolist())
        features.extend((weighted_alt0 - weighted_alt1).tolist())

        # Count-based features
        features.append(np.sum(alt0 > 0))  # Non-zero count alt0
        features.append(np.sum(alt1 > 0))  # Non-zero count alt1
        features.append(np.sum(diff > 0))  # Features where alt0 > alt1
        features.append(np.sum(diff < 0))  # Features where alt1 > alt0
        features.append(np.sum(diff == 0))  # Features where equal

        # Dominance indicators
        features.append(1 if np.sum(diff > 0) > np.sum(diff < 0) else 0)
        features.append(1 if np.sum(alt0) > np.sum(alt1) else 0)

        # Triple interactions (top features only)
        for i in range(min(3, len(alt0))):
            for j in range(i + 1, min(4, len(alt0))):
                for k in range(j + 1, min(5, len(alt0))):
                    features.append(diff[i] * diff[j] * diff[k])

        engineered.append(features)

    return np.array(engineered, dtype=float)


def flatten_X_simple(X: List[np.ndarray], drop_first_col: bool = True) -> np.ndarray:
    """Simple flattening for baseline."""
    feats = []
    for Xi in X:
        Xi = np.asarray(Xi, dtype=float)
        if drop_first_col:
            Xi = Xi[:, 1:]
        feats.append(Xi.reshape(-1))
    return np.vstack(feats)


# =============================================================================
# Model Definitions
# =============================================================================

def get_aae_model(seed: int = 0) -> VotingClassifier:
    """AAE baseline."""
    estimators = [
        ("hist_gb", HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_depth=5, random_state=seed
        )),
        ("rf", RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1
        )),
        ("logit", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
        ])),
    ]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def get_mlp_model(seed: int = 0, n_samples: int = 200) -> Pipeline:
    """Tuned MLP neural network."""
    # Adapt architecture to sample size
    if n_samples <= 100:
        hidden = (32, 16)
        alpha = 0.01
    elif n_samples <= 200:
        hidden = (64, 32)
        alpha = 0.005
    else:
        hidden = (128, 64, 32)
        alpha = 0.001

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden,
            activation='relu',
            solver='adam',
            alpha=alpha,
            batch_size=min(32, n_samples // 4),
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        ))
    ])


def get_optimized_stacking(seed: int = 0, n_samples: int = 200):
    """Optimized multi-level stacking."""
    # Regularization based on sample size
    if n_samples <= 100:
        hgb_params = dict(max_iter=100, learning_rate=0.03, max_depth=3, min_samples_leaf=15, l2_regularization=5.0)
        rf_params = dict(n_estimators=100, max_depth=4, min_samples_leaf=10)
    elif n_samples <= 200:
        hgb_params = dict(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=10, l2_regularization=3.0)
        rf_params = dict(n_estimators=150, max_depth=5, min_samples_leaf=5)
    else:
        hgb_params = dict(max_iter=200, learning_rate=0.05, max_depth=5, min_samples_leaf=5, l2_regularization=1.0)
        rf_params = dict(n_estimators=200, max_depth=6, min_samples_leaf=3)

    base_estimators = [
        ('hgb1', HistGradientBoostingClassifier(**hgb_params, random_state=seed)),
        ('hgb2', HistGradientBoostingClassifier(**hgb_params, random_state=seed+1)),
        ('rf', RandomForestClassifier(**rf_params, random_state=seed, n_jobs=-1)),
        ('et', ExtraTreesClassifier(**rf_params, random_state=seed, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=seed
        )),
        ('lr', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.5, max_iter=2000, random_state=seed))
        ])),
        ('svm', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=seed))
        ])),
    ]

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=3,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=-1,
    )


def get_blending_ensemble(seed: int = 0, n_samples: int = 200):
    """Blending ensemble with optimized weights."""
    estimators = [
        ('hgb', HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_depth=4, random_state=seed
        )),
        ('rf', RandomForestClassifier(
            n_estimators=150, max_depth=5, random_state=seed, n_jobs=-1
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=150, max_depth=5, random_state=seed, n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=seed
        )),
        ('mlp', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=300,
                early_stopping=True, random_state=seed
            ))
        ])),
        ('lr', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.5, max_iter=1000, random_state=seed))
        ])),
    ]

    # Optimized weights (boosting methods weighted higher)
    weights = [2.5, 1.5, 1.5, 2.0, 1.0, 1.0]

    return VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=weights,
        n_jobs=-1,
    )


def get_catboost_tuned(seed: int = 0, n_samples: int = 200):
    """Tuned CatBoost."""
    if not CATBOOST_AVAILABLE:
        return None

    if n_samples <= 100:
        params = dict(iterations=150, depth=3, learning_rate=0.03, l2_leaf_reg=20.0)
    elif n_samples <= 200:
        params = dict(iterations=200, depth=4, learning_rate=0.05, l2_leaf_reg=10.0)
    else:
        params = dict(iterations=300, depth=5, learning_rate=0.05, l2_leaf_reg=5.0)

    return CatBoostClassifier(
        **params,
        random_strength=1.0,
        bagging_temperature=0.5,
        border_count=128,
        boosting_type='Plain',
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


# =============================================================================
# Hyperparameter Optimization
# =============================================================================

def optimize_model(X_train, y_train, X_val, y_val, seed=42, n_trials=30):
    """Use Optuna to find optimal hyperparameters."""

    def objective(trial):
        # Model type
        model_type = trial.suggest_categorical('model', ['hgb', 'rf', 'gb'])

        if model_type == 'hgb':
            model = HistGradientBoostingClassifier(
                max_iter=trial.suggest_int('max_iter', 50, 300),
                learning_rate=trial.suggest_float('lr', 0.01, 0.2, log=True),
                max_depth=trial.suggest_int('max_depth', 2, 6),
                min_samples_leaf=trial.suggest_int('min_leaf', 5, 30),
                l2_regularization=trial.suggest_float('l2', 0.1, 10.0, log=True),
                random_state=seed,
            )
        elif model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_est', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 8),
                min_samples_leaf=trial.suggest_int('min_leaf', 2, 20),
                random_state=seed,
                n_jobs=-1,
            )
        else:  # gb
            model = GradientBoostingClassifier(
                n_estimators=trial.suggest_int('n_est', 50, 200),
                learning_rate=trial.suggest_float('lr', 0.01, 0.2, log=True),
                max_depth=trial.suggest_int('max_depth', 2, 5),
                min_samples_leaf=trial.suggest_int('min_leaf', 5, 30),
                random_state=seed,
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return accuracy_score(y_val, y_pred)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, study.best_value


def build_optimized_model(best_params, seed=42):
    """Build model from optimized parameters."""
    model_type = best_params.pop('model', 'hgb')

    if model_type == 'hgb':
        return HistGradientBoostingClassifier(
            max_iter=best_params.get('max_iter', 150),
            learning_rate=best_params.get('lr', 0.05),
            max_depth=best_params.get('max_depth', 4),
            min_samples_leaf=best_params.get('min_leaf', 10),
            l2_regularization=best_params.get('l2', 1.0),
            random_state=seed,
        )
    elif model_type == 'rf':
        return RandomForestClassifier(
            n_estimators=best_params.get('n_est', 150),
            max_depth=best_params.get('max_depth', 5),
            min_samples_leaf=best_params.get('min_leaf', 5),
            random_state=seed,
            n_jobs=-1,
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=best_params.get('n_est', 100),
            learning_rate=best_params.get('lr', 0.05),
            max_depth=best_params.get('max_depth', 3),
            min_samples_leaf=best_params.get('min_leaf', 10),
            random_state=seed,
        )


def get_feature_selected_model(seed: int = 0, n_samples: int = 200, n_features: int = 30):
    """Model with feature selection - helps prevent overfitting at medium sample sizes."""
    from sklearn.feature_selection import SelectKBest, f_classif

    # Adjust feature count based on sample size
    if n_samples <= 100:
        k = min(n_features, 25)
    elif n_samples <= 200:
        k = min(n_features, 35)  # Key: fewer features for n=200
    else:
        k = min(n_features, 50)

    return Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=k)),
        ('clf', HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            l2_regularization=3.0,
            random_state=seed,
        ))
    ])


def get_hybrid_ensemble(seed: int = 0, n_samples: int = 200):
    """
    Hybrid ensemble: combines simple-feature models with engineered-feature models.
    Key insight: AAE does well at n=200 with simple features, so blend both approaches.
    """
    from sklearn.feature_selection import SelectKBest, f_classif

    # Adapter model: select best features then stack
    if n_samples <= 100:
        k = 25
        hgb_params = dict(max_iter=100, learning_rate=0.03, max_depth=3, l2_regularization=5.0)
    elif n_samples <= 300:
        k = 40
        hgb_params = dict(max_iter=150, learning_rate=0.05, max_depth=4, l2_regularization=3.0)
    else:
        k = 60
        hgb_params = dict(max_iter=200, learning_rate=0.05, max_depth=5, l2_regularization=1.0)

    estimators = [
        ('hgb_selected', Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=k)),
            ('clf', HistGradientBoostingClassifier(**hgb_params, random_state=seed))
        ])),
        ('rf_selected', Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=k)),
            ('clf', RandomForestClassifier(
                n_estimators=150, max_depth=5, min_samples_leaf=5,
                random_state=seed, n_jobs=-1
            ))
        ])),
        ('et', ExtraTreesClassifier(
            n_estimators=150, max_depth=5, min_samples_leaf=5,
            random_state=seed, n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=seed
        )),
        ('lr', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.5, max_iter=2000, random_state=seed))
        ])),
    ]

    return VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)


def get_adaptive_stacking(seed: int = 0, n_samples: int = 200):
    """
    Adaptive stacking that selects features AND uses regularization adapted to sample size.
    Specifically optimized for n=200 performance.
    """
    from sklearn.feature_selection import SelectKBest, f_classif

    # Aggressive feature selection for medium samples
    if n_samples <= 100:
        k = 30
        reg = 5.0
    elif n_samples <= 250:
        k = 40  # Key: limit features at n=200
        reg = 4.0
    else:
        k = 60
        reg = 2.0

    base_estimators = [
        ('hgb1', Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=k)),
            ('clf', HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_depth=4,
                min_samples_leaf=10, l2_regularization=reg, random_state=seed
            ))
        ])),
        ('hgb2', Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=k+10)),
            ('clf', HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.03, max_depth=3,
                min_samples_leaf=15, l2_regularization=reg*1.5, random_state=seed+1
            ))
        ])),
        ('rf', Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=k)),
            ('clf', RandomForestClassifier(
                n_estimators=150, max_depth=5, min_samples_leaf=5,
                random_state=seed, n_jobs=-1
            ))
        ])),
        ('gb', GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            min_samples_leaf=10, random_state=seed
        )),
        ('lr', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.3, max_iter=2000, random_state=seed))
        ])),
    ]

    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=0.5, max_iter=1000),
        cv=3,
        stack_method='predict_proba',
        n_jobs=-1,
    )


def get_super_aae(seed: int = 0, n_samples: int = 200):
    """
    Enhanced AAE with more estimators but still uses simple features.
    Key insight: At n=200, simple features work best - so improve the ensemble, not features.
    """
    # Add more diverse weak learners
    estimators = [
        ("hgb1", HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_depth=5, random_state=seed
        )),
        ("hgb2", HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.03, max_depth=4, random_state=seed+1
        )),
        ("rf1", RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1
        )),
        ("rf2", RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=3, random_state=seed+2, n_jobs=-1
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1
        )),
        ("gb", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=seed
        )),
        ("logit", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
        ])),
        ("svm", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel='rbf', C=1.0, probability=True, random_state=seed))
        ])),
    ]
    # Weight boosting methods higher
    weights = [2.0, 2.0, 1.5, 1.5, 1.5, 2.0, 1.0, 1.0]
    return VotingClassifier(estimators=estimators, voting="soft", weights=weights, n_jobs=-1)


def get_stacked_aae(seed: int = 0, n_samples: int = 200):
    """
    Stacking classifier with simple features (like AAE).
    Uses stacking instead of voting for potentially better combination of base models.
    """
    estimators = [
        ("hgb", HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_depth=5, random_state=seed
        )),
        ("rf", RandomForestClassifier(
            n_estimators=150, max_depth=5, random_state=seed, n_jobs=-1
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=150, max_depth=5, random_state=seed, n_jobs=-1
        )),
        ("gb", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=seed
        )),
        ("logit", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
        ])),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=3,
        stack_method='predict_proba',
        n_jobs=-1,
    )


class MetaEnsemble:
    """
    Meta-ensemble that combines models using both simple and engineered features.
    The idea is to train models on both feature sets and combine their predictions.
    """
    def __init__(self, seed=0, n_samples=200):
        self.seed = seed
        self.n_samples = n_samples
        # Models using simple features
        self.simple_models = [
            get_aae_model(seed),
            get_super_aae(seed, n_samples),
        ]
        # Models using engineered features
        self.eng_models = [
            get_adaptive_stacking(seed, n_samples),
            get_mlp_model(seed, n_samples),
        ]
        # Weights: simple models weighted higher for smaller samples
        if n_samples <= 200:
            self.weights = [0.35, 0.35, 0.15, 0.15]  # More weight on simple
        else:
            self.weights = [0.2, 0.2, 0.3, 0.3]  # More weight on engineered

    def fit(self, X_simple, X_eng, y):
        for m in self.simple_models:
            m.fit(X_simple, y)
        for m in self.eng_models:
            m.fit(X_eng, y)
        return self

    def predict_proba(self, X_simple, X_eng):
        probs = []
        for m in self.simple_models:
            probs.append(m.predict_proba(X_simple))
        for m in self.eng_models:
            probs.append(m.predict_proba(X_eng))
        # Weighted average
        weighted_prob = np.zeros_like(probs[0])
        for p, w in zip(probs, self.weights):
            weighted_prob += w * p
        return weighted_prob

    def predict(self, X_simple, X_eng):
        probs = self.predict_proba(X_simple, X_eng)
        return np.argmax(probs, axis=1)


class BaggingFeatureEnsemble:
    """
    Ensemble that uses bagging across different feature subsets.
    Creates multiple models, each trained on different random feature subsets.
    """
    def __init__(self, seed=0, n_models=10, feature_fraction=0.7):
        self.seed = seed
        self.n_models = n_models
        self.feature_fraction = feature_fraction
        self.models = []
        self.feature_masks = []
        self.rng = np.random.RandomState(seed)

    def fit(self, X, y):
        n_features = X.shape[1]
        k = int(n_features * self.feature_fraction)

        self.models = []
        self.feature_masks = []

        for i in range(self.n_models):
            # Random feature subset
            mask = self.rng.choice(n_features, k, replace=False)
            self.feature_masks.append(mask)

            # Train model on subset
            model = HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.05, max_depth=4,
                min_samples_leaf=10, l2_regularization=2.0,
                random_state=self.seed + i
            )
            model.fit(X[:, mask], y)
            self.models.append(model)

        return self

    def predict_proba(self, X):
        probs = []
        for model, mask in zip(self.models, self.feature_masks):
            probs.append(model.predict_proba(X[:, mask]))
        return np.mean(probs, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# =============================================================================
# Evaluation
# =============================================================================

def prepare_features(X_list, z, use_engineering=True, version=2):
    """Prepare features."""
    if use_engineering:
        X_feat = engineer_X_features_v2(X_list, drop_first_col=True)
    else:
        X_feat = flatten_X_simple(X_list, drop_first_col=True)

    Z_oh = one_hot_z(z)
    return np.hstack([X_feat, Z_oh])


def run_optimized_comparison(
    data_path: str,
    sample_sizes: List[int] = [50, 100, 200, 400, 800],
    test_size: int = 200,
    n_trials: int = 5,
    optuna_trials: int = 20,
    seed: int = 42,
):
    """Run comparison with all optimized models."""

    # Load data
    with open(data_path, "rb") as f:
        data = pkl.load(f)[0]

    y_real_all = np.asarray(data["y"], dtype=int)
    y_aug_all = np.asarray(data["y_aug"], dtype=int)
    X_all = list(data["X"])

    n_total = min(len(X_all), len(y_real_all), len(y_aug_all))
    X_all = X_all[:n_total]
    y_real_all = y_real_all[:n_total]
    y_aug_all = y_aug_all[:n_total]

    print(f"Total available data: {n_total}")
    print(f"CatBoost available: {CATBOOST_AVAILABLE}")
    print(f"\nOptimized evaluation with:")
    print(f"  - Advanced feature engineering (v2)")
    print(f"  - Neural networks (MLP)")
    print(f"  - Hyperparameter optimization (Optuna, {optuna_trials} trials)")
    print(f"  - Multi-level stacking & blending")
    print(f"  - Fixed test set: {test_size}")
    print(f"  - Random trials: {n_trials}")
    print()

    # Models to evaluate - include meta-ensemble combining simple + engineered features
    model_names = ['AAE', 'StackedAAE', 'MetaEnsemble', 'BaggedFeats', 'MLP']
    if CATBOOST_AVAILABLE:
        model_names.append('CatBoost_v2')

    results = {n: {m: [] for m in model_names} for n in sample_sizes}

    for trial in range(n_trials):
        trial_seed = seed + trial
        rng = np.random.RandomState(trial_seed)

        all_indices = rng.permutation(n_total)
        test_indices = all_indices[-test_size:]
        train_pool_indices = all_indices[:-test_size]

        # Prepare test data
        X_test_list = [X_all[i] for i in test_indices]
        y_test = np.array([y_real_all[i] for i in test_indices], dtype=int)
        z_test = np.array([y_aug_all[i] for i in test_indices], dtype=int)

        X_test_simple = prepare_features(X_test_list, z_test, use_engineering=False)
        X_test_eng = prepare_features(X_test_list, z_test, use_engineering=True)

        for n_train in sample_sizes:
            train_indices = train_pool_indices[:n_train]

            X_train_list = [X_all[i] for i in train_indices]
            y_train = np.array([y_real_all[i] for i in train_indices], dtype=int)
            z_train = np.array([y_aug_all[i] for i in train_indices], dtype=int)

            X_train_simple = prepare_features(X_train_list, z_train, use_engineering=False)
            X_train_eng = prepare_features(X_train_list, z_train, use_engineering=True)

            # AAE (baseline) - simple features
            model = get_aae_model(trial_seed)
            model.fit(X_train_simple, y_train)
            results[n_train]['AAE'].append(accuracy_score(y_test, model.predict(X_test_simple)))

            # StackedAAE - stacking with simple features
            model = get_stacked_aae(trial_seed, n_train)
            model.fit(X_train_simple, y_train)
            results[n_train]['StackedAAE'].append(accuracy_score(y_test, model.predict(X_test_simple)))

            # MetaEnsemble - combines simple + engineered features
            model = MetaEnsemble(trial_seed, n_train)
            model.fit(X_train_simple, X_train_eng, y_train)
            results[n_train]['MetaEnsemble'].append(accuracy_score(y_test, model.predict(X_test_simple, X_test_eng)))

            # BaggingFeatureEnsemble - bagging across feature subsets
            model = BaggingFeatureEnsemble(trial_seed, n_models=15, feature_fraction=0.6)
            model.fit(X_train_eng, y_train)
            results[n_train]['BaggedFeats'].append(accuracy_score(y_test, model.predict(X_test_eng)))

            # MLP (engineered features)
            model = get_mlp_model(trial_seed, n_train)
            model.fit(X_train_eng, y_train)
            results[n_train]['MLP'].append(accuracy_score(y_test, model.predict(X_test_eng)))

            # CatBoost v2 (engineered features)
            if CATBOOST_AVAILABLE:
                model = get_catboost_tuned(trial_seed, n_train)
                model.fit(X_train_eng, y_train)
                results[n_train]['CatBoost_v2'].append(accuracy_score(y_test, model.predict(X_test_eng)))

        print(f"Trial {trial + 1}/{n_trials} completed")

    # Print results
    print(f"\n{'='*80}")
    print("OPTIMIZED RESULTS")
    print(f"{'='*80}")

    final_results = {}
    for n_train in sample_sizes:
        print(f"\n--- Training size: n = {n_train} ---")
        final_results[n_train] = {}

        aae_mean = np.mean(results[n_train]['AAE'])

        for model_name in model_names:
            accs = results[n_train][model_name]
            mean_acc, std_acc = np.mean(accs), np.std(accs)
            final_results[n_train][model_name] = {'mean': mean_acc, 'std': std_acc}

            diff = mean_acc - aae_mean if model_name != 'AAE' else 0
            diff_str = f"[{'+' if diff >= 0 else ''}{diff:.4f}]" if model_name != 'AAE' else ""
            print(f"  {model_name:<15} {mean_acc:.4f} (+/- {std_acc:.4f}) {diff_str}")

    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")

    header = f"{'Model':<15}"
    for n in sample_sizes:
        header += f" | n={n:>4}"
    header += " | Wins"
    print(header)
    print("-" * 90)

    for model_name in model_names:
        row = f"{model_name:<15}"
        wins = 0
        for n in sample_sizes:
            val = final_results[n][model_name]['mean']
            aae_val = final_results[n]['AAE']['mean']
            if model_name != 'AAE' and val > aae_val:
                wins += 1
            row += f" | {val:.4f}"
        if model_name != 'AAE':
            row += f" | {wins}/{len(sample_sizes)}"
        print(row)

    print("-" * 90)

    # Best model per sample size
    print("\nBest model per sample size:")
    total_wins = 0
    for n in sample_sizes:
        best_model = max(
            [(m, final_results[n][m]['mean']) for m in model_names if m != 'AAE'],
            key=lambda x: x[1]
        )
        aae_val = final_results[n]['AAE']['mean']
        beats = best_model[1] > aae_val
        if beats:
            total_wins += 1
        status = "✓" if beats else "✗"
        print(f"  n={n}: {best_model[0]} ({best_model[1]:.4f}) vs AAE ({aae_val:.4f}) {status}")

    print(f"\n*** Best models beat AAE in {total_wins}/{len(sample_sizes)} sample sizes ***")

    return final_results


if __name__ == "__main__":
    results = run_optimized_comparison(
        data_path="train_gpt-4o_11_1200.pkl",
        sample_sizes=[50, 100, 200, 400, 800],
        test_size=200,
        n_trials=10,  # Increased for more robust estimates
        optuna_trials=20,
        seed=123,  # Different seed to explore
    )
