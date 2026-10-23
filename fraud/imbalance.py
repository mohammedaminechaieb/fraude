"""Module 2 — stratégies contre le déséquilibre, appliquées au seul jeu d'entraînement.

Les stratégies sont comparées sur la validation (proportion réelle de fraudes) par la PR-AUC.
Après un sous-échantillonnage qui garde une proportion β des transactions normales, la cote de
fraude apprise est multipliée par 1/β ; on corrige les probabilités par Bayes :
    p = β·ps / (β·ps − ps + 1)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import RANDOM_STATE

STRATEGIES = ["none", "class_weight", "under_1:5", "under_1:10", "under_1:20", "smote"]
STRATEGY_LABELS = {
    "none": "Aucun rééchantillonnage",
    "class_weight": "Pondération des classes",
    "under_1:5": "Sous-échantillonnage 1:5",
    "under_1:10": "Sous-échantillonnage 1:10",
    "under_1:20": "Sous-échantillonnage 1:20",
    "smote": "SMOTE (fraudes = 10 % des normales)",
}


def default_classifier(**kw) -> LGBMClassifier:
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    params.update(kw)
    return LGBMClassifier(**params)


def correct_proba(ps: np.ndarray, beta: float) -> np.ndarray:
    """Ramène des probabilités apprises après sous-échantillonnage à la prévalence réelle."""
    ps = np.asarray(ps, dtype=float)
    return beta * ps / (beta * ps - ps + 1)


def resample(strategy: str, X: pd.DataFrame, y: pd.Series, seed: int = RANDOM_STATE):
    """Applique une stratégie. Retourne (X, y, params du modèle, β pour la correction)."""
    y = pd.Series(np.asarray(y), index=X.index)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if strategy == "none":
        return X, y, {}, 1.0
    if strategy == "class_weight":
        return X, y, {"scale_pos_weight": n_neg / max(n_pos, 1)}, 1.0
    if strategy.startswith("under_1:"):
        ratio = int(strategy.split(":")[1])
        n_keep = min(n_neg, ratio * n_pos)
        rng = np.random.default_rng(seed)
        neg_idx = rng.choice(np.flatnonzero(y.to_numpy() == 0), n_keep, replace=False)
        idx = np.sort(np.r_[np.flatnonzero(y.to_numpy() == 1), neg_idx])
        return X.iloc[idx], y.iloc[idx], {}, n_keep / n_neg
    if strategy == "smote":
        Xr, yr = SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=seed).fit_resample(X, y)
        return Xr, yr, {}, 1.0
    raise ValueError(f"Stratégie inconnue : {strategy}")


def compare_strategies(X_tr, y_tr, X_val, y_val, strategies=STRATEGIES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Entraîne le modèle de référence avec chaque stratégie et mesure la validation."""
    rows = []
    for s in strategies:
        t = time.perf_counter()
        Xr, yr, params, beta = resample(s, X_tr, y_tr, seed)
        model = default_classifier(**params).fit(Xr, yr)
        p = correct_proba(model.predict_proba(X_val)[:, 1], beta)
        rows.append({
            "strategy": s, "label": STRATEGY_LABELS[s], "n_train": len(yr),
            "fraud_share_train": float(np.mean(yr)), "beta": beta,
            "pr_auc_val": average_precision_score(y_val, p), "roc_auc_val": roc_auc_score(y_val, p),
            "fit_seconds": time.perf_counter() - t,
        })
    return pd.DataFrame(rows).sort_values("pr_auc_val", ascending=False).reset_index(drop=True)
