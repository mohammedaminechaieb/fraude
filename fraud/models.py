"""Module 3 — détecteurs supervisés et non supervisés.

Chaque détecteur expose `fit(X, y)` et `score(X)` : un score d'anomalie / de risque où
« plus grand = plus suspect ». Les détecteurs non supervisés ignorent les étiquettes, sauf
l'autoencodeur qui, comme dans le cours, apprend sur les seules transactions normales
(détection de nouveauté).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor

from .config import RANDOM_STATE
from .imbalance import correct_proba, default_classifier, resample


def _sample(X: pd.DataFrame, n: int, seed: int, mask=None) -> pd.DataFrame:
    if mask is not None:
        X = X[np.asarray(mask)]
    if len(X) <= n:
        return X
    return X.iloc[np.random.default_rng(seed).choice(len(X), n, replace=False)]


class Detector:
    name = "detector"
    family = "supervisé"
    description = ""

    def fit(self, X: pd.DataFrame, y) -> "Detector":
        raise NotImplementedError

    def score(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class LightGBMDetector(Detector):
    name, family = "LightGBM", "supervisé"
    description = "Gradient boosting d'arbres, stratégie de déséquilibre choisie en validation."

    def __init__(self, strategy: str = "none", seed: int = RANDOM_STATE):
        self.strategy, self.seed = strategy, seed

    def fit(self, X, y):
        Xr, yr, params, self.beta = resample(self.strategy, X, y, self.seed)
        self.model = default_classifier(random_state=self.seed, **params).fit(Xr, yr)
        self.feature_names = list(X.columns)
        return self

    def score(self, X):
        return correct_proba(self.model.predict_proba(X[self.feature_names])[:, 1], self.beta)


class LogisticDetector(Detector):
    name, family = "Régression logistique pondérée", "supervisé"
    description = "Modèle linéaire lisible, class_weight='balanced'."

    def fit(self, X, y):
        self.model = LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5).fit(X, y)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


class IsolationForestDetector(Detector):
    name, family = "Isolation Forest", "non supervisé"
    description = "Moins il faut de coupures aléatoires pour isoler un point, plus il est suspect (ψ=256, t=200)."

    def __init__(self, seed: int = RANDOM_STATE):
        self.seed = seed

    def fit(self, X, y=None):
        self.model = IsolationForest(n_estimators=200, max_samples=256, random_state=self.seed, n_jobs=-1).fit(X)
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class LOFDetector(Detector):
    name, family = "LOF", "non supervisé"
    description = "Densité locale comparée à celle des k=20 voisins (novelty=True, échantillon de 20 000 points)."

    def __init__(self, n_neighbors: int = 20, max_train: int = 20_000, seed: int = RANDOM_STATE):
        self.n_neighbors, self.max_train, self.seed = n_neighbors, max_train, seed

    def fit(self, X, y=None):
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True, n_jobs=-1)
        self.model.fit(_sample(X, self.max_train, self.seed).to_numpy())
        return self

    def score(self, X):
        return -self.model.score_samples(X.to_numpy())


class AutoencoderDetector(Detector):
    name, family = "Autoencodeur", "non supervisé"
    description = "Réseau à goulot (16-6-16) appris sur des transactions normales ; score = erreur de reconstruction."

    def __init__(self, max_train: int = 50_000, seed: int = RANDOM_STATE):
        self.max_train, self.seed = max_train, seed

    def fit(self, X, y=None):
        mask = None if y is None else np.asarray(y) == 0
        Xs = _sample(X, self.max_train, self.seed, mask)
        self.model = MLPRegressor(hidden_layer_sizes=(16, 6, 16), activation="relu", max_iter=200,
                                  early_stopping=True, random_state=self.seed).fit(Xs, Xs)
        return self

    def score(self, X):
        return np.mean((self.model.predict(X) - X.to_numpy()) ** 2, axis=1)


def build_detectors(strategy: str = "none", seed: int = RANDOM_STATE) -> dict[str, Detector]:
    dets = [LightGBMDetector(strategy, seed), LogisticDetector(), IsolationForestDetector(seed),
            LOFDetector(seed=seed), AutoencoderDetector(seed=seed)]
    return {d.name: d for d in dets}
