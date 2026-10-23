"""Module 2 — construction des variables et séparation chronologique.

Toutes les variables comportementales sont **causales** : pour une transaction à l'instant t,
elles n'utilisent que les transactions de la même carte strictement antérieures (la
transaction courante est exclue des fenêtres). Elles peuvent donc être calculées sur le jeu
complet avant la séparation sans fuite d'information du futur vers le passé ; c'est aussi
ce qu'un système en production connaît au moment d'autoriser la transaction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import AMOUNT_COL, CARD_COL, CATEGORY_COL, RANDOM_STATE, SPLIT_FRACTIONS, TARGET, TIME_COL

HOUR, DAY = 3600, 86400
TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_night", "is_weekend"]
BEHAVIOR_FEATURES = ["n_tx_1h", "n_tx_24h", "amt_ratio_30d", "log_secs_since_prev", "amt_sum_24h_log"]
PROFILE_FEATURES = ["log_amt", "distance_km", "age", "log_city_pop", "gender_m"]
CATEGORICAL_FEATURES = [CATEGORY_COL]

FEATURE_DESCRIPTIONS = {
    "hour_sin": "heure (sinus)", "hour_cos": "heure (cosinus)",
    "dow_sin": "jour de semaine (sinus)", "dow_cos": "jour de semaine (cosinus)",
    "is_night": "transaction nocturne (22 h – 4 h)", "is_weekend": "week-end",
    "n_tx_1h": "nb de transactions de la carte dans l'heure précédente",
    "n_tx_24h": "nb de transactions de la carte dans les 24 h précédentes",
    "amt_ratio_30d": "montant / moyenne de la carte sur 30 jours",
    "log_secs_since_prev": "temps écoulé depuis la transaction précédente (log s)",
    "amt_sum_24h_log": "montant cumulé de la carte sur 24 h (log)",
    "log_amt": "montant (log)", "distance_km": "distance porteur – marchand (km)",
    "age": "âge du porteur", "log_city_pop": "population de la ville (log)", "gender_m": "porteur masculin",
}


# ---------------------------------------------------------------- variables
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df[TIME_COL]
    hour = ts.dt.hour + ts.dt.minute / 60
    dow = ts.dt.dayofweek
    return df.assign(
        hour_sin=np.sin(2 * np.pi * hour / 24), hour_cos=np.cos(2 * np.pi * hour / 24),
        dow_sin=np.sin(2 * np.pi * dow / 7), dow_cos=np.cos(2 * np.pi * dow / 7),
        is_night=((ts.dt.hour >= 22) | (ts.dt.hour < 4)).astype(int),
        is_weekend=(dow >= 5).astype(int),
    )


def add_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vélocité et écart au profil par carte, en O(n log n) via `searchsorted`.

    On trie par (carte, temps) et on construit une clé monotone `carte * SPAN + t`. Pour une
    fenêtre de durée w, le début de fenêtre est `searchsorted(clé - w)` ; la marge de SPAN
    garantit qu'on ne déborde jamais sur la carte précédente.
    """
    order = np.lexsort((df[TIME_COL].to_numpy(), df[CARD_COL].to_numpy()))
    d = df.iloc[order]
    t = (d[TIME_COL] - d[TIME_COL].min()).dt.total_seconds().to_numpy().astype(np.int64)
    card = pd.factorize(d[CARD_COL])[0].astype(np.int64)
    span = int(t.max()) + 40 * DAY
    key = card * span + t
    pos = np.arange(len(d))
    amt = d[AMOUNT_COL].to_numpy(dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(amt)])

    def window_start(w):
        return np.searchsorted(key, key - w, side="left")

    l1h, l24h, l30d = window_start(HOUR), window_start(DAY), window_start(30 * DAY)
    n30 = pos - l30d
    mean30 = np.divide(cs[pos] - cs[l30d], n30, out=np.full(len(d), np.nan), where=n30 > 0)
    same_prev = np.r_[False, card[1:] == card[:-1]]
    gap = np.where(same_prev, t - np.r_[0, t[:-1]], 30 * DAY)

    feats = pd.DataFrame({
        "n_tx_1h": pos - l1h,
        "n_tx_24h": pos - l24h,
        "amt_ratio_30d": np.where(np.isnan(mean30), 1.0, amt / np.where(np.isnan(mean30), 1.0, mean30)),
        "log_secs_since_prev": np.log1p(gap),
        "amt_sum_24h_log": np.log1p(cs[pos] - cs[l24h]),
    }, index=d.index)
    return df.join(feats)


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def add_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.assign(log_amt=np.log1p(df[AMOUNT_COL]))
    if {"lat", "long", "merch_lat", "merch_long"} <= set(df.columns):
        out["distance_km"] = _haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    if "dob" in df.columns and TIME_COL in df.columns:
        out["age"] = (df[TIME_COL] - pd.to_datetime(df["dob"])).dt.days / 365.25
    if "city_pop" in df.columns:
        out["log_city_pop"] = np.log1p(df["city_pop"])
    if "gender" in df.columns:
        out["gender_m"] = (df["gender"] == "M").astype(int)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute toutes les variables disponibles selon les colonnes présentes."""
    out = add_profile_features(df)
    if TIME_COL in df.columns:
        out = add_time_features(out)
        if CARD_COL in df.columns:
            out = add_behavior_features(out)
    return out


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Variables numériques et catégorielles effectivement présentes."""
    num = [c for c in TIME_FEATURES + BEHAVIOR_FEATURES + PROFILE_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return num, cat


# ---------------------------------------------------------------- séparation
def chronological_split(df: pd.DataFrame, fractions=SPLIT_FRACTIONS, seed: int = RANDOM_STATE):
    """60/20/20 dans l'ordre du temps ; aléatoire stratifiée (signalée) faute de temps.

    Retourne (train, val, test, info) où info décrit la méthode et les bornes de dates.
    """
    f_train, f_val, _ = fractions
    if TIME_COL in df.columns and df[TIME_COL].notna().all():
        d = df.sort_values(TIME_COL, kind="stable")
        n = len(d)
        i1, i2 = int(n * f_train), int(n * (f_train + f_val))
        parts = d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]
        info = {"method": "chronologique", "warning": None}
        for name, p in zip(("train", "val", "test"), parts):
            info[name] = {"start": p[TIME_COL].min(), "end": p[TIME_COL].max(), "n": len(p),
                          "frauds": int(p[TARGET].sum()), "fraud_rate": float(p[TARGET].mean())}
        return (*parts, info)
    tr, rest = train_test_split(df, train_size=f_train, stratify=df[TARGET], random_state=seed)
    va, te = train_test_split(rest, train_size=f_val / (1 - f_train), stratify=rest[TARGET], random_state=seed)
    info = {"method": "aléatoire stratifiée",
            "warning": "Aucune information temporelle : séparation aléatoire stratifiée (risque d'optimisme)."}
    for name, p in zip(("train", "val", "test"), (tr, va, te)):
        info[name] = {"n": len(p), "frauds": int(p[TARGET].sum()), "fraud_rate": float(p[TARGET].mean())}
    return tr, va, te, info


def make_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    """Standardisation + one-hot ; à ajuster sur l'entraînement uniquement."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ], verbose_feature_names_out=False)
    return pre.set_output(transform="pandas")
