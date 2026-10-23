"""Module 1 — chargement des données et statistiques exploratoires."""
from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from .config import AMOUNT_COL, CARD_COL, KAGGLE_DATASET, REQUIRED_COLUMNS, TARGET, TIME_COL
from .synthetic import generate_synthetic


# ---------------------------------------------------------------- chargement
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie un jeu au schéma Sparkov : types, tri chronologique, colonnes parasites."""
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes obligatoires absentes : {missing}")
    df = df.copy()
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        df = df.sort_values(TIME_COL, kind="stable")
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"])
    df[TARGET] = df[TARGET].astype(int)
    df[AMOUNT_COL] = df[AMOUNT_COL].astype(float)
    return df.reset_index(drop=True)


def has_time_info(df: pd.DataFrame) -> bool:
    return TIME_COL in df.columns and df[TIME_COL].notna().all()


def sample_by_card(df: pd.DataFrame, n_cards: int | None, seed: int = 42) -> pd.DataFrame:
    """Échantillonne des cartes entières : l'historique de chaque porteur reste complet."""
    cards = df[CARD_COL].unique()
    if n_cards is None or n_cards >= len(cards):
        return df
    keep = np.random.default_rng(seed).choice(cards, n_cards, replace=False)
    return df[df[CARD_COL].isin(keep)].reset_index(drop=True)


def load_csv(source: str | Path | IO) -> pd.DataFrame:
    return normalize(pd.read_csv(source))


def load_kaggle(n_cards: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Télécharge le jeu Sparkov depuis Kaggle (identifiants Kaggle requis) et fusionne train + test."""
    import kagglehub

    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    files = sorted(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {path}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return normalize(sample_by_card(df, n_cards, seed))


def load_synthetic(n_cards: int = 400, seed: int = 42) -> pd.DataFrame:
    return normalize(generate_synthetic(n_cards=n_cards, seed=seed))


def load_dataset(source: str = "synthetic", csv: str | Path | None = None,
                 n_cards: int | None = None, seed: int = 42) -> pd.DataFrame:
    if csv is not None:
        return normalize(sample_by_card(pd.read_csv(csv), n_cards, seed))
    if source == "kaggle":
        return load_kaggle(n_cards, seed)
    if source == "synthetic":
        return load_synthetic(n_cards or 400, seed)
    raise ValueError(f"Source inconnue : {source}")


# ---------------------------------------------------------------- exploration
def summary(df: pd.DataFrame) -> dict:
    out = {
        "n_transactions": int(len(df)),
        "n_frauds": int(df[TARGET].sum()),
        "fraud_rate": float(df[TARGET].mean()),
        "n_cards": int(df[CARD_COL].nunique()),
        "fraud_amount_total": float(df.loc[df[TARGET] == 1, AMOUNT_COL].sum()),
        "has_time": bool(has_time_info(df)),
    }
    if out["has_time"]:
        out["start"] = df[TIME_COL].min()
        out["end"] = df[TIME_COL].max()
        out["n_days"] = int((out["end"] - out["start"]).days) + 1
    return out


def amount_stats_by_class(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby(TARGET)[AMOUNT_COL].describe(percentiles=[0.25, 0.5, 0.75, 0.95])
    return stats.rename(index={0: "légitime", 1: "fraude"})


def fraud_rate_over_time(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    g = df.groupby(pd.Grouper(key=TIME_COL, freq=freq))[TARGET]
    return pd.DataFrame({"n": g.size(), "frauds": g.sum(), "fraud_rate": g.mean()}).reset_index()


def with_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out[TIME_COL].dt.hour
    out["day_of_week"] = out[TIME_COL].dt.dayofweek
    if "dob" in out.columns:
        age = (out[TIME_COL] - out["dob"]).dt.days / 365.25
        out["age_band"] = pd.cut(age, [0, 25, 35, 50, 65, 120], labels=["<25", "25-34", "35-49", "50-64", "65+"])
    return out


def fraud_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True)[TARGET]
    return (pd.DataFrame({"n": g.size(), "frauds": g.sum(), "fraud_rate": g.mean()})
            .reset_index().sort_values(col if col in ("hour", "day_of_week") else "fraud_rate",
                                       ascending=col in ("hour", "day_of_week")))
