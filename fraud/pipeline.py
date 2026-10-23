"""Orchestration : données → variables → séparation → modèles → artefacts.

    python -m fraud.pipeline --source synthetic
    python -m fraud.pipeline --source kaggle --n-cards 300
    python -m fraud.pipeline --csv data/fraudTrain.csv
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import __version__
from .config import AMOUNT_COL, ARTIFACTS_DIR, CARD_COL, CATEGORY_COL, RANDOM_STATE, TARGET, TIME_COL
from .data import load_dataset
from .features import build_features, chronological_split, feature_columns, make_preprocessor
from .imbalance import compare_strategies
from .models import build_detectors

BUNDLE_PATH = ARTIFACTS_DIR / "bundle.joblib"
ID_COLS = ["trans_num", CARD_COL, TIME_COL, AMOUNT_COL, CATEGORY_COL, "merchant", TARGET]


def _keep_cols(df: pd.DataFrame, num, cat) -> list[str]:
    return [c for c in dict.fromkeys(ID_COLS + num + cat) if c in df.columns]


def train(df: pd.DataFrame, strategy: str = "auto", seed: int = RANDOM_STATE, progress=None) -> dict:
    """Entraîne tous les détecteurs. `strategy='auto'` choisit le déséquilibre en validation."""
    log = progress or (lambda msg: None)
    t0 = time.perf_counter()
    log("Construction des variables")
    feats = build_features(df)
    tr, va, te, split_info = chronological_split(feats, seed=seed)
    num, cat = feature_columns(feats)
    pre = make_preprocessor(num, cat).fit(tr)
    X = {k: pre.transform(p) for k, p in (("train", tr), ("val", va), ("test", te))}
    y = {"train": tr[TARGET].to_numpy(), "val": va[TARGET].to_numpy(), "test": te[TARGET].to_numpy()}

    imbalance = None
    if strategy == "auto":
        log("Comparaison des stratégies de déséquilibre")
        imbalance = compare_strategies(X["train"], y["train"], X["val"], y["val"], seed=seed)
        strategy = imbalance.iloc[0]["strategy"]

    detectors = build_detectors(strategy, seed)
    frames = {"val": va[_keep_cols(va, num, cat)].copy(), "test": te[_keep_cols(te, num, cat)].copy()}
    timings = {}
    for name, det in detectors.items():
        log(f"Entraînement : {name}")
        t = time.perf_counter()
        det.fit(X["train"], y["train"])
        timings[name] = time.perf_counter() - t
        for split in ("val", "test"):
            frames[split][f"score::{name}"] = det.score(X[split])

    return {
        "version": __version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "preprocessor": pre, "num_features": num, "cat_features": cat,
        "feature_names": list(X["train"].columns),
        "strategy": strategy, "imbalance": imbalance, "split_info": split_info,
        "detectors": detectors, "fit_seconds": timings, "frames": frames,
        "train_sample": X["train"].sample(min(2000, len(X["train"])), random_state=seed),
        "duration_seconds": time.perf_counter() - t0,
    }


def model_names(bundle: dict) -> list[str]:
    return list(bundle["detectors"])


def scores(bundle: dict, split: str, name: str) -> np.ndarray:
    return bundle["frames"][split][f"score::{name}"].to_numpy()


def comparison_table(bundle: dict) -> pd.DataFrame:
    rows = []
    for name, det in bundle["detectors"].items():
        row = {"méthode": name, "famille": det.family, "entraînement (s)": bundle["fit_seconds"][name]}
        for split in ("val", "test"):
            yt = bundle["frames"][split][TARGET]
            s = scores(bundle, split, name)
            row[f"PR-AUC {split}"] = average_precision_score(yt, s)
            row[f"ROC-AUC {split}"] = roc_auc_score(yt, s)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("PR-AUC val", ascending=False).reset_index(drop=True)


def save_bundle(bundle: dict, path: Path = BUNDLE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: Path = BUNDLE_PATH) -> dict:
    return joblib.load(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Entraîne tous les détecteurs et écrit artifacts/bundle.joblib")
    ap.add_argument("--source", choices=["synthetic", "kaggle"], default="synthetic")
    ap.add_argument("--csv", help="CSV au schéma Sparkov (prioritaire sur --source)")
    ap.add_argument("--n-cards", type=int, default=None, help="échantillon de cartes")
    ap.add_argument("--strategy", default="auto", help="auto | none | class_weight | under_1:10 | smote ...")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--out", default=str(BUNDLE_PATH))
    args = ap.parse_args(argv)

    df = load_dataset(args.source, args.csv, args.n_cards, args.seed)
    print(f"{len(df):,} transactions, {df[TARGET].mean():.3%} de fraudes")
    bundle = train(df, args.strategy, args.seed, progress=print)
    if bundle["imbalance"] is not None:
        print(bundle["imbalance"][["label", "pr_auc_val", "roc_auc_val"]].to_string(index=False))
    print(f"Stratégie retenue : {bundle['strategy']}")
    print(comparison_table(bundle).to_string(index=False, float_format="%.4f"))
    print(f"Artefacts : {save_bundle(bundle, args.out)}")
    return bundle


if __name__ == "__main__":
    main()
