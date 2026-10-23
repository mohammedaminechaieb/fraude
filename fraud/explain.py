"""Module 5 — Explainable AI : SHAP global / local et codes motifs.

Le modèle expliqué est le LightGBM de production. TreeExplainer décompose sa marge (log-cote)
en contributions additives. La correction bayésienne après sous-échantillonnage ajoute log(β)
à la log-cote : on l'intègre à la valeur de base, si bien que base + Σ contributions donne
exactement la log-cote du score publié.

Les variables techniques sont regroupées pour l'analyste : sinus, cosinus de l'heure et indicateur de nuit → « heure »,
indicatrices de catégorie → « catégorie ».
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score

from .config import AMOUNT_COL, CATEGORY_COL, RANDOM_STATE, TIME_COL
from .features import FEATURE_DESCRIPTIONS

GROUP_LABELS = {
    "hour": "heure", "dow": "jour de semaine", "category": "catégorie", "n_tx_1h": "vélocité 1 h", "n_tx_24h": "vélocité 24 h",
    "amt_ratio_30d": "écart au montant habituel", "log_secs_since_prev": "délai depuis la transaction précédente",
    "amt_sum_24h_log": "cumul 24 h", "log_amt": "montant", "distance_km": "distance porteur–marchand",
    "age": "âge", "log_city_pop": "population de la ville", "gender_m": "genre",
}


def feature_group(name: str) -> str:
    if name.startswith("hour_") or name == "is_night":
        return "hour"
    if name.startswith("dow_") or name == "is_weekend":
        return "dow"
    if name.startswith(f"{CATEGORY_COL}_"):
        return "category"
    return name


_EXPLAINERS: dict[int, shap.TreeExplainer] = {}


def _model(bundle):
    return bundle["detectors"]["LightGBM"]


def explainer(bundle) -> shap.TreeExplainer:
    model = _model(bundle).model
    if id(model) not in _EXPLAINERS:
        _EXPLAINERS[id(model)] = shap.TreeExplainer(model)
    return _EXPLAINERS[id(model)]


def shap_explanation(bundle, frame: pd.DataFrame) -> shap.Explanation:
    """Contributions SHAP (log-cote) ; `data` contient les valeurs brutes, lisibles."""
    det = _model(bundle)
    X = bundle["preprocessor"].transform(frame)[det.feature_names]
    exp = explainer(bundle)(X)
    raw = frame.reindex(columns=X.columns).copy()
    for c in X.columns:  # indicatrices de catégorie : valeur 0/1 telle qu'encodée
        if raw[c].isna().all():
            raw[c] = X[c].to_numpy()
    return shap.Explanation(values=exp.values, base_values=exp.base_values + np.log(det.beta),
                            data=raw.to_numpy(dtype=float), feature_names=list(X.columns))


def grouped(exp: shap.Explanation, frame: pd.DataFrame) -> shap.Explanation:
    """Somme les contributions par groupe ; `data` devient une valeur lisible (texte)."""
    groups = list(dict.fromkeys(feature_group(f) for f in exp.feature_names))
    idx = {g: [i for i, f in enumerate(exp.feature_names) if feature_group(f) == g] for g in groups}
    values = np.stack([exp.values[:, idx[g]].sum(axis=1) for g in groups], axis=1)
    data = np.empty((len(frame), len(groups)), dtype=object)
    for j, g in enumerate(groups):
        data[:, j] = [_display_value(g, row) for _, row in frame.iterrows()]
    return shap.Explanation(values=values, base_values=exp.base_values, data=data,
                            feature_names=[GROUP_LABELS.get(g, g) for g in groups])


def _display_value(group: str, row: pd.Series) -> str:
    if group == "hour":
        return f"{pd.Timestamp(row[TIME_COL]):%H:%M}" if TIME_COL in row else "?"
    if group == "dow":
        return ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"][pd.Timestamp(row[TIME_COL]).dayofweek]
    if group == "category":
        return str(row.get(CATEGORY_COL, "?"))
    if group == "log_amt":
        return f"{row[AMOUNT_COL]:.2f}"
    if group == "log_secs_since_prev":
        return _duration(np.expm1(row[group]))
    if group == "amt_sum_24h_log":
        return f"{np.expm1(row[group]):.2f}"
    if group == "log_city_pop":
        return f"{np.expm1(row[group]):,.0f}".replace(",", " ")
    if group == "gender_m":
        return "M" if row[group] == 1 else "F"
    v = row.get(group)
    if v is None or pd.isna(v):
        return "?"
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"


def _duration(secs: float) -> str:
    if secs >= 30 * 86400 - 1:
        return "aucune transaction depuis 30 j"
    if secs < 3600:
        return f"{secs / 60:.0f} min"
    if secs < 86400:
        return f"{secs / 3600:.1f} h"
    return f"{secs / 86400:.1f} j"


def _reason_text(group: str, row: pd.Series) -> str:
    v = row.get(group)
    if group == "log_amt":
        return f"Montant élevé ({row[AMOUNT_COL]:.2f})"
    if group == "amt_ratio_30d":
        return f"Montant {v:.1f} × la moyenne de la carte sur 30 jours"
    if group == "n_tx_1h":
        return f"{int(v)} transaction(s) de la même carte dans l'heure précédente"
    if group == "n_tx_24h":
        return f"{int(v)} transaction(s) de la même carte dans les 24 h précédentes"
    if group == "amt_sum_24h_log":
        return f"Cumul de {np.expm1(v):.2f} dépensé sur la carte en 24 h"
    if group == "log_secs_since_prev":
        return f"Délai depuis la transaction précédente : {_duration(np.expm1(v))}"
    if group == "hour":
        return f"Heure inhabituelle ({pd.Timestamp(row[TIME_COL]):%H:%M})"
    if group == "dow":
        return f"Jour de semaine : {_display_value('dow', row)}"
    if group == "category":
        return f"Catégorie de marchand à risque : {row.get(CATEGORY_COL)}"
    if group == "distance_km":
        return f"Marchand à {v:.0f} km du domicile du porteur"
    if group == "age":
        return f"Âge du porteur : {v:.0f} ans"
    return f"{FEATURE_DESCRIPTIONS.get(group, group)} = {_display_value(group, row)}"


def reason_codes(bundle, frame: pd.DataFrame, n: int = 5, min_share: float = 0.05) -> list[list[dict]]:
    """Pour chaque transaction, les 3 à `n` groupes qui ont le plus poussé vers l'alerte.

    On garde au moins 3 motifs positifs, puis ceux qui pèsent au moins `min_share` du total positif.
    """
    exp = shap_explanation(bundle, frame)
    out = []
    for i, (_, row) in enumerate(frame.iterrows()):
        contrib = {}
        for f, v in zip(exp.feature_names, exp.values[i]):
            contrib[feature_group(f)] = contrib.get(feature_group(f), 0.0) + float(v)
        pos = sorted(((g, v) for g, v in contrib.items() if v > 0), key=lambda t: -t[1])
        total = sum(v for _, v in pos) or 1.0
        keep = [(g, v) for j, (g, v) in enumerate(pos[:n]) if j < 3 or v / total >= min_share]
        out.append([{"code": g.upper(), "feature": GROUP_LABELS.get(g, g), "contribution": round(v, 4),
                     "text": _reason_text(g, row)} for g, v in keep])
    return out


# ---------------------------------------------------------------- importance globale
def permutation_importance(bundle, frame: pd.DataFrame, target: str = "is_fraud", n_repeats: int = 3,
                           max_rows: int = 30_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Baisse de PR-AUC quand on permute un groupe de variables (indicatrices permutées ensemble)."""
    det = _model(bundle)
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=seed)
    X = bundle["preprocessor"].transform(frame)[det.feature_names]
    y = frame[target].to_numpy()
    base = average_precision_score(y, det.score(X))
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    for f in X.columns:
        groups.setdefault(feature_group(f), []).append(f)
    rows = []
    for g, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            drops.append(base - average_precision_score(y, det.score(Xp)))
        rows.append({"variable": GROUP_LABELS.get(g, g), "importance": float(np.mean(drops)),
                     "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def global_sample(frame: pd.DataFrame, target: str = "is_fraud", n: int = 2000, seed: int = RANDOM_STATE):
    """Échantillon pour le beeswarm : toutes les fraudes (jusqu'à n/4) + transactions normales."""
    fraud = frame[frame[target] == 1]
    fraud = fraud.sample(min(len(fraud), n // 4), random_state=seed)
    normal = frame[frame[target] == 0]
    normal = normal.sample(min(len(normal), n - len(fraud)), random_state=seed)
    return pd.concat([fraud, normal])
