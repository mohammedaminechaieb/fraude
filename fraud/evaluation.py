"""Module 4 — métriques pour événements rares, budget d'alertes et seuil au coût minimal.

Coût d'un seuil τ (alerte si score ≥ τ) :
    C(τ) = Σ_{fraudes manquées} montant + Ca × N_alertes(τ)
Le seuil est choisi sur la validation puis confirmé sur le test, et comparé au coût « sans
modèle » (aucune alerte : toutes les fraudes passent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, matthews_corrcoef, precision_recall_curve, roc_auc_score

from .config import AMOUNT_COL, BUDGET_FRAUD_MULTIPLIER, DEFAULT_ALERT_COST, DEFAULT_DAILY_BUDGET, TARGET, TIME_COL


# ---------------------------------------------------------------- budget d'alertes
def n_days(frame: pd.DataFrame) -> int:
    if TIME_COL not in frame.columns:
        return 1
    return max(1, int(np.ceil((frame[TIME_COL].max() - frame[TIME_COL].min()) / pd.Timedelta(days=1))))


def suggest_daily_budget(split_info: dict, multiplier: float = BUDGET_FRAUD_MULTIPLIER) -> int:
    """Capacité quotidienne justifiée par les données d'entraînement.

    On dimensionne l'équipe pour examiner environ `multiplier` fois le nombre de fraudes observé
    par jour sur la période d'entraînement : la précision@k plafonne alors à 1/multiplier, mais
    le rappel@k peut approcher 100 % si le classement est bon.
    """
    tr = split_info.get("train", {})
    if "start" not in tr:
        return DEFAULT_DAILY_BUDGET
    days = max(1.0, (tr["end"] - tr["start"]) / pd.Timedelta(days=1))
    return max(1, int(round(multiplier * tr["frauds"] / days)))


def budget_k(frame: pd.DataFrame, daily_budget: int = DEFAULT_DAILY_BUDGET) -> int:
    """k = capacité quotidienne de l'équipe × nombre de jours de la période évaluée."""
    return int(min(len(frame), daily_budget * n_days(frame)))


def precision_recall_at_k(y, s, k: int) -> tuple[float, float]:
    y = np.asarray(y)
    top = np.argsort(-np.asarray(s), kind="stable")[:k]
    tp = y[top].sum()
    return float(tp / max(k, 1)), float(tp / max(y.sum(), 1))


# ---------------------------------------------------------------- seuil
def metrics_at_threshold(y, s, threshold: float, amounts=None, alert_cost: float = DEFAULT_ALERT_COST) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(s) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    out = {
        "threshold": float(threshold), "alerts": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if 0 < pred.sum() < len(pred) else 0.0,
    }
    if amounts is not None:
        amounts = np.asarray(amounts, dtype=float)
        missed = float(amounts[(y == 1) & (pred == 0)].sum())
        out.update(missed_amount=missed, review_cost=alert_cost * out["alerts"],
                   cost=missed + alert_cost * out["alerts"], cost_no_model=float(amounts[y == 1].sum()))
        out["savings"] = out["cost_no_model"] - out["cost"]
    return out


def cost_curve(y, s, amounts, alert_cost: float = DEFAULT_ALERT_COST) -> pd.DataFrame:
    """Coût pour chaque seuil candidat (une ligne par groupe de scores ex aequo), O(n log n)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    fraud_amt = np.where(y[order] == 1, amounts[order], 0.0)
    total = fraud_amt.sum()
    # i alertes = les i premiers scores ; on ne garde que les fins de groupes ex aequo
    last_of_group = np.r_[s_sorted[1:] < s_sorted[:-1], True]
    n_alerts = np.arange(1, len(s) + 1)[last_of_group]
    caught = np.cumsum(fraud_amt)[last_of_group]
    tp = np.cumsum(y[order])[last_of_group]
    curve = pd.DataFrame({
        "threshold": s_sorted[last_of_group], "alerts": n_alerts, "tp": tp,
        "missed_amount": total - caught, "review_cost": alert_cost * n_alerts,
    })
    curve["cost"] = curve["missed_amount"] + curve["review_cost"]
    # seuil « aucune alerte »
    none = pd.DataFrame({"threshold": [np.inf], "alerts": [0], "tp": [0], "missed_amount": [total],
                         "review_cost": [0.0], "cost": [total]})
    return pd.concat([none, curve], ignore_index=True)


def optimal_threshold(y, s, amounts, alert_cost: float = DEFAULT_ALERT_COST) -> float:
    curve = cost_curve(y, s, amounts, alert_cost)
    return float(curve.loc[curve["cost"].idxmin(), "threshold"])


# ---------------------------------------------------------------- évaluation complète
def evaluate_model(val: pd.DataFrame, test: pd.DataFrame, score_col: str,
                   alert_cost: float = DEFAULT_ALERT_COST, daily_budget: int = DEFAULT_DAILY_BUDGET) -> dict:
    tau = optimal_threshold(val[TARGET], val[score_col], val[AMOUNT_COL], alert_cost)
    out = {"threshold": tau}
    for name, frame in (("val", val), ("test", test)):
        y, s = frame[TARGET].to_numpy(), frame[score_col].to_numpy()
        k = budget_k(frame, daily_budget)
        p_k, r_k = precision_recall_at_k(y, s, k)
        m = metrics_at_threshold(y, s, tau, frame[AMOUNT_COL], alert_cost)
        out[name] = {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s), "k": k,
                     "precision_at_k": p_k, "recall_at_k": r_k, **m}
    return out


def evaluation_table(bundle: dict, alert_cost: float = DEFAULT_ALERT_COST,
                     daily_budget: int = DEFAULT_DAILY_BUDGET, split: str = "test") -> pd.DataFrame:
    val, test = bundle["frames"]["val"], bundle["frames"]["test"]
    rows = []
    for name, det in bundle["detectors"].items():
        r = evaluate_model(val, test, f"score::{name}", alert_cost, daily_budget)[split]
        rows.append({"méthode": name, "famille": det.family, "PR-AUC": r["pr_auc"], "ROC-AUC": r["roc_auc"],
                     f"précision@{r['k']}": r["precision_at_k"], f"rappel@{r['k']}": r["recall_at_k"],
                     "alertes": r["alerts"], "précision": r["precision"], "rappel": r["recall"],
                     "F2": r["f2"], "MCC": r["mcc"], "coût": r["cost"], "économie": r["savings"]})
    return pd.DataFrame(rows).sort_values("coût").reset_index(drop=True)


def pr_curve(y, s) -> pd.DataFrame:
    p, r, _ = precision_recall_curve(y, s)
    return pd.DataFrame({"precision": p, "recall": r})
