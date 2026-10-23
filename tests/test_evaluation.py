import numpy as np
import pandas as pd

from fraud.config import AMOUNT_COL, TARGET
from fraud.evaluation import (cost_curve, evaluation_table, metrics_at_threshold, optimal_threshold,
                              precision_recall_at_k, suggest_daily_budget)


def _brute_force_cost(y, s, amt, tau, ca):
    alert = s >= tau
    return amt[(y == 1) & ~alert].sum() + ca * alert.sum()


def test_cost_curve_matches_brute_force():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.05).astype(int)
    s = np.round(rng.random(400) + y * 0.5, 2)  # arrondi : beaucoup d'ex aequo
    amt = rng.lognormal(4, 1, 400)
    curve = cost_curve(y, s, amt, 7.0)
    for _, row in curve.sample(30, random_state=0).iterrows():
        assert np.isclose(row.cost, _brute_force_cost(y, s, amt, row.threshold, 7.0))
    tau = optimal_threshold(y, s, amt, 7.0)
    best = min(_brute_force_cost(y, s, amt, t, 7.0) for t in np.r_[np.unique(s), np.inf])
    assert np.isclose(_brute_force_cost(y, s, amt, tau, 7.0), best)


def test_metrics_at_threshold():
    y = np.array([1, 1, 0, 0, 0, 1])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.05, 0.7])
    m = metrics_at_threshold(y, s, 0.5, amounts=[100, 50, 1, 1, 1, 30], alert_cost=2)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert np.isclose(m["precision"], 2 / 3) and np.isclose(m["recall"], 2 / 3)
    assert m["missed_amount"] == 50 and m["cost"] == 50 + 2 * 3 and m["cost_no_model"] == 180


def test_precision_recall_at_k():
    y = np.array([0, 1, 0, 1, 0])
    s = np.array([0.1, 0.9, 0.8, 0.3, 0.2])
    assert precision_recall_at_k(y, s, 2) == (0.5, 0.5)


def test_suggested_budget():
    info = {"train": {"start": pd.Timestamp("2020-01-01"), "end": pd.Timestamp("2020-01-11"), "frauds": 20}}
    assert suggest_daily_budget(info) == 6


def test_decision_and_table(bundle):
    d = bundle["decision"]
    assert d["model"] == "LightGBM"
    t = d["test"]
    assert t["cost"] < t["cost_no_model"]
    val = bundle["frames"]["val"]
    assert np.isclose(d["threshold"], optimal_threshold(val[TARGET], val["score::LightGBM"], val[AMOUNT_COL]))
    table = evaluation_table(bundle)
    assert len(table) == 5 and {"F2", "MCC", "PR-AUC"} <= set(table.columns)
