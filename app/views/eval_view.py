"""Page Module 4 — Évaluation et seuil (curseurs en direct)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.common import require_bundle
from fraud.config import AMOUNT_COL, TARGET
from fraud.evaluation import (budget_k, cost_curve, evaluation_table, metrics_at_threshold, n_days,
                              optimal_threshold, pr_curve, precision_recall_at_k)

SPLIT_LABEL = {"test": "test", "val": "validation"}


def _round_down(v: float) -> float:
    """Arrondi à 6 chiffres significatifs, par défaut (le seuil n'exclut jamais le score visé)."""
    r = float(f"{v:.6g}")
    if r > v and v != 0:
        r = float(f"{r - 10 ** (np.floor(np.log10(abs(r))) - 5):.6g}")
    return r


def _fmt(x):
    return f"{x:,.0f}".replace(",", " ")


def render():
    st.header("4 · Évaluation et seuil")
    bundle = require_bundle()
    decision = bundle["decision"]
    frames = bundle["frames"]
    names = list(bundle["detectors"])

    c = st.columns(4)
    model = c[0].selectbox("Modèle", names, index=names.index(decision["model"]))
    alert_cost = c[1].number_input("Coût d'analyse d'une alerte (Ca)", 0.0, 1000.0, float(decision["alert_cost"]), 1.0)
    daily_budget = c[2].slider("Budget d'alertes / jour", 1, max(50, 5 * decision["daily_budget"]),
                               int(decision["daily_budget"]))
    split = c[3].radio("Jeu affiché", ["test", "val"], format_func=SPLIT_LABEL.get, horizontal=True)

    col = f"score::{model}"
    val, frame = frames["val"], frames[split]
    tau_opt = optimal_threshold(val[TARGET], val[col], val[AMOUNT_COL], alert_cost)

    # curseur de seuil : quantiles des scores de validation + le seuil optimal
    tau_default = _round_down(tau_opt)
    grid = sorted({_round_down(v) for v in np.quantile(val[col], np.linspace(0.5, 1, 801))} | {tau_default})
    tau = st.select_slider("Seuil d'alerte τ (alerte si score ≥ τ) — initialisé au coût minimal en validation",
                           options=grid, value=tau_default, format_func=lambda v: f"{v:.6g}", key=f"tau::{model}")

    y, s, amt = frame[TARGET].to_numpy(), frame[col].to_numpy(), frame[AMOUNT_COL].to_numpy()
    m = metrics_at_threshold(y, s, tau, amt, alert_cost)
    k = budget_k(frame, daily_budget)
    p_k, r_k = precision_recall_at_k(y, s, k)
    days = n_days(frame)

    st.markdown(f"**{SPLIT_LABEL[split].capitalize()}** — {days} jours, {len(frame):,} transactions, "
                f"{int(y.sum())} fraudes · seuil optimal (validation) τ* = `{tau_opt:.5g}`".replace(",", " "))
    k1 = st.columns(6)
    k1[0].metric("Alertes", _fmt(m["alerts"]), f"{m['alerts'] / days:.1f}/jour", delta_color="off")
    k1[1].metric("Précision", f"{m['precision']:.1%}")
    k1[2].metric("Rappel", f"{m['recall']:.1%}")
    k1[3].metric("F2", f"{m['f2']:.3f}")
    k1[4].metric("MCC", f"{m['mcc']:.3f}")
    k1[5].metric("Coût", _fmt(m["cost"]), f"-{_fmt(m['savings'])} vs sans modèle", delta_color="inverse")
    k2 = st.columns(4)
    k2[0].metric(f"Budget k = {daily_budget}/j × {days} j", _fmt(k))
    k2[1].metric("Précision@k", f"{p_k:.1%}")
    k2[2].metric("Rappel@k", f"{r_k:.1%}")
    k2[3].metric("Coût sans modèle", _fmt(m["cost_no_model"]))
    if m["alerts"] > k:
        st.warning(f"Le seuil génère {m['alerts']} alertes, au-delà du budget de {k} : l'équipe ne pourra pas tout traiter.")

    c1, c2 = st.columns(2)
    curve = cost_curve(y, s, amt, alert_cost)
    curve = curve[curve.alerts <= max(3 * k, 3 * m["alerts"], 50)]
    fig = go.Figure()
    fig.add_scatter(x=curve.alerts, y=curve.cost, name="coût total", mode="lines")
    fig.add_scatter(x=curve.alerts, y=curve.missed_amount, name="fraudes manquées", mode="lines", line_dash="dot")
    fig.add_scatter(x=curve.alerts, y=curve.review_cost, name="coût d'analyse", mode="lines", line_dash="dot")
    fig.add_vline(x=m["alerts"], line_color="red", annotation_text="seuil courant")
    fig.add_vline(x=k, line_color="gray", line_dash="dash", annotation_text="budget k")
    fig.update_layout(title="Coût métier selon le nombre d'alertes", xaxis_title="alertes", yaxis_title="coût")
    c1.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for name in names:
        pr = pr_curve(y, frame[f"score::{name}"])
        fig.add_scatter(x=pr.recall, y=pr.precision, name=name, mode="lines",
                        line_width=3 if name == model else 1.5)
    fig.add_scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", marker_size=12, marker_color="red",
                    name="seuil courant")
    fig.add_hline(y=y.mean(), line_dash="dash", line_color="gray", annotation_text="aléatoire")
    fig.update_layout(title="Courbes précision-rappel superposées", xaxis_title="rappel", yaxis_title="précision")
    c2.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns([1, 2])
    cm = pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], index=["réel : légitime", "réel : fraude"],
                      columns=["prédit : légitime", "prédit : alerte"])
    c1.markdown("**Matrice de confusion**")
    c1.dataframe(cm, width="stretch")
    c2.markdown(f"**Toutes les méthodes** — seuil de chacune choisi au coût minimal en validation, Ca = {alert_cost:g}")
    table = evaluation_table(bundle, alert_cost, daily_budget, split)
    c2.dataframe(table.style.format({c: "{:.3f}" for c in table.columns if c not in ("méthode", "famille", "alertes",
                                                                                      "coût", "économie")}
                                    | {"coût": "{:,.0f}", "économie": "{:,.0f}"}), hide_index=True, width="stretch")

    st.subheader("File d'alertes")
    alerts = frame[s >= tau].assign(score=s[s >= tau]).sort_values("score", ascending=False)
    show = [c for c in ["trans_date_trans_time", "cc_num", "merchant", "category", AMOUNT_COL, "score", TARGET]
            if c in alerts.columns]
    st.dataframe(alerts[show].head(500), hide_index=True, width="stretch")
