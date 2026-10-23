"""Page Module 2 — Préparation et variables."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.common import get_prepared
from fraud.config import TARGET, TIME_COL
from fraud.features import BEHAVIOR_FEATURES, FEATURE_DESCRIPTIONS
from fraud.imbalance import compare_strategies


def render():
    st.header("2 · Préparation et variables")
    prep = get_prepared()
    info = prep["info"]

    st.subheader("Variables construites")
    rows = [{"variable": c, "type": "catégorielle" if c in prep["cat"] else "numérique",
             "description": FEATURE_DESCRIPTIONS.get(c, "catégorie du marchand (canal pos / net)")}
            for c in prep["num"] + prep["cat"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Les variables comportementales n'utilisent que les transactions **antérieures** de la même carte "
               "(fenêtres fermées à gauche) : aucune information du futur. Le préprocesseur (standardisation, "
               "one-hot) est ajusté sur l'entraînement seul.")

    feats = prep["feats"]
    var = st.selectbox("Distribution d'une variable comportementale", BEHAVIOR_FEATURES + ["distance_km"])
    fig = px.histogram(feats.assign(classe=feats[TARGET].map({0: "légitime", 1: "fraude"})), x=var,
                       color="classe", histnorm="probability density", barmode="overlay", nbins=60)
    st.plotly_chart(fig, width="stretch")

    st.subheader(f"Séparation {info['method']} 60 / 20 / 20")
    if info["warning"]:
        st.warning(info["warning"])
    split = pd.DataFrame({k: info[k] for k in ("train", "val", "test")}).T
    split.index = ["entraînement", "validation", "test"]
    st.dataframe(split, width="stretch")
    if TIME_COL in feats.columns:
        fig = go.Figure()
        for name, part in zip(split.index, (prep["train"], prep["val"], prep["test"])):
            w = part.groupby(pd.Grouper(key=TIME_COL, freq="W"))[TARGET].mean()
            fig.add_scatter(x=w.index, y=w.values, name=name, mode="lines")
        fig.update_layout(title="Taux de fraude hebdomadaire par partition", yaxis_tickformat=".2%")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Stratégie contre le déséquilibre")
    st.markdown("Chaque stratégie est appliquée **au seul entraînement** puis évaluée sur la validation, qui garde la "
                "proportion réelle de fraudes. Critère : PR-AUC. Après sous-échantillonnage, les probabilités sont "
                "corrigées par Bayes : $p = \\beta p_s / (\\beta p_s - p_s + 1)$.")
    if st.button("Comparer les stratégies (LightGBM)", type="primary"):
        pre = prep["pre"]
        with st.spinner("Entraînement de 6 modèles…"):
            st.session_state["imbalance"] = compare_strategies(
                pre.transform(prep["train"]), prep["train"][TARGET], pre.transform(prep["val"]), prep["val"][TARGET])
    res = st.session_state.get("imbalance")
    if res is not None:
        st.dataframe(res.style.format({"fraud_share_train": "{:.2%}", "beta": "{:.3f}", "pr_auc_val": "{:.4f}",
                                       "roc_auc_val": "{:.4f}", "fit_seconds": "{:.1f}"}),
                     hide_index=True, width="stretch")
        st.success(f"Stratégie retenue : **{res.iloc[0]['label']}** (meilleure PR-AUC en validation).")
