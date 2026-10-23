"""Page Module 3 — Modélisation."""
import pandas as pd
import plotly.express as px
import streamlit as st

from app.common import get_bundle, get_df
from fraud.imbalance import STRATEGIES, STRATEGY_LABELS
from fraud.pipeline import comparison_table, save_bundle, train


def _train_panel():
    df = get_df()
    if df is None:
        st.info("Chargez des données (page 1) pour entraîner, ou lancez `python -m fraud.pipeline`.")
        return
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Stratégie de déséquilibre (LightGBM)", ["auto"] + STRATEGIES,
                          format_func=lambda s: "auto — meilleure PR-AUC en validation" if s == "auto"
                          else STRATEGY_LABELS[s])
    if c2.button("Entraîner tous les modèles", type="primary"):
        with st.status("Entraînement…", expanded=True) as status:
            bundle = train(df, choice, progress=st.write)
            path = save_bundle(bundle)
            status.update(label=f"Terminé en {bundle['duration_seconds']:.0f} s — {path.name} enregistré",
                          state="complete", expanded=False)
        st.session_state["bundle"] = bundle
        st.session_state["bundle_origin"] = f"session ({st.session_state.get('source', '?')})"


def render():
    st.header("3 · Modélisation")
    _train_panel()
    bundle = get_bundle()
    if bundle is None:
        return
    st.caption(f"Modèles entraînés le {bundle['trained_at']} — origine : {st.session_state.get('bundle_origin', '?')}"
               f" — stratégie LightGBM : **{STRATEGY_LABELS.get(bundle['strategy'], bundle['strategy'])}**")

    st.subheader("Méthodes")
    st.dataframe(pd.DataFrame([{"méthode": n, "famille": d.family, "principe": d.description}
                               for n, d in bundle["detectors"].items()]), hide_index=True, width="stretch")

    st.subheader("Tableau comparatif")
    table = comparison_table(bundle)
    st.dataframe(table.style.format({c: "{:.4f}" for c in table.columns if "AUC" in c} | {"entraînement (s)": "{:.1f}"})
                 .background_gradient(subset=["PR-AUC val", "PR-AUC test"], cmap="Greens"),
                 hide_index=True, width="stretch")
    long = table.melt(id_vars=["méthode"], value_vars=["PR-AUC val", "PR-AUC test"], var_name="jeu",
                      value_name="PR-AUC")
    st.plotly_chart(px.bar(long, x="méthode", y="PR-AUC", color="jeu", barmode="group",
                           title="PR-AUC par méthode (référence aléatoire = taux de fraude)"), width="stretch")
    st.markdown("Les méthodes **non supervisées** repèrent l'*inhabituel*, qui n'est pas toujours *frauduleux* : "
                "elles produisent davantage de fausses alertes que le modèle supervisé, mais n'ont besoin d'aucune "
                "étiquette et peuvent signaler des schémas inédits.")
    if bundle["imbalance"] is not None:
        with st.expander("Comparaison des stratégies de déséquilibre (validation)"):
            st.dataframe(bundle["imbalance"], hide_index=True, width="stretch")
