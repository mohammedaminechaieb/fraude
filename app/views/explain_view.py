"""Page Module 5 — Explainable AI."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import shap  # noqa: E402
import streamlit as st  # noqa: E402

from app.common import require_bundle  # noqa: E402
from fraud.config import AMOUNT_COL, TARGET  # noqa: E402
from fraud.explain import global_sample, grouped, permutation_importance, reason_codes, shap_explanation  # noqa: E402

SCORE = "score::LightGBM"


@st.cache_data(show_spinner="Importance par permutation…")
def _perm(_bundle, key):
    return permutation_importance(_bundle, _bundle["frames"]["val"])


@st.cache_data(show_spinner="Valeurs SHAP…")
def _beeswarm_png(_bundle, key):
    sample = global_sample(_bundle["frames"]["test"])
    exp = shap_explanation(_bundle, sample)
    fig = plt.figure()
    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.title("SHAP beeswarm — LightGBM (contributions en log-cote)")
    return _to_png(fig)


def _to_png(fig):
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def render():
    st.header("5 · Explainable AI")
    bundle = require_bundle()
    key = bundle["trained_at"]
    st.caption("Modèle expliqué : LightGBM. Les contributions SHAP s'additionnent à la valeur de base pour donner la "
               "log-cote du score (correction de rééchantillonnage incluse dans la base).")

    tab_g, tab_l = st.tabs(["Explication globale", "Explication d'une alerte"])
    with tab_g:
        c1, c2 = st.columns([3, 2])
        c1.image(_beeswarm_png(bundle, key), width="stretch")
        perm = _perm(bundle, key)
        c2.plotly_chart(px.bar(perm.iloc[::-1], x="importance", y="variable", error_x="std", orientation="h",
                               title="Importance par permutation (baisse de PR-AUC, validation)"), width="stretch")
        st.markdown("Chaque point du beeswarm est une transaction ; la couleur indique la valeur de la variable "
                    "(rouge = élevée) et la position son effet sur le score (à droite = vers la fraude).")

    with tab_l:
        test = bundle["frames"]["test"]
        tau = bundle["decision"]["threshold"]
        alerts = test[test[SCORE] >= tau].sort_values(SCORE, ascending=False)
        if alerts.empty:
            st.info("Aucune alerte au seuil courant.")
            return
        st.markdown(f"**{len(alerts)} alertes** sur le test au seuil coût-optimal τ = `{tau:.5g}` "
                    f"(dont {int(alerts[TARGET].sum())} fraudes avérées).")
        labels = [f"#{i + 1} · {r.trans_date_trans_time:%Y-%m-%d %H:%M} · {r.category} · {r[AMOUNT_COL]:.2f} · "
                  f"score {r[SCORE]:.3f}{' · FRAUDE' if r[TARGET] else ''}"
                  for i, (_, r) in enumerate(alerts.head(300).iterrows())]
        pick = st.selectbox("Alerte", range(len(labels)), format_func=labels.__getitem__)
        row = alerts.iloc[[pick]]

        c1, c2 = st.columns([2, 3])
        with c1:
            r = row.iloc[0]
            st.metric("Score de fraude", f"{r[SCORE]:.1%}")
            st.markdown("**Codes motifs**")
            for code in reason_codes(bundle, row)[0]:
                st.markdown(f"- `{code['code']}` {code['text']}  \n  <small>contribution +{code['contribution']:.2f}"
                            f" log-cote</small>", unsafe_allow_html=True)
            details = {c: r[c] for c in ["trans_date_trans_time", "cc_num", "merchant", "category", AMOUNT_COL,
                                         "distance_km", "n_tx_24h", "amt_ratio_30d"] if c in r.index}
            details = {c: round(v, 2) if isinstance(v, float) else v for c, v in details.items()}
            st.dataframe(pd.Series(details, name="valeur").astype(str), width="stretch")
            st.caption(f"Étiquette réelle : {'fraude' if r[TARGET] else 'légitime'}")
        with c2:
            exp = grouped(shap_explanation(bundle, row), row)
            fig = plt.figure()
            shap.plots.waterfall(exp[0], max_display=10, show=False)
            st.image(_to_png(plt.gcf()), width="stretch")
