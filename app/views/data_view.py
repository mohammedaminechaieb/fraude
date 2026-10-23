"""Page Module 1 — Données."""
import plotly.express as px
import streamlit as st

from app.common import cached_csv, cached_kaggle, cached_synthetic, get_df
from fraud import data as fdata
from fraud.config import AMOUNT_COL, TARGET


def _load_panel():
    st.subheader("Source")
    source = st.radio("Jeu de données", ["Synthétique (schéma Sparkov)", "Kaggle (Sparkov)", "Fichier CSV"],
                      horizontal=True)
    c1, c2 = st.columns(2)
    seed = c2.number_input("Graine", value=42, step=1)
    if source.startswith("Synthétique"):
        n_cards = c1.slider("Nombre de cartes", 50, 1500, 400, 50)
        if st.button("Générer", type="primary"):
            st.session_state["df"] = cached_synthetic(n_cards, int(seed))
            st.session_state["source"] = f"synthétique ({n_cards} cartes)"
    else:
        n_cards = c1.number_input("Échantillon de cartes (0 = toutes)", value=300, min_value=0, step=50) or None
        if source.startswith("Kaggle"):
            st.caption("Nécessite des identifiants Kaggle (`~/.kaggle/kaggle.json` ou `KAGGLE_USERNAME`/`KAGGLE_KEY`).")
            if st.button("Télécharger", type="primary"):
                try:
                    st.session_state["df"] = cached_kaggle(n_cards, int(seed))
                    st.session_state["source"] = "Kaggle Sparkov"
                except Exception as exc:  # réseau ou identifiants
                    st.error(f"Téléchargement impossible : {exc}")
        else:
            up = st.file_uploader("CSV au schéma Sparkov (fraudTrain.csv / fraudTest.csv)", type="csv")
            if up is not None and st.button("Importer", type="primary"):
                try:
                    st.session_state["df"] = cached_csv(up.getvalue(), n_cards, int(seed))
                    st.session_state["source"] = up.name
                except ValueError as exc:
                    st.error(str(exc))


def render():
    st.header("1 · Données")
    _load_panel()
    df = get_df()
    if df is None:
        return
    s = fdata.summary(df)
    st.divider()
    st.caption(f"Source : {st.session_state.get('source', '?')}")
    k = st.columns(5)
    k[0].metric("Transactions", f"{s['n_transactions']:,}".replace(",", " "))
    k[1].metric("Fraudes", f"{s['n_frauds']:,}".replace(",", " "))
    k[2].metric("Taux de fraude", f"{s['fraud_rate']:.3%}")
    k[3].metric("Cartes", s["n_cards"])
    k[4].metric("Montant fraudé", f"{s['fraud_amount_total']:,.0f}".replace(",", " "))
    if not s["has_time"]:
        st.warning("Aucune information temporelle : la séparation sera aléatoire stratifiée.")
        return
    st.caption(f"Période : {s['start']:%Y-%m-%d} → {s['end']:%Y-%m-%d} ({s['n_days']} jours)")

    cal = fdata.with_calendar(df)
    lab = cal[TARGET].map({0: "légitime", 1: "fraude"})

    c1, c2 = st.columns(2)
    fig = px.histogram(cal.assign(classe=lab), x=AMOUNT_COL, color="classe", log_x=False, nbins=80,
                       histnorm="probability density", barmode="overlay", range_x=[0, 1500],
                       title="Distribution des montants par classe")
    c1.plotly_chart(fig, width="stretch")
    fig = px.box(cal.assign(classe=lab), x="classe", y=AMOUNT_COL, log_y=True, title="Montants (échelle log)")
    c2.plotly_chart(fig, width="stretch")
    st.dataframe(fdata.amount_stats_by_class(df).style.format("{:,.2f}"), width="stretch")

    freq = st.radio("Granularité", ["W", "MS"], format_func={"W": "semaine", "MS": "mois"}.get, horizontal=True)
    ts = fdata.fraud_rate_over_time(df, freq)
    st.plotly_chart(px.line(ts, x="trans_date_trans_time", y="fraud_rate", markers=True,
                            title="Évolution du taux de fraude", labels={"trans_date_trans_time": "date"}),
                    width="stretch")

    c1, c2 = st.columns(2)
    by_h = fdata.fraud_rate_by(cal, "hour")
    c1.plotly_chart(px.bar(by_h, x="hour", y="fraud_rate", title="Taux de fraude par heure"),
                    width="stretch")
    by_d = fdata.fraud_rate_by(cal, "day_of_week")
    by_d["jour"] = by_d["day_of_week"].map(dict(enumerate(["lun", "mar", "mer", "jeu", "ven", "sam", "dim"])))
    c2.plotly_chart(px.bar(by_d, x="jour", y="fraud_rate", title="Taux de fraude par jour de semaine"),
                    width="stretch")
    c1, c2 = st.columns(2)
    by_c = fdata.fraud_rate_by(cal, "category")
    c1.plotly_chart(px.bar(by_c, x="fraud_rate", y="category", orientation="h",
                           title="Taux de fraude par catégorie (canal pos / net)"), width="stretch")
    if "age_band" in cal.columns:
        by_a = fdata.fraud_rate_by(cal, "age_band")
        c2.plotly_chart(px.bar(by_a.sort_values("age_band"), x="age_band", y="fraud_rate",
                               title="Taux de fraude par tranche d'âge"), width="stretch")
    with st.expander("Aperçu des données"):
        st.dataframe(df.head(200), width="stretch")
