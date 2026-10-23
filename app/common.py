"""État partagé de l'application Streamlit (données chargées, artefacts)."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fraud import data as fdata  # noqa: E402


@st.cache_data(show_spinner="Génération des données synthétiques…")
def cached_synthetic(n_cards: int, seed: int):
    return fdata.load_synthetic(n_cards, seed)


@st.cache_data(show_spinner="Téléchargement depuis Kaggle…")
def cached_kaggle(n_cards: int | None, seed: int):
    return fdata.load_kaggle(n_cards, seed)


@st.cache_data(show_spinner="Lecture du fichier…")
def cached_csv(content: bytes, n_cards: int | None, seed: int):
    import io
    import pandas as pd

    return fdata.normalize(fdata.sample_by_card(pd.read_csv(io.BytesIO(content)), n_cards, seed))


def get_df():
    return st.session_state.get("df")


def require_df():
    df = get_df()
    if df is None:
        st.info("Chargez d'abord un jeu de données depuis la page **1 · Données**.")
        st.stop()
    return df


def get_prepared():
    """Variables + séparation chronologique + préprocesseur, recalculés si les données changent."""
    from fraud.features import build_features, chronological_split, feature_columns, make_preprocessor

    df = require_df()
    prep = st.session_state.get("prep")
    if prep is None or prep["df_id"] != id(df):
        with st.spinner("Construction des variables…"):
            feats = build_features(df)
            tr, va, te, info = chronological_split(feats)
            num, cat = feature_columns(feats)
            pre = make_preprocessor(num, cat).fit(tr)
        prep = {"df_id": id(df), "feats": feats, "train": tr, "val": va, "test": te, "info": info,
                "num": num, "cat": cat, "pre": pre}
        st.session_state["prep"] = prep
        st.session_state.pop("imbalance", None)
    return prep
