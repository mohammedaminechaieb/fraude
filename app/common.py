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
