"""État partagé de l'application Streamlit (données chargées, artefacts)."""
from __future__ import annotations

import os
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


def get_bundle():
    """Artefacts d'entraînement : session courante, sinon fichier artifacts/bundle.joblib."""
    from fraud.pipeline import BUNDLE_PATH, load_bundle

    if "bundle" not in st.session_state and BUNDLE_PATH.exists():
        st.session_state["bundle"] = load_bundle(BUNDLE_PATH)
        st.session_state["bundle_origin"] = f"fichier {BUNDLE_PATH.name}"
    return st.session_state.get("bundle")


def require_bundle():
    bundle = get_bundle()
    if bundle is None:
        st.info("Entraînez d'abord les modèles depuis la page **3 · Modélisation**.")
        st.stop()
    return bundle


@st.cache_resource(show_spinner=False)
def _autotrained_bundle(n_cards: int, seed: int):
    """Entraînement unique, partagé par toutes les sessions du serveur."""
    from fraud.pipeline import BUNDLE_PATH, save_bundle, train

    bundle = train(fdata.load_synthetic(n_cards, seed))
    save_bundle(bundle, BUNDLE_PATH)
    return bundle


def ensure_ready():
    """Auto-entraînement : au premier lancement, données synthétiques + modèles sans clic.

    Désactivable avec FRAUD_AUTOTRAIN=0. Si un modèle a déjà été entraîné (fichier d'artefacts, par
    exemple au build Docker / Render), il est simplement chargé ; les données par défaut sont
    régénérées à l'identique (même taille FRAUD_N_CARDS, même graine).
    """
    if os.environ.get("FRAUD_AUTOTRAIN", "1") == "0":
        return
    from fraud.config import DEFAULT_N_CARDS, RANDOM_STATE

    if get_df() is None:
        st.session_state["df"] = cached_synthetic(DEFAULT_N_CARDS, RANDOM_STATE)
        st.session_state["source"] = f"synthétique ({DEFAULT_N_CARDS} cartes, chargé automatiquement)"
    if get_bundle() is None:
        with st.spinner("Premier lancement : entraînement automatique des modèles (environ 1 minute)…"):
            st.session_state["bundle"] = _autotrained_bundle(DEFAULT_N_CARDS, RANDOM_STATE)
        st.session_state["bundle_origin"] = "auto-entraînement au premier lancement"
