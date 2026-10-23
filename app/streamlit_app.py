"""Application Streamlit — détection de fraude explicable.

Lancement : streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.common import ensure_ready  # noqa: E402
from app.views import api_view, data_view, eval_view, explain_view, model_view, prep_view  # noqa: E402

st.set_page_config(page_title="Détection de fraude explicable", page_icon="🛡️", layout="wide")

PAGES = {
    "1 · Données": data_view.render,
    "2 · Préparation": prep_view.render,
    "3 · Modélisation": model_view.render,
    "4 · Évaluation et seuil": eval_view.render,
    "5 · Explications (XAI)": explain_view.render,
    "6 · API": api_view.render,
}

st.sidebar.title("🛡️ Fraude explicable")
page = st.sidebar.radio("Module", list(PAGES))
ensure_ready()
PAGES[page]()
