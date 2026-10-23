"""Page Module 6 — API FastAPI.

Si FRAUD_API_URL est défini (variable d'environnement ou secret Streamlit), la page interroge l'API
déployée (par exemple sur Render) ; sinon, ou si elle est injoignable, une copie de l'API tourne dans
le processus de l'application.
"""
import json
import os

import pandas as pd
import streamlit as st
from fastapi.testclient import TestClient

from app.common import get_df, require_bundle
from fraud.config import TIME_COL

FIELDS = ["trans_date_trans_time", "cc_num", "category", "amt", "gender", "lat", "long", "merch_lat",
          "merch_long", "city_pop", "dob", "merchant", "trans_num"]


def _payload(row: pd.Series) -> dict:
    d = {k: row[k] for k in FIELDS if k in row.index}
    d["trans_date_trans_time"] = pd.Timestamp(row[TIME_COL]).isoformat()
    d["dob"] = pd.Timestamp(row["dob"]).date().isoformat()
    d["cc_num"], d["city_pop"], d["amt"] = int(d["cc_num"]), int(d["city_pop"]), float(d["amt"])
    for k in ("lat", "long", "merch_lat", "merch_long"):
        d[k] = float(d[k])
    return d


def _api_url() -> str:
    url = os.environ.get("FRAUD_API_URL", "")
    if not url:
        try:
            url = st.secrets.get("FRAUD_API_URL", "")
        except Exception:  # aucun fichier de secrets
            url = ""
    return url.strip().rstrip("/")


def _client(bundle):
    """Client HTTP vers l'API déployée, ou client en processus en repli."""
    from api.main import create_app

    url = _api_url()
    if url:
        import httpx

        client = httpx.Client(base_url=url, timeout=120)
        try:
            with st.spinner("Connexion à l'API déployée (jusqu'à 1 min si elle sort de veille)…"):
                client.get("/health").raise_for_status()
            st.success(f"API déployée : [{url}/docs]({url}/docs)")
            return client
        except httpx.HTTPError as exc:
            st.warning(f"API déployée injoignable ({exc}) : démonstration avec l'API intégrée.")
    else:
        st.caption("Démonstration avec l'API intégrée. Définir `FRAUD_API_URL` pour interroger l'API déployée.")
    return TestClient(create_app(bundle))


def render():
    st.header("6 · API FastAPI")
    bundle = require_bundle()
    st.markdown("""
Lancement : `uvicorn api.main:app --reload` puis documentation interactive sur `http://localhost:8000/docs`.

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | état du service et du modèle |
| GET | `/model` | modèle, seuil coût-optimal, coût d'analyse, budget, métriques de test |
| POST | `/predict` | score, décision d'alerte et codes motifs pour 1 à 1 000 transactions |
| POST | `/explain` | contributions SHAP complètes d'une transaction |

L'API est **sans état** : l'historique de la carte (`history`) sert à calculer les variables de vélocité et
d'écart au profil, exactement comme hors ligne.
""")
    client = _client(bundle)
    st.subheader("GET /model")
    st.json(client.get("/model").json(), expanded=False)

    df = get_df()
    if df is None:
        st.info("Chargez les données (page 1) pour tester /predict sur une transaction réelle avec son historique.")
        return
    test = bundle["frames"]["test"]
    known = test[test["trans_num"].isin(df["trans_num"])] if "trans_num" in test.columns else test.iloc[0:0]
    if known.empty:
        st.info("Les données chargées ne correspondent pas aux modèles entraînés.")
        return
    st.subheader("POST /predict")
    top = known.nlargest(20, "score::LightGBM")
    pick = st.selectbox("Transaction du test", top["trans_num"],
                        format_func=lambda t: f"{t[:10]}… · score hors ligne "
                                              f"{top.set_index('trans_num').loc[t, 'score::LightGBM']:.4f}")
    row = df[df["trans_num"] == pick].iloc[0]
    hist = df[(df["cc_num"] == row["cc_num"]) & (df[TIME_COL] < row[TIME_COL])
              & (df[TIME_COL] >= row[TIME_COL] - pd.Timedelta(days=31))]
    body = {"transactions": [_payload(row)], "history": [_payload(r) for _, r in hist.iterrows()]}
    c1, c2 = st.columns(2)
    c1.caption(f"Requête ({len(hist)} transactions d'historique)")
    c1.code(json.dumps(body | {"history": body["history"][:2] + (["…"] if len(hist) > 2 else [])}, indent=2,
                       ensure_ascii=False), language="json")
    resp = client.post("/predict", json=body)
    c2.caption(f"Réponse — HTTP {resp.status_code}")
    c2.json(resp.json())
