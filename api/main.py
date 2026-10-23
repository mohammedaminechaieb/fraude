"""Module 6 — API FastAPI de scoring et d'explication.

    uvicorn api.main:app --reload            # documentation interactive : http://localhost:8000/docs

L'API est sans état : les variables comportementales (vélocité, écart au profil sur 30 jours)
sont calculées à partir de l'historique de la carte transmis dans la requête (`history`) et
des autres transactions du lot. Sans historique, la transaction est traitée comme la première
de la carte.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from fraud import __version__
from fraud.config import TIME_COL
from fraud.explain import grouped, reason_codes, shap_explanation
from fraud.features import build_features
from fraud.pipeline import BUNDLE_PATH, load_bundle


class Transaction(BaseModel):
    trans_date_trans_time: datetime
    cc_num: int
    category: str = Field(examples=["shopping_net"])
    amt: float = Field(gt=0, examples=[949.9])
    gender: Literal["F", "M"]
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    city_pop: int = Field(ge=0)
    dob: date
    merchant: str | None = None
    trans_num: str | None = None


class PredictRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)
    history: list[Transaction] = Field(default_factory=list, max_length=20000,
                                       description="Transactions passées des mêmes cartes (≥ 30 jours conseillés)")
    threshold: float | None = Field(None, description="Seuil personnalisé ; défaut = seuil coût-optimal")
    explain: Literal["alerts", "all", "none"] = "alerts"
    n_reasons: int = Field(5, ge=3, le=5)


class ReasonCode(BaseModel):
    code: str
    feature: str
    contribution: float
    text: str


class Prediction(BaseModel):
    trans_num: str | None
    score: float
    alert: bool
    threshold: float
    reasons: list[ReasonCode] | None = None


class PredictResponse(BaseModel):
    model: str
    predictions: list[Prediction]


class ExplainRequest(BaseModel):
    transaction: Transaction
    history: list[Transaction] = Field(default_factory=list, max_length=20000)


class Contribution(BaseModel):
    feature: str
    value: str
    shap: float


class ExplainResponse(BaseModel):
    score: float
    base_value: float
    log_odds: float
    contributions: list[Contribution]
    reasons: list[ReasonCode]


def _features(bundle: dict, transactions: list[Transaction], history: list[Transaction]) -> pd.DataFrame:
    rows = [t.model_dump() | {"_req": i} for i, t in enumerate(transactions)]
    rows += [t.model_dump() | {"_req": -1} for t in history]
    df = pd.DataFrame(rows)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    if df[TIME_COL].dt.tz is not None:
        df[TIME_COL] = df[TIME_COL].dt.tz_convert(None)
    df["dob"] = pd.to_datetime(df["dob"])
    df = df.sort_values([TIME_COL, "_req"], kind="stable").reset_index(drop=True)
    feats = build_features(df)
    return feats[feats["_req"] >= 0].sort_values("_req")


def _score(bundle: dict, feats: pd.DataFrame) -> np.ndarray:
    det = bundle["detectors"][bundle["decision"]["model"]]
    return det.score(bundle["preprocessor"].transform(feats))


def create_app(bundle: dict | None = None, bundle_path: Path | str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.bundle is None:
            path = Path(bundle_path or os.environ.get("FRAUD_BUNDLE", BUNDLE_PATH))
            if path.exists():
                app.state.bundle = load_bundle(path)
        yield

    app = FastAPI(title="API de détection de fraude explicable", version=__version__, lifespan=lifespan,
                  description="Scoring de transactions (schéma Sparkov), seuil au coût minimal et codes motifs SHAP.")
    app.state.bundle = bundle

    def get_bundle(request: Request) -> dict:
        b = request.app.state.bundle
        if b is None:
            raise HTTPException(503, "Aucun modèle chargé : lancez `python -m fraud.pipeline`.")
        return b

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_loaded": request.app.state.bundle is not None, "version": __version__}

    @app.get("/model")
    def model_info(request: Request):
        b = get_bundle(request)
        d = b["decision"]
        test = {k: v for k, v in d["test"].items() if isinstance(v, (int, float))}
        return {"model": d["model"], "trained_at": b["trained_at"], "imbalance_strategy": b["strategy"],
                "threshold": d["threshold"], "alert_cost": d["alert_cost"], "daily_budget": d["daily_budget"],
                "features": b["feature_names"], "split": b["split_info"]["method"], "test_metrics": test}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, request: Request):
        b = get_bundle(request)
        feats = _features(b, req.transactions, req.history)
        scores = _score(b, feats)
        tau = b["decision"]["threshold"] if req.threshold is None else req.threshold
        alerts = scores >= tau
        reasons = [None] * len(feats)
        wanted = alerts if req.explain == "alerts" else np.full(len(feats), req.explain == "all")
        if wanted.any():
            idx = np.flatnonzero(wanted)
            for i, rc in zip(idx, reason_codes(b, feats.iloc[idx], n=req.n_reasons)):
                reasons[i] = rc
        preds = [Prediction(trans_num=t.trans_num, score=float(s), alert=bool(a), threshold=float(tau), reasons=r)
                 for t, s, a, r in zip(req.transactions, scores, alerts, reasons)]
        return PredictResponse(model=b["decision"]["model"], predictions=preds)

    @app.post("/explain", response_model=ExplainResponse)
    def explain(req: ExplainRequest, request: Request):
        b = get_bundle(request)
        feats = _features(b, [req.transaction], req.history)
        exp = shap_explanation(b, feats)
        g = grouped(exp, feats)
        score = float(b["detectors"]["LightGBM"].score(b["preprocessor"].transform(feats))[0])
        contribs = sorted((Contribution(feature=f, value=str(v), shap=float(s))
                           for f, v, s in zip(g.feature_names, g.data[0], g.values[0])), key=lambda c: -abs(c.shap))
        return ExplainResponse(score=score, base_value=float(g.base_values[0]),
                               log_odds=float(g.base_values[0] + g.values[0].sum()), contributions=contribs,
                               reasons=reason_codes(b, feats)[0])

    return app


app = create_app()
