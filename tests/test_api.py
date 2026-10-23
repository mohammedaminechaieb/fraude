import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from fraud.config import TIME_COL

FIELDS = ["trans_date_trans_time", "cc_num", "category", "amt", "gender", "lat", "long", "merch_lat",
          "merch_long", "city_pop", "dob", "merchant", "trans_num"]


def _payload(row: pd.Series) -> dict:
    d = {k: row[k] for k in FIELDS}
    d["trans_date_trans_time"] = row[TIME_COL].isoformat()
    d["dob"] = row["dob"].date().isoformat()
    d["cc_num"] = int(d["cc_num"])
    d["city_pop"] = int(d["city_pop"])
    return d


def _with_history(small_df, trans_num):
    row = small_df[small_df.trans_num == trans_num].iloc[0]
    card = small_df[(small_df.cc_num == row.cc_num) & (small_df[TIME_COL] < row[TIME_COL])
                    & (small_df[TIME_COL] >= row[TIME_COL] - pd.Timedelta(days=31))]
    return _payload(row), [_payload(r) for _, r in card.iterrows()]


@pytest.fixture(scope="module")
def client(bundle):
    with TestClient(create_app(bundle)) as c:
        yield c


def test_health_and_model(client, bundle):
    assert client.get("/health").json()["model_loaded"] is True
    info = client.get("/model").json()
    assert info["model"] == "LightGBM" and np.isclose(info["threshold"], bundle["decision"]["threshold"])


def test_predict_matches_offline_scores(client, bundle, small_df):
    """Avec l'historique de la carte, l'API reproduit exactement le score calculé hors ligne."""
    test = bundle["frames"]["test"]
    sample = pd.concat([test.nlargest(3, "score::LightGBM"), test.sample(3, random_state=1)])
    for _, r in sample.iterrows():
        tx, hist = _with_history(small_df, r.trans_num)
        res = client.post("/predict", json={"transactions": [tx], "history": hist}).json()
        p = res["predictions"][0]
        assert np.isclose(p["score"], r["score::LightGBM"], rtol=1e-6)
        assert p["alert"] == (r["score::LightGBM"] >= bundle["decision"]["threshold"])
        assert (p["reasons"] is not None) == p["alert"]
        if p["alert"]:
            assert 3 <= len(p["reasons"]) <= 5


def test_predict_batch_and_options(client, small_df):
    rows = [_payload(r) for _, r in small_df.head(20).iterrows()]
    res = client.post("/predict", json={"transactions": rows, "threshold": 0.0, "explain": "all",
                                        "n_reasons": 3}).json()
    assert len(res["predictions"]) == 20
    assert all(p["alert"] and len(p["reasons"]) == 3 for p in res["predictions"])
    assert [p["trans_num"] for p in res["predictions"]] == [r["trans_num"] for r in rows]


def test_explain(client, bundle, small_df):
    r = bundle["frames"]["test"].nlargest(1, "score::LightGBM").iloc[0]
    tx, hist = _with_history(small_df, r.trans_num)
    res = client.post("/explain", json={"transaction": tx, "history": hist}).json()
    assert np.isclose(res["score"], r["score::LightGBM"], rtol=1e-6)
    assert np.isclose(res["log_odds"], np.log(res["score"] / (1 - res["score"])), atol=1e-6)
    assert abs(res["contributions"][0]["shap"]) >= abs(res["contributions"][-1]["shap"])


def test_validation_errors(client, small_df):
    tx = _payload(small_df.iloc[0]) | {"amt": -5}
    assert client.post("/predict", json={"transactions": [tx]}).status_code == 422
    assert client.post("/predict", json={"transactions": []}).status_code == 422


def test_no_model_returns_503(tmp_path):
    with TestClient(create_app(bundle_path=tmp_path / "absent.joblib")) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.get("/model").status_code == 503
