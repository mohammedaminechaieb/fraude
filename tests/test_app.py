"""Tests de fumée de l'application Streamlit (AppTest, sans navigateur)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def _app_with_data():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.slider[0].set_value(100)
    at.button[0].click().run()
    assert not at.exception
    return at


def test_data_page_loads_synthetic():
    at = _app_with_data()
    assert at.metric[0].label == "Transactions"


def test_prep_page():
    at = _app_with_data()
    at.sidebar.radio[0].set_value("2 · Préparation").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("Stratégie retenue" in s.value for s in at.success)


def test_model_page_trains():
    at = _app_with_data()
    at.sidebar.radio[0].set_value("3 · Modélisation").run()
    at.selectbox[0].set_value("none")
    at.button[0].click().run()
    assert not at.exception
    assert any("PR-AUC" in str(df.value.columns.tolist()) for df in at.dataframe)


def _app_with_bundle():
    at = _app_with_data()
    at.sidebar.radio[0].set_value("3 · Modélisation").run()
    at.selectbox[0].set_value("none")
    at.button[0].click().run()
    assert not at.exception
    return at


def test_eval_page_reacts_to_sliders():
    at = _app_with_bundle()
    at.sidebar.radio[0].set_value("4 · Évaluation et seuil").run()
    assert not at.exception
    alerts_before = at.metric[0].value
    at.select_slider[0].set_value(float(at.select_slider[0].options[-1])).run()  # seuil maximal → moins d'alertes
    assert not at.exception
    assert int(at.metric[0].value.replace(" ", "")) <= int(alerts_before.replace(" ", ""))


def test_explain_page():
    at = _app_with_bundle()
    at.sidebar.radio[0].set_value("5 · Explications (XAI)").run()
    assert not at.exception
    assert any("Codes motifs" in m.value for m in at.markdown)
