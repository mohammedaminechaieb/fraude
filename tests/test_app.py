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
