import pytest

from fraud.data import load_synthetic


@pytest.fixture(scope="session")
def small_df():
    return load_synthetic(n_cards=120, seed=7)
