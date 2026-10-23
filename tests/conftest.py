import os
import tempfile

# les tests n'écrivent jamais dans artifacts/ du projet
os.environ.setdefault("FRAUD_ARTIFACTS_DIR", tempfile.mkdtemp(prefix="fraud-test-"))

import pytest

from fraud.data import load_synthetic


@pytest.fixture(scope="session")
def small_df():
    return load_synthetic(n_cards=120, seed=7)


@pytest.fixture(scope="session")
def bundle(small_df):
    from fraud.pipeline import train

    return train(small_df, strategy="under_1:10")
