import io

import pandas as pd

from fraud.config import SPARKOV_COLUMNS, TARGET, TIME_COL
from fraud.data import (amount_stats_by_class, fraud_rate_by, fraud_rate_over_time, load_csv,
                        sample_by_card, summary, with_calendar)


def test_synthetic_schema_and_order(small_df):
    assert list(small_df.columns) == SPARKOV_COLUMNS
    assert small_df[TIME_COL].is_monotonic_increasing
    rate = small_df[TARGET].mean()
    assert 0.001 < rate < 0.02


def test_csv_roundtrip(small_df):
    buf = io.StringIO()
    small_df.head(500).assign(**{"Unnamed: 0": 0}).to_csv(buf, index=False)
    buf.seek(0)
    df = load_csv(buf)
    assert "Unnamed: 0" not in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df[TIME_COL])


def test_sample_by_card_keeps_whole_history(small_df):
    sub = sample_by_card(small_df, 10)
    assert sub["cc_num"].nunique() == 10
    for card, g in sub.groupby("cc_num"):
        assert len(g) == (small_df["cc_num"] == card).sum()


def test_eda_helpers(small_df):
    s = summary(small_df)
    assert s["n_frauds"] == small_df[TARGET].sum() and s["has_time"]
    assert set(amount_stats_by_class(small_df).index) == {"légitime", "fraude"}
    assert fraud_rate_over_time(small_df)["n"].sum() == len(small_df)
    by_hour = fraud_rate_by(with_calendar(small_df), "hour")
    assert len(by_hour) == 24
    # les fraudes du simulateur sont surtout nocturnes
    night = by_hour[by_hour.hour.isin([22, 23, 0, 1, 2, 3])].fraud_rate.mean()
    day = by_hour[by_hour.hour.isin(range(9, 18))].fraud_rate.mean()
    assert night > day
