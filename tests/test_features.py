import numpy as np
import pandas as pd

from fraud.config import TARGET, TIME_COL
from fraud.features import build_features, chronological_split, feature_columns, make_preprocessor
from fraud.imbalance import correct_proba, resample


def _naive_counts(df, card, ts, window):
    g = df[(df.cc_num == card) & (df[TIME_COL] < ts) & (df[TIME_COL] >= ts - window)]
    return len(g), g


def test_behavior_features_match_naive_computation(small_df):
    f = build_features(small_df)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(f), 40, replace=False):
        row = f.iloc[i]
        n24, _ = _naive_counts(small_df, row.cc_num, row[TIME_COL], pd.Timedelta(hours=24))
        n30, g30 = _naive_counts(small_df, row.cc_num, row[TIME_COL], pd.Timedelta(days=30))
        # égalités de timestamps : la transaction antérieure dans l'ordre compte aussi
        same = small_df[(small_df.cc_num == row.cc_num) & (small_df[TIME_COL] == row[TIME_COL])]
        if len(same) > 1:
            continue
        assert row.n_tx_24h == n24
        expected = row.amt / g30.amt.mean() if n30 else 1.0
        assert np.isclose(row.amt_ratio_30d, expected)


def test_no_future_leakage(small_df):
    """Modifier le futur ne doit pas changer les variables du passé."""
    cut = small_df[TIME_COL].quantile(0.5)
    base = build_features(small_df)
    altered = small_df.copy()
    future = altered[TIME_COL] > cut
    altered.loc[future, "amt"] *= 50
    f2 = build_features(altered)
    past = base[TIME_COL] <= cut
    cols = ["n_tx_1h", "n_tx_24h", "amt_ratio_30d", "log_secs_since_prev"]
    pd.testing.assert_frame_equal(base.loc[past, cols], f2.loc[past, cols])


def test_chronological_split(small_df):
    tr, va, te, info = chronological_split(build_features(small_df))
    assert info["method"] == "chronologique"
    assert tr[TIME_COL].max() <= va[TIME_COL].min() <= va[TIME_COL].max() <= te[TIME_COL].min()
    assert abs(len(tr) / len(small_df) - 0.6) < 0.01


def test_random_split_fallback_is_flagged(small_df):
    *_, info = chronological_split(build_features(small_df).drop(columns=[TIME_COL]))
    assert info["method"] == "aléatoire stratifiée" and info["warning"]


def test_preprocessor_fit_on_train_only(small_df):
    f = build_features(small_df)
    tr, va, _, _ = chronological_split(f)
    num, cat = feature_columns(f)
    pre = make_preprocessor(num, cat).fit(tr)
    Xtr = pre.transform(tr)
    assert np.allclose(Xtr[num].mean(), 0, atol=1e-6)
    assert not np.allclose(pre.transform(va)[num].mean(), 0, atol=1e-6)


def test_resample_and_correction(small_df):
    f = build_features(small_df)
    X, y = f[["log_amt", "n_tx_24h"]], f[TARGET]
    Xr, yr, _, beta = resample("under_1:10", X, y)
    assert yr.sum() == y.sum() and (yr == 0).sum() == 10 * y.sum()
    assert np.isclose(beta, 10 * y.sum() / (y == 0).sum())
    assert np.allclose(correct_proba([0.2, 0.5], 1.0), [0.2, 0.5])
    # la cote est bien multipliée par β
    p = correct_proba(np.array([0.5]), beta)[0]
    assert np.isclose(p / (1 - p), beta)
