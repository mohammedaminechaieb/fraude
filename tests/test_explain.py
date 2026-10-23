import numpy as np

from fraud.explain import grouped, permutation_importance, reason_codes, shap_explanation


def _alerts(bundle, n=5):
    test = bundle["frames"]["test"]
    return test.sort_values("score::LightGBM", ascending=False).head(n)


def test_shap_is_additive_to_published_score(bundle):
    rows = _alerts(bundle)
    exp = shap_explanation(bundle, rows)
    p = rows["score::LightGBM"].to_numpy()
    assert np.allclose(exp.values.sum(axis=1) + exp.base_values, np.log(p / (1 - p)), atol=1e-6)


def test_grouping_preserves_total(bundle):
    rows = _alerts(bundle)
    exp = shap_explanation(bundle, rows)
    g = grouped(exp, rows)
    assert np.allclose(g.values.sum(axis=1), exp.values.sum(axis=1))
    assert "catégorie" in g.feature_names and len(g.feature_names) < len(exp.feature_names)


def test_reason_codes(bundle):
    codes = reason_codes(bundle, _alerts(bundle))
    for rc in codes:
        assert 3 <= len(rc) <= 5
        contribs = [c["contribution"] for c in rc]
        assert all(c > 0 for c in contribs) and contribs == sorted(contribs, reverse=True)
        assert len({c["code"] for c in rc}) == len(rc) and all(c["text"] for c in rc)


def test_permutation_importance(bundle):
    imp = permutation_importance(bundle, bundle["frames"]["val"], n_repeats=1)
    assert imp["importance"].iloc[0] > 0
    assert imp["importance"].is_monotonic_decreasing
