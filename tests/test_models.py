import numpy as np

from fraud.config import TARGET
from fraud.pipeline import comparison_table, load_bundle, save_bundle, scores


def test_all_detectors_scored(bundle):
    assert set(bundle["detectors"]) == {"LightGBM", "Régression logistique pondérée", "Isolation Forest",
                                        "LOF", "Autoencodeur"}
    fams = {d.family for d in bundle["detectors"].values()}
    assert fams == {"supervisé", "non supervisé"}
    for name in bundle["detectors"]:
        s = scores(bundle, "test", name)
        assert s.shape == (len(bundle["frames"]["test"]),) and np.isfinite(s).all()


def test_supervised_beats_random(bundle):
    table = comparison_table(bundle).set_index("méthode")
    base = bundle["frames"]["test"][TARGET].mean()
    assert table.loc["LightGBM", "PR-AUC test"] > 5 * base
    assert table.loc["Isolation Forest", "ROC-AUC test"] > 0.6


def test_lightgbm_probabilities_are_corrected(bundle):
    p = scores(bundle, "val", "LightGBM")
    assert 0 <= p.min() and p.max() <= 1
    # après correction, la probabilité moyenne reste proche de la prévalence réelle
    assert p.mean() < 5 * bundle["frames"]["val"][TARGET].mean()


def test_bundle_roundtrip(bundle, tmp_path):
    path = save_bundle(bundle, tmp_path / "b.joblib")
    b2 = load_bundle(path)
    X = bundle["preprocessor"].transform(bundle["frames"]["test"].head(50))
    assert np.allclose(b2["detectors"]["LightGBM"].score(X), bundle["detectors"]["LightGBM"].score(X))
