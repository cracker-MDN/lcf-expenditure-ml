import pytest

from src.config import FEATURES_B, TARGET
from src.data import load_clean, make_split
from src.model import load_models, train_models
from src.predict import check_inputs, predict_household
from sklearn.metrics import f1_score, r2_score

EXAMPLE = {"income": 560.0, "maininc": 0, "A172": 1, "A093r": 1, "A094r": 2, "A121r": 3,
           "SexHRP": 1, "hhsize": 2, "G018r": 2, "G019r": 1, "Gorx": 7}


@pytest.fixture(scope="module")
def models():
    regressor, classifier, _ = train_models()
    return regressor, classifier


def test_models_reproduce_thesis_scores(models):
    regressor, classifier = models
    _, test, _ = make_split(load_clean())
    r2 = r2_score(test[TARGET], regressor.predict(test[FEATURES_B]))
    f1 = f1_score(test["segment"], classifier.predict(test[FEATURES_B]), average="macro")
    assert round(r2, 3) == 0.568
    assert round(f1, 3) == 0.672


def test_prediction_is_sensible(models):
    result = predict_household(EXAMPLE, *models)
    assert 30 < result["amount"] < 1200
    assert result["group"] in {"Low", "Mid", "High"}
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-9


def test_higher_income_gives_higher_estimate(models):
    low = predict_household({**EXAMPLE, "income": 150.0}, *models)["amount"]
    high = predict_household({**EXAMPLE, "income": 1000.0}, *models)["amount"]
    assert high > low


def test_bad_inputs_are_rejected(models):
    assert check_inputs({**EXAMPLE, "income": -5})
    assert check_inputs({**EXAMPLE, "Gorx": 99})
    assert check_inputs({**EXAMPLE, "hhsize": 1})          # 2 adults cannot be 1 person
    with pytest.raises(ValueError):
        predict_household({**EXAMPLE, "income": 5000}, *models)


def test_every_real_household_passes_the_checks():
    df = load_clean()
    for _, row in df.iterrows():
        assert not check_inputs({k: row[k] for k in FEATURES_B})
