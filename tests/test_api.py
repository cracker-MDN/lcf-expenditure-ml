import pytest
from fastapi.testclient import TestClient

from api.main import app

EXAMPLE = {"income": 550, "maininc": 0, "A172": 1, "A093r": 1, "A094r": 1, "A121r": 1,
           "SexHRP": 1, "hhsize": 2, "G018r": 2, "G019r": 1, "Gorx": 1}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:          # "with" runs the start-up code that loads the models
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_works(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 30 < body["amount"] < 1200
    assert body["group"] in {"Low", "Mid", "High"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-6


def test_api_matches_the_app_function(client):
    from src.model import load_models
    from src.predict import predict_household
    direct = predict_household(EXAMPLE, *load_models())
    assert client.post("/predict", json=EXAMPLE).json()["amount"] == pytest.approx(direct["amount"])


def test_impossible_household_is_rejected(client):
    r = client.post("/predict", json={**EXAMPLE, "hhsize": 1})      # 2 adults, 1 person
    assert r.status_code == 422


def test_out_of_range_income_is_rejected(client):
    assert client.post("/predict", json={**EXAMPLE, "income": 5000}).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "income": -1}).status_code == 422


def test_missing_field_is_rejected(client):
    bad = {k: v for k, v in EXAMPLE.items() if k != "Gorx"}
    assert client.post("/predict", json=bad).status_code == 422
