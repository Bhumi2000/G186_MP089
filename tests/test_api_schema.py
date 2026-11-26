import json
from io import StringIO

from fastapi.testclient import TestClient
from src.api import app, prediction_logger

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_price_rejected():
    payload = dict(price_rs=9.99, temp_c=30, footfall_index=1, competitor_shops=2,
                   shelf_hours=12, orders_previous_day=100, day_type="weekday", outlet="Station Road")
    assert client.post("/predict", json=payload).status_code == 422


def test_orders_rejected():
    payload = dict(price_rs=20, temp_c=30, footfall_index=1, competitor_shops=2,
                   shelf_hours=12, orders_previous_day=851, day_type="weekday", outlet="Station Road")
    assert client.post("/predict", json=payload).status_code == 422


def test_unexpected_field_rejected():
    payload = dict(price_rs=20, temp_c=30, footfall_index=1, competitor_shops=2,
                   shelf_hours=12, orders_previous_day=100, day_type="weekday",
                   outlet="Station Road", unexpected="extra")
    assert client.post("/predict", json=payload).status_code == 422


def test_prediction_returns_value_and_logs_json(monkeypatch):
    class StubModel:
        def predict(self, rows):
            assert len(rows) == 1
            return [123.5]

    monkeypatch.setattr("src.api.MODEL_PATH", type("PathStub", (), {"exists": lambda self: True})())
    monkeypatch.setattr("src.api.joblib.load", lambda path: StubModel())
    log_stream = StringIO()
    monkeypatch.setattr(prediction_logger.handlers[0], "stream", log_stream)
    payload = dict(price_rs=20, temp_c=30, footfall_index=1, competitor_shops=2,
                   shelf_hours=12, orders_previous_day=100, day_type="weekday", outlet="Station Road")

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": 123.5}
    logged = log_stream.getvalue().strip()
    assert json.loads(logged) == {"event": "prediction", "prediction": 123.5}
