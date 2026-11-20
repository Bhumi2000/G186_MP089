from fastapi.testclient import TestClient
from src.api import app

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
