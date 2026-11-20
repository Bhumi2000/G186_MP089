"""Shared constants and helpers."""
TARGET = "loaves_sold"
ID_COLUMN = "record_id"
FEATURES = [
    "price_rs", "temp_c", "footfall_index", "competitor_shops",
    "shelf_hours", "orders_previous_day", "day_type", "outlet",
]
NUMERIC_FEATURES = [
    "price_rs", "temp_c", "footfall_index", "competitor_shops",
    "shelf_hours", "orders_previous_day",
]
