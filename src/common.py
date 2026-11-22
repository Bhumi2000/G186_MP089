
"""Shared constants and data-cleaning functions for MP089."""

import pandas as pd

TARGET = "loaves_sold"
ID_COLUMN = "record_id"

FEATURES = [
    "price_rs",
    "temp_c",
    "footfall_index",
    "competitor_shops",
    "shelf_hours",
    "orders_previous_day",
    "day_type",
    "outlet",
]

NUMERIC_FEATURES = [
    "price_rs",
    "temp_c",
    "footfall_index",
    "competitor_shops",
    "shelf_hours",
    "orders_previous_day",
]

CATEGORICAL_FEATURES = ["day_type", "outlet"]


def clean_data(df, has_target=True):
    """Apply the documented MP089 cleaning rules without using leakage."""

    data = df.copy()

    # Normalize categorical spelling, case and whitespace.
    data["day_type"] = (
        data["day_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "wee.": "weekend",
            "fes.": "festival",
        })
    )

    # College Gate shelf_hours were recorded in minutes, not hours.
    mask = data["outlet"].astype(str).str.strip() == "College Gate"
    data.loc[mask, "shelf_hours"] = data.loc[mask, "shelf_hours"] / 60

    # Correct the 11 extra-zero target entries supported by the audit.
    if has_target and "loaves_in_closing_register" in data.columns:
        suspicious = (
            (data[TARGET] > 1000)
            & (data[TARGET] % 10 == 0)
            & (
                (data[TARGET] / 10
                 - data["loaves_in_closing_register"]).abs() <= 20
            )
        )
        data.loc[suspicious, TARGET] = data.loc[suspicious, TARGET] / 10

    # Never allow this post-event field into the model.
    data = data.drop(columns=["loaves_in_closing_register"], errors="ignore")

    return data
