
"""Shared constants and data-cleaning functions for MP089."""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "params.yaml").open("r", encoding="utf-8") as file:
    PARAMS = yaml.safe_load(file)

CLEANING_PARAMS = PARAMS["cleaning"]

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
    day_type_mapping = CLEANING_PARAMS["day_type_mapping"]
    data["day_type"] = (
        data["day_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(day_type_mapping)
    )

    # The audit confirmed College Gate shelf_hours are recorded in minutes.
    mask = data["outlet"].astype(str).str.strip() == "College Gate"
    data.loc[mask, "shelf_hours"] = (
        data.loc[mask, "shelf_hours"]
        / CLEANING_PARAMS["shelf_hours_conversion_divisor"]
    )

    # Correct the 11 extra-zero target entries supported by the audit.
    if has_target and "loaves_in_closing_register" in data.columns:
        suspicious = (
            (data[TARGET] > CLEANING_PARAMS["target_outlier_threshold"])
            & (data[TARGET] % CLEANING_PARAMS["target_scale_divisor"] == 0)
            & (
                (data[TARGET] / CLEANING_PARAMS["target_scale_divisor"]
                 - data["loaves_in_closing_register"]).abs()
                <= CLEANING_PARAMS["target_register_tolerance"]
            )
        )
        data.loc[suspicious, TARGET] = (
            data.loc[suspicious, TARGET]
            / CLEANING_PARAMS["target_scale_divisor"]
        )

    # Never allow this post-event field into the model.
    data = data.drop(columns=["loaves_in_closing_register"], errors="ignore")

    return data
