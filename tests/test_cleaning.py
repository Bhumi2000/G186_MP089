import pandas as pd

from src.common import clean_data


def test_day_type_normalization():
    df = pd.DataFrame({
        "day_type": [" weekday ", "wee.", "fes."],
        "outlet": ["Station Road"] * 3,
        "shelf_hours": [10, 10, 10],
    })

    result = clean_data(df, has_target=False)

    assert result["day_type"].tolist() == [
        "weekday", "weekend", "festival"
    ]


def test_college_gate_shelf_hours_conversion():
    df = pd.DataFrame({
        "day_type": ["weekday"],
        "outlet": ["College Gate"],
        "shelf_hours": [600],
    })

    result = clean_data(df, has_target=False)

    assert result.loc[0, "shelf_hours"] == 10


def test_target_typo_correction_and_leakage_removal():
    df = pd.DataFrame({
        "day_type": ["weekday"],
        "outlet": ["Station Road"],
        "shelf_hours": [10],
        "loaves_sold": [1830],
        "loaves_in_closing_register": [180],
    })

    result = clean_data(df, has_target=True)

    assert result.loc[0, "loaves_sold"] == 183
    assert "loaves_in_closing_register" not in result.columns


def test_holdout_cleaning_without_target():
    df = pd.DataFrame({
        "day_type": ["wee."],
        "outlet": ["Station Road"],
        "shelf_hours": [10],
        "price_rs": [20],
    })

    result = clean_data(df, has_target=False)

    assert result.loc[0, "day_type"] == "weekend"
    assert "loaves_sold" not in result.columns
