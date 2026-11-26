
"""MP089 data audit: detect seven data issues and generate findings."""

from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUTPUT = ROOT / "reports" / "findings.yaml"
ROOT_OUTPUT = ROOT / "findings.yaml"


def load_data():
    files = {
        "train": DATA / "train.csv",
        "later": DATA / "later.csv",
        "holdout": DATA / "holdout_features.csv",
    }
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
    return {name: pd.read_csv(path) for name, path in files.items()}


def find_leakage(train, holdout):
    column = "loaves_in_closing_register"
    present = column in train.columns and column not in holdout.columns

    if present:
        corrected_target = train["loaves_sold"].where(
            train["loaves_sold"] <= 1000,
            train["loaves_sold"] / 10,
        )
        print("\nLEAKAGE")
        print("Column:", column)
        print("Unavailable in holdout:", column not in holdout.columns)
        print(
            "Correlation with corrected target:",
            train[column].corr(corrected_target),
        )
        print(
            "Mean absolute difference:",
            (train[column] - corrected_target).abs().mean(),
        )

    return {
        "present": present,
        "column": column if present else None,
    }


def find_unit_mixup(train):
    column = "shelf_hours"
    medians = train.groupby("outlet")[column].median()
    print("\nSHELF HOURS MEDIANS BY OUTLET")
    print(medians)

    result = {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None,
    }

    if len(medians) >= 2:
        ratio = medians.max() / medians.min()
        if 54 <= ratio <= 66:
            group = str(medians.idxmax())
            result = {
                "present": True,
                "column": column,
                "group_value": group,
                "factor": 60,
            }

    return result


def find_sentinel(train):
    candidates = {-1, -9, -99, -999, 999, 9999}

    for column in train.select_dtypes(include="number").columns:
        counts = train[column].value_counts()
        for value, count in counts.items():
            if value in candidates and count >= 5:
                return {
                    "present": True,
                    "column": column,
                    "value": int(value),
                    "count": int(count),
                }

    return {
        "present": False,
        "column": None,
        "value": None,
        "count": None,
    }


def find_duplicates(train):
    count = int(train.duplicated().sum())
    return {"present": count > 0, "count": count}


def find_stuck_sensor(train):
    column = "competitor_shops"
    data = train.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    runs = []

    for outlet, group in data.dropna(subset=["date"]).groupby("outlet"):
        daily = (
            group.groupby("date")[column]
            .agg(lambda x: x.mode().iloc[0])
            .sort_index()
        )

        dates = list(daily.index)
        values = list(daily.values)
        start = 0

        for i in range(1, len(values) + 1):
            if i == len(values) or values[i] != values[start]:
                length = i - start
                if length >= 10:
                    span = (dates[i - 1] - dates[start]).days
                    if span >= 14:
                        runs.append({
                            "outlet": str(outlet),
                            "value": int(values[start]),
                            "start_date": dates[start].strftime("%Y-%m-%d"),
                            "end_date": dates[i - 1].strftime("%Y-%m-%d"),
                            "observed_days": length,
                            "span_days": span,
                        })
                start = i

    print("\nSTUCK SENSOR RUNS")
    for run in runs:
        print(run)

    if not runs:
        return {
            "present": False,
            "column": None,
            "start_date": None,
            "end_date": None,
        }

    longest = max(runs, key=lambda x: x["span_days"])
    return {
        "present": True,
        "column": column,
        "start_date": longest["start_date"],
        "end_date": longest["end_date"],
    }


def find_target_typos(train):
    suspicious = train[train["loaves_sold"] > 1000].copy()
    corrected = suspicious["loaves_sold"] / 10
    difference = (
        corrected - suspicious["loaves_in_closing_register"]
    ).abs()

    suspicious = suspicious[
        (suspicious["loaves_sold"] % 10 == 0)
        & (difference <= 20)
    ]

    print("\nTARGET TYPO CANDIDATES")
    if not suspicious.empty:
        print(suspicious[
            ["record_id", "loaves_sold", "loaves_in_closing_register"]
        ])

    count = int(len(suspicious))
    return {"present": count > 0, "count": count}


def find_spelling_variants(train, later):
    column = "day_type"
    raw = train[column].astype(str)

    normalized = (
        raw.str.strip()
        .str.lower()
        .replace({"wee.": "weekend", "fes.": "festival"})
    )

    later_levels = (
        later[column].astype(str).str.strip().str.lower().nunique()
    )

    print("\nSPELLING VARIANTS")
    print("Original categories:", raw.nunique())
    print("Normalized categories:", normalized.nunique())
    print("Rows normalized:", int((raw != normalized).sum()))
    print("Later dataset categories:", later_levels)

    return {
        "present": raw.nunique() > later_levels,
        "column": column if raw.nunique() > later_levels else None,
        "true_levels": int(later_levels),
    }


def main():
    datasets = load_data()
    train = datasets["train"]
    later = datasets["later"]
    holdout = datasets["holdout"]

    print("DATASET SHAPES")
    for name, df in datasets.items():
        print(f"{name}: {df.shape}")

    findings = {
        "leaking_column": find_leakage(train, holdout),
        "unit_mixup": find_unit_mixup(train),
        "sentinel_value": find_sentinel(train),
        "duplicate_rows": find_duplicates(train),
        "stuck_sensor": find_stuck_sensor(train),
        "target_typos": find_target_typos(train),
        "spelling_variants": find_spelling_variants(train, later),
    }

    report = {
        "group_id": "G186",
        "problem_id": "MP089",
        "data_issues": findings,
        "tracking": {
            "experiment_name": None,
            "best_run_id": None,
            "registered_model_name": None,
            "champion_version": None,
        },
        "monitoring": {
            "psi": {
                "price_rs": None,
                "temp_c": None,
                "footfall_index": None,
                "competitor_shops": None,
                "shelf_hours": None,
                "orders_previous_day": None,
            },
            "most_drifted_feature": None,
            "mae_by_group": {
                "Station Road": None,
                "Mall Road": None,
                "College Gate": None,
            },
            "worst_served_group": None,
            "mae_v1_on_check": None,
            "mae_v2_on_check": None,
            "decision": None,
        },
    }

    for output_path in (OUTPUT, ROOT_OUTPUT):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                report,
                file,
                sort_keys=False,
                allow_unicode=True,
            )

    print("\nDATA ISSUE SUMMARY")
    for name, result in findings.items():
        print(f"{name}: {result['present']}")

    print(f"\nGenerated: {OUTPUT}")
    print(f"Generated: {ROOT_OUTPUT}")


if __name__ == "__main__":
    main()
