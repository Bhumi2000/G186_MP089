from pathlib import Path
import pandas as pd

DATA = Path("data")

def audit(path):
    df = pd.read_csv(path)
    print(f"\n{'='*60}\nFILE: {path}\n{'='*60}")

    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("Duplicate record IDs:", df["record_id"].duplicated().sum())

    if "date" in df:
        dates = pd.to_datetime(df["date"], errors="coerce")
        print("\nInvalid dates:", dates.isna().sum())
        print("Date range:", dates.min(), "to", dates.max())

    print("\nCategory counts:")
    for col in ["day_type", "outlet"]:
        if col in df:
            print(f"\n{col}:\n", df[col].value_counts(dropna=False))

    print("\nNumeric summary:\n", df.select_dtypes(include="number").describe())

    if "loaves_sold" in df:
        print("\nTARGET VALUES > 1000:")
        print(df[df["loaves_sold"] > 1000][["record_id", "date", "loaves_sold"]])

    if "orders_previous_day" in df:
        print("\nORDERS > 850:")
        print(df[df["orders_previous_day"] > 850][
            ["record_id", "date", "orders_previous_day"]
        ])

    if "shelf_hours" in df:
        print("\nSHELF HOURS > 24:")
        print(df[df["shelf_hours"] > 24][["record_id", "date", "shelf_hours"]])

    if "loaves_in_closing_register" in df and "loaves_sold" in df:
        print("\nREGISTER/TARGET MISMATCHES:",
              (df["loaves_in_closing_register"] != df["loaves_sold"]).sum())

    if "outlet" in df:
        print("\nNUMERIC STATISTICS BY OUTLET:")
        print(df.groupby("outlet").select_dtypes(include="number").mean()
              if False else df.groupby("outlet")[df.select_dtypes(include="number").columns].mean())

if __name__ == "__main__":
    for name in ["train.csv", "later.csv", "holdout_features.csv"]:
        path = DATA / name
        if path.exists():
            audit(path)