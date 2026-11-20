"""Non-destructive initial data audit. Extend to match the seven handbook checks."""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def audit(path: Path) -> None:
    df = pd.read_csv(path)
    print(f"\n=== {path.name} ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Dtypes:\n", df.dtypes)
    print("Missing values:\n", df.isna().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Duplicate record_id:", df["record_id"].duplicated().sum() if "record_id" in df else "record_id missing")
    print("Numeric summary:\n", df.describe(include="all").transpose())
    if "date" in df:
        dates = pd.to_datetime(df["date"], errors="coerce")
        print("Invalid dates:", dates.isna().sum())
        print("Date range:", dates.min(), "to", dates.max())
    if "loaves_in_closing_register" in df and "loaves_sold" in df:
        print("Closing-register/target mismatches:", (df["loaves_in_closing_register"] != df["loaves_sold"]).sum())


if __name__ == "__main__":
    for filename in ("train.csv", "later.csv", "holdout_features.csv"):
        p = DATA / filename
        if p.exists():
            audit(p)
        else:
            print(f"Missing file: {p}")
