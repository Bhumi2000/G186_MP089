"""Generate final holdout prediction CSV using the champion model."""
from pathlib import Path
import pandas as pd
import joblib
from common import FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/champion.joblib"


def main():
    holdout = pd.read_csv(ROOT / "data/holdout_features.csv")
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(holdout[FEATURES])
    out = pd.DataFrame({"record_id": holdout["record_id"], "prediction": predictions})
    out_path = ROOT / "predictions/holdout_predictions.csv"
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out_path)

if __name__ == "__main__":
    main()
