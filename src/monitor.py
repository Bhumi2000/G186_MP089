
"""MP089 monitoring, v2 training, model promotion and holdout prediction."""

from pathlib import Path
import hashlib

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from common import (
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    clean_data,
)
from train import (
    build_pipeline,
    REGISTERED_MODEL_NAME,
    TRUSTED_TYPES,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_PATH = ROOT / "reports" / "findings.yaml"
TRACKING_DB = ROOT / "mlflow.db"

TRAIN_PATH = DATA / "train.csv"
LATER_PATH = DATA / "later.csv"
HOLDOUT_PATH = DATA / "holdout_features.csv"

V1_PATH = MODEL_DIR / "model.joblib"
V2_PATH = MODEL_DIR / "model_v2.joblib"

PREDICTION_PATH = ROOT / "predictions" / "holdout_predictions.csv"

EXPERIMENT_NAME = "MP089-Bakery-Bread-Sales"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_psi(reference, current, bins=10):
    """Calculate PSI using quantile bins from the reference dataset."""

    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if reference.empty or current.empty:
        return None

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = pd.cut(
        reference, bins=edges, include_lowest=True
    ).value_counts(sort=False)

    cur_counts = pd.cut(
        current, bins=edges, include_lowest=True
    ).value_counts(sort=False)

    epsilon = 1e-6

    ref_pct = np.maximum(
        ref_counts.to_numpy(dtype=float) / len(reference),
        epsilon,
    )
    cur_pct = np.maximum(
        cur_counts.to_numpy(dtype=float) / len(current),
        epsilon,
    )

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def main():
    for path in [TRAIN_PATH, LATER_PATH, HOLDOUT_PATH, V1_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # ---------------------------------------------------------
    # 1. Load and clean datasets
    # ---------------------------------------------------------
    train_raw = pd.read_csv(TRAIN_PATH)
    later_raw = pd.read_csv(LATER_PATH)
    holdout_raw = pd.read_csv(HOLDOUT_PATH)

    train = clean_data(train_raw, has_target=True)
    later = clean_data(later_raw, has_target=True)
    holdout = clean_data(holdout_raw, has_target=False)

    train = train.dropna(subset=[TARGET])

    later["date"] = pd.to_datetime(later["date"], errors="coerce")
    later = later.dropna(subset=["date"]).sort_values("date")

    # Chronological 70/30 split of later data.
    split_index = int(len(later) * 0.70)

    later_train = later.iloc[:split_index].copy()
    later_check = later.iloc[split_index:].copy()

    if later_train.empty or later_check.empty:
        raise ValueError("Insufficient later.csv rows for 70/30 split.")

    print("TRAIN ROWS:", len(train))
    print("LATER TRAIN ROWS:", len(later_train))
    print("LATER CHECK ROWS:", len(later_check))
    print("HOLDOUT ROWS:", len(holdout))

    X_check = later_check[FEATURES]
    y_check = later_check[TARGET]

    # ---------------------------------------------------------
    # 2. Calculate PSI drift for six numeric features
    # ---------------------------------------------------------
    psi_results = {}

    for feature in NUMERIC_FEATURES:
        psi = calculate_psi(
            train[feature],
            later[feature],
            bins=10,
        )
        psi_results[feature] = psi
        print(f"PSI {feature}: {psi:.6f}" if psi is not None
              else f"PSI {feature}: unavailable")

    most_drifted = max(
        psi_results,
        key=lambda feature: (
            psi_results[feature]
            if psi_results[feature] is not None
            else -1
        ),
    )

    print("MOST DRIFTED FEATURE:", most_drifted)

    # ---------------------------------------------------------
    # 3. Evaluate v1 on the chronological check set
    # ---------------------------------------------------------
    v1 = joblib.load(V1_PATH)

    v1_predictions = v1.predict(X_check)
    v1_mae = float(mean_absolute_error(y_check, v1_predictions))

    print("\nV1 CHECK MAE:", v1_mae)

    group_mae = {}

    for outlet in sorted(later_check["outlet"].dropna().unique()):
        mask = later_check["outlet"] == outlet

        group_mae[str(outlet)] = float(
            mean_absolute_error(
                y_check.loc[mask],
                v1.predict(later_check.loc[mask, FEATURES]),
            )
        )

    print("V1 MAE BY OUTLET:")
    for outlet, value in group_mae.items():
        print(f"{outlet}: {value:.4f}")

    worst_group = max(group_mae, key=group_mae.get)

    # ---------------------------------------------------------
    # 4. Train v2 on train + first 70% of later data
    # ---------------------------------------------------------
    combined = pd.concat(
        [train, later_train],
        ignore_index=True,
    )

    X_v2_train = combined[FEATURES]
    y_v2_train = combined[TARGET]

    v2 = build_pipeline(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=0.8,
    )

    train_hash = sha256_file(TRAIN_PATH)

    with mlflow.start_run(run_name="mp089-v2-training") as run:
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("model_version", "v2")
        mlflow.log_param("later_training_fraction", 0.70)
        mlflow.log_param("later_training_rows", len(later_train))
        mlflow.log_param("training_rows_total", len(combined))
        mlflow.log_param("n_estimators", 400)
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_param("max_features", 0.8)
        mlflow.log_param("data_sha256", train_hash)

        v2.fit(X_v2_train, y_v2_train)

        v2_predictions = v2.predict(X_check)
        v2_mae = float(mean_absolute_error(y_check, v2_predictions))

        mlflow.log_metric("check_mae", v2_mae)
        mlflow.log_metric("v1_check_mae", v1_mae)

        mlflow.sklearn.log_model(
            sk_model=v2,
            name="model",
            input_example=X_v2_train.head(3),
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=TRUSTED_TYPES,
        )

        v2_run_id = run.info.run_id

    joblib.dump(v2, V2_PATH)

    print("\nV2 CHECK MAE:", v2_mae)

    # ---------------------------------------------------------
    # 5. Apply the 5% promotion rule
    # ---------------------------------------------------------
    improvement = (
        (v1_mae - v2_mae) / v1_mae
        if v1_mae > 0
        else 0.0
    )

    promote_v2 = improvement >= 0.05

    if promote_v2:
        champion_model = v2
        decision = "promote_v2"
    else:
        champion_model = v1
        decision = "retain_v1"

    print("V2 IMPROVEMENT:", f"{improvement * 100:.2f}%")
    print("PROMOTION DECISION:", decision)

    # ---------------------------------------------------------
    # 6. Set MLflow champion alias
    # ---------------------------------------------------------
    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    versions = sorted(versions, key=lambda item: int(item.version))

    if not versions:
        raise RuntimeError("No registered model versions found.")

    v1_version = versions[0].version

    # v2 was just registered; locate the newest version.
    v2_version = versions[-1].version

    champion_version = (
        v2_version if promote_v2 else v1_version
    )

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        champion_version,
    )

    print("REGISTERED V1 VERSION:", v1_version)
    print("REGISTERED V2 VERSION:", v2_version)
    print("CHAMPION VERSION:", champion_version)

    # ---------------------------------------------------------
    # 7. Predict holdout using the selected champion
    # ---------------------------------------------------------
    holdout_predictions = champion_model.predict(holdout[FEATURES])

    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = pd.DataFrame({
        "record_id": holdout_raw["record_id"],
        "prediction": holdout_predictions,
    })

    output.to_csv(PREDICTION_PATH, index=False)

    print("\nHOLDOUT PREDICTIONS SAVED:", PREDICTION_PATH)
    print("PREDICTION ROWS:", len(output))
    print("OUTPUT COLUMNS:", list(output.columns))

    # ---------------------------------------------------------
    # 8. Update findings.yaml
    # ---------------------------------------------------------
    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = yaml.safe_load(file)

    report["tracking"] = {
        "experiment_name": EXPERIMENT_NAME,
        "best_run_id": v2_run_id if promote_v2 else None,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "champion_version": int(champion_version),
    }

    report["monitoring"] = {
        "psi": psi_results,
        "most_drifted_feature": most_drifted,
        "mae_by_group": {
            "Station Road": group_mae.get("Station Road"),
            "Mall Road": group_mae.get("Mall Road"),
            "College Gate": group_mae.get("College Gate"),
        },
        "worst_served_group": worst_group,
        "mae_v1_on_check": v1_mae,
        "mae_v2_on_check": v2_mae,
        "decision": decision,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            report,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    print("\nFINDINGS REPORT UPDATED:", REPORT_PATH)
    print("\nMONITORING COMPLETE")


if __name__ == "__main__":
    main()
