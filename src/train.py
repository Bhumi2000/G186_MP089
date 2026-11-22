
"""Train, evaluate and register MP089 Random Forest models."""

from pathlib import Path
import hashlib
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    clean_data,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
TRACKING_DB = ROOT / "mlflow.db"

EXPERIMENT_NAME = "MP089-Bakery-Bread-Sales"
REGISTERED_MODEL_NAME = "MP089-Bakery-RandomForest"

TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
]


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_pipeline(n_estimators, min_samples_leaf, max_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")

    original_hash = sha256_file(TRAIN_PATH)

    raw = pd.read_csv(TRAIN_PATH)
    cleaned = clean_data(raw, has_target=True)
    cleaned = cleaned.dropna(subset=[TARGET])

    X = cleaned[FEATURES]
    y = cleaned[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    configurations = [
        {
            "n_estimators": 200,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        },
        {
            "n_estimators": 300,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        },
        {
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "max_features": 0.8,
        },
    ]

    results = []

    for index, config in enumerate(configurations, start=1):
        model = build_pipeline(**config)

        with mlflow.start_run(
            run_name=f"mp089-v1-candidate-{index}"
        ) as run:

            mlflow.log_params(config)
            mlflow.log_param("model", "RandomForestRegressor")
            mlflow.log_param("cleaning", "mp089_audit_rules")
            mlflow.log_param("validation_test_size", 0.2)
            mlflow.log_param("validation_random_state", 42)
            mlflow.log_param("data_sha256", original_hash)

            model.fit(X_train, y_train)

            predictions = model.predict(X_val)
            mae = float(mean_absolute_error(y_val, predictions))

            mlflow.log_metric("validation_mae", mae)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=X_train.head(3),
                skops_trusted_types=TRUSTED_TYPES,
            )

            results.append({
                "run_id": run.info.run_id,
                "mae": mae,
                "model": model,
                "config": config,
            })

            print(
                f"Run {index} | MAE: {mae:.4f} "
                f"| ID: {run.info.run_id}"
            )

    best = min(results, key=lambda result: result["mae"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)

    with mlflow.start_run(
        run_name="mp089-v1-registration"
    ):
        mlflow.log_param("selected_run_id", best["run_id"])
        mlflow.log_param("data_sha256", original_hash)
        mlflow.log_metric("validation_mae", best["mae"])

        mlflow.sklearn.log_model(
            sk_model=best["model"],
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(3),
            skops_trusted_types=TRUSTED_TYPES,
        )

    print("\nTRAINING COMPLETE")
    print("Best validation MAE:", round(best["mae"], 4))
    print("Best configuration:", json.dumps(best["config"]))
    print("Best run ID:", best["run_id"])
    print("Original train SHA256:", original_hash)
    print("Saved model:", MODEL_PATH)
    print("MLflow database:", TRACKING_DB)
    print("Registered model:", REGISTERED_MODEL_NAME)


if __name__ == "__main__":
    main()
