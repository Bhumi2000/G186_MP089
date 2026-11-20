"""Starter training script. Refine preprocessing after the audit."""
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from common import FEATURES, NUMERIC_FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/train.csv"


def build_pipeline():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, ["day_type", "outlet"])])
    return Pipeline([("prep", prep), ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))])


def main():
    df = pd.read_csv(DATA)
    # TODO: apply the justified cleaning rules discovered in the audit.
    df = df.dropna(subset=[TARGET])
    X, y = df[FEATURES], df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_pipeline()
    mlflow.set_experiment("MP089-Bakery-Bread-Sales")
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 300)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        mlflow.log_metric("validation_mae", mae)
        mlflow.sklearn.log_model(model, "model")
        print("Validation MAE:", mae)

if __name__ == "__main__":
    main()
