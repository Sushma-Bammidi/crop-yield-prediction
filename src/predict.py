"""Load trained models and run yield predictions."""

from pathlib import Path

import joblib
import pandas as pd

from .config import CATEGORICAL_COLUMNS, MODELS_DIR, NUMERIC_COLUMNS
from .feature_engineering import engineer_features, get_engineered_numeric_columns


def models_available() -> bool:
    return (
        (MODELS_DIR / "random_forest.joblib").exists()
        and (MODELS_DIR / "xgboost.joblib").exists()
    )


def load_model(name: str):
    filename = {
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
    }.get(name)
    if not filename:
        raise ValueError(f"Unknown model: {name}")
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python main.py` first.")
    return joblib.load(path)


def build_input_dataframe(inputs: dict) -> pd.DataFrame:
    row = {col: inputs[col] for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS}
    df = pd.DataFrame([row])
    return engineer_features(df)


def predict_yield(inputs: dict, model_name: str = "XGBoost") -> float:
    df = build_input_dataframe(inputs)
    feature_cols = get_engineered_numeric_columns() + CATEGORICAL_COLUMNS
    model = load_model(model_name)
    return float(model.predict(df[feature_cols])[0])


def predict_all_models(inputs: dict) -> dict[str, float]:
    return {
        "Random Forest": predict_yield(inputs, "Random Forest"),
        "XGBoost": predict_yield(inputs, "XGBoost"),
    }
