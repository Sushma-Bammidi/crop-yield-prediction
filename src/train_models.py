"""Model training and evaluation for crop yield prediction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .config import MODELS_DIR, RANDOM_STATE
from .data_preprocessing import build_preprocessor
from .visualization import plot_actual_vs_predicted, plot_feature_importance


@dataclass
class ModelResult:
    name: str
    r2: float
    mae: float
    rmse: float
    model_path: Path
    predictions: np.ndarray


def _get_feature_names(preprocessor, numeric_cols: list[str]) -> list[str]:
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_features = list(cat_encoder.get_feature_names_out())
    return numeric_cols + cat_features


def _evaluate(name: str, y_true, y_pred) -> dict[str, float]:
    return {
        "model": name,
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numeric_cols: list[str],
    models_dir: Path = MODELS_DIR,
) -> ModelResult:
    models_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_cols)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    model_path = models_dir / "random_forest.joblib"
    joblib.dump(pipeline, model_path)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = _get_feature_names(preprocessor, numeric_cols)
    plot_feature_importance(feature_names, model.feature_importances_, "Random Forest")
    plot_actual_vs_predicted(y_test, predictions, "Random Forest")

    metrics = _evaluate("Random Forest", y_test, predictions)
    return ModelResult(
        name=metrics["model"],
        r2=metrics["r2"],
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        model_path=model_path,
        predictions=predictions,
    )


def train_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numeric_cols: list[str],
    models_dir: Path = MODELS_DIR,
) -> ModelResult:
    models_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_cols)),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    objective="reg:squarederror",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    model_path = models_dir / "xgboost.joblib"
    joblib.dump(pipeline, model_path)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = _get_feature_names(preprocessor, numeric_cols)
    plot_feature_importance(feature_names, model.feature_importances_, "XGBoost")
    plot_actual_vs_predicted(y_test, predictions, "XGBoost")

    metrics = _evaluate("XGBoost", y_test, predictions)
    return ModelResult(
        name=metrics["model"],
        r2=metrics["r2"],
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        model_path=model_path,
        predictions=predictions,
    )


def print_results(results: list[ModelResult]) -> None:
    print("\n=== Model Evaluation (Test Set) ===")
    for result in results:
        print(
            f"{result.name:15} | R²: {result.r2:.4f} | "
            f"MAE: {result.mae:.4f} | RMSE: {result.rmse:.4f}"
        )
        print(f"  Saved model: {result.model_path}")


def save_metrics_report(results: list[ModelResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = [
        {"model": r.name, "r2": r.r2, "mae": r.mae, "rmse": r.rmse}
        for r in results
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
