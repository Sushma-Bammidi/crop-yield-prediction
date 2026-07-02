"""Crop Yield Prediction — main pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATH, OUTPUT_DIR
from src.data_preprocessing import (
    clean_data,
    load_data,
    split_features_target,
    train_val_split,
)
from src.eda import print_summary, summarize_dataset
from src.feature_engineering import engineer_features, get_engineered_numeric_columns
from src.train_models import (
    print_results,
    save_metrics_report,
    train_random_forest,
    train_xgboost,
)
from src.visualization import (
    plot_correlation_heatmap,
    plot_yield_by_crop,
    plot_yield_distribution,
)


def ensure_data_exists() -> None:
    if DATA_PATH.exists():
        return

    print("Dataset not found. Generating sample data...")
    from scripts.generate_sample_data import generate_dataset

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset().to_csv(DATA_PATH, index=False)
    print(f"Created dataset at {DATA_PATH}")


def run_pipeline() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_data_exists()

    print("Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    summary = summarize_dataset(df)
    print_summary(summary)

    numeric_cols = get_engineered_numeric_columns()
    print("\nGenerating exploratory visualizations...")
    plot_yield_distribution(df)
    plot_yield_by_crop(df)
    plot_correlation_heatmap(df, numeric_cols)

    X, y = split_features_target(df, numeric_cols)
    X_train, X_test, y_train, y_test = train_val_split(X, y)

    print(f"\nTraining on {len(X_train)} samples, evaluating on {len(X_test)} samples...")
    results = [
        train_random_forest(X_train, X_test, y_train, y_test, numeric_cols),
        train_xgboost(X_train, X_test, y_train, y_test, numeric_cols),
    ]

    print_results(results)
    save_metrics_report(results, OUTPUT_DIR / "model_metrics.csv")

    best = max(results, key=lambda r: r.r2)
    print(f"\nBest model: {best.name} (R² = {best.r2:.4f})")
    print(f"Figures saved to: {OUTPUT_DIR / 'figures'}")
    print(f"Metrics saved to: {OUTPUT_DIR / 'model_metrics.csv'}")


if __name__ == "__main__":
    run_pipeline()
