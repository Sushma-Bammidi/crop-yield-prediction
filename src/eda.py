"""Exploratory data analysis summary."""

import pandas as pd

from .config import TARGET_COLUMN


def summarize_dataset(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_mean": round(df[TARGET_COLUMN].mean(), 3),
        "target_std": round(df[TARGET_COLUMN].std(), 3),
        "target_min": round(df[TARGET_COLUMN].min(), 3),
        "target_max": round(df[TARGET_COLUMN].max(), 3),
        "crops": sorted(df["crop"].unique().tolist()),
        "states": sorted(df["state"].unique().tolist()),
    }


def print_summary(summary: dict) -> None:
    print("\n=== Dataset Summary ===")
    print(f"Rows: {summary['rows']}, Columns: {summary['columns']}")
    print(f"Missing values: {summary['missing_values']}, Duplicates: {summary['duplicate_rows']}")
    print(
        f"Yield — mean: {summary['target_mean']}, std: {summary['target_std']}, "
        f"min: {summary['target_min']}, max: {summary['target_max']}"
    )
    print(f"Crops: {', '.join(summary['crops'])}")
    print(f"States: {', '.join(summary['states'])}")
