"""Data loading and preprocessing utilities."""

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and invalid records."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()

    for col in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned[NUMERIC_COLUMNS] = cleaned[NUMERIC_COLUMNS].fillna(
        cleaned[NUMERIC_COLUMNS].median()
    )
    cleaned[CATEGORICAL_COLUMNS] = cleaned[CATEGORICAL_COLUMNS].fillna("Unknown")

    # Remove physically implausible values
    cleaned = cleaned[cleaned["yield_tons_per_hectare"] > 0]
    cleaned = cleaned[cleaned["soil_ph"].between(4.5, 9.5)]
    cleaned = cleaned[cleaned["humidity_pct"].between(10, 100)]

    return cleaned.reset_index(drop=True)


def build_preprocessor(numeric_columns: list[str] | None = None) -> ColumnTransformer:
    numeric_cols = numeric_columns or NUMERIC_COLUMNS
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def split_features_target(
    df: pd.DataFrame, numeric_columns: list[str] | None = None
) -> Tuple[pd.DataFrame, pd.Series]:
    numeric_cols = numeric_columns or NUMERIC_COLUMNS
    feature_columns = numeric_cols + CATEGORICAL_COLUMNS
    X = df[feature_columns]
    y = df[TARGET_COLUMN]
    return X, y


def train_val_split(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
