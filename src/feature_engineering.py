"""Feature engineering for crop yield prediction."""

import pandas as pd

from .config import NUMERIC_COLUMNS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived environmental and agronomic features."""
    engineered = df.copy()

    engineered["temp_range_c"] = engineered["max_temp_c"] - engineered["min_temp_c"]
    engineered["npk_total"] = (
        engineered["nitrogen_kg_ha"]
        + engineered["phosphorus_kg_ha"]
        + engineered["potassium_kg_ha"]
    )
    engineered["rainfall_per_temp"] = engineered["rainfall_mm"] / (
        engineered["avg_temp_c"].abs() + 1
    )
    engineered["irrigation_rainfall_ratio"] = engineered["irrigation_mm"] / (
        engineered["rainfall_mm"] + 1
    )
    engineered["pesticide_per_area"] = (
        engineered["pesticide_kg"] / (engineered["area_hectares"] + 0.1)
    )
    engineered["soil_ph_deviation"] = (engineered["soil_ph"] - 6.8).abs()
    engineered["nitrogen_phosphorus_ratio"] = engineered["nitrogen_kg_ha"] / (
        engineered["phosphorus_kg_ha"] + 1
    )

    return engineered


def get_engineered_numeric_columns() -> list[str]:
    return NUMERIC_COLUMNS + [
        "temp_range_c",
        "npk_total",
        "rainfall_per_temp",
        "irrigation_rainfall_ratio",
        "pesticide_per_area",
        "soil_ph_deviation",
        "nitrogen_phosphorus_ratio",
    ]
