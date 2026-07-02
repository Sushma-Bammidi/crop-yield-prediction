"""Generate a realistic synthetic crop yield dataset for demonstration."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CROPS, SEASONS, STATES


def _base_yield(crop: str) -> float:
    return {
        "Wheat": 3.2,
        "Rice": 3.8,
        "Maize": 2.9,
        "Cotton": 1.5,
        "Soybean": 1.8,
        "Barley": 2.5,
    }[crop]


def generate_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_samples):
        crop = rng.choice(CROPS)
        season = rng.choice(SEASONS)
        state = rng.choice(STATES)

        rainfall = float(rng.normal(850, 220))
        avg_temp = float(rng.normal(26, 5))
        max_temp = avg_temp + float(rng.uniform(4, 10))
        min_temp = avg_temp - float(rng.uniform(4, 10))
        humidity = float(np.clip(rng.normal(65, 12), 30, 95))
        soil_ph = float(np.clip(rng.normal(6.5, 0.6), 5.0, 8.5))
        nitrogen = float(rng.uniform(20, 120))
        phosphorus = float(rng.uniform(10, 60))
        potassium = float(rng.uniform(15, 80))
        area_hectares = float(rng.uniform(0.5, 25))
        pesticide_kg = float(rng.uniform(0.5, 15))
        irrigation_mm = float(rng.uniform(200, 1200))

        # Yield model with noise and realistic environmental relationships
        yield_tph = (
            _base_yield(crop)
            + 0.0012 * rainfall
            - 0.04 * abs(avg_temp - 24)
            + 0.008 * nitrogen
            + 0.012 * phosphorus
            + 0.006 * potassium
            - 0.15 * abs(soil_ph - 6.8)
            + 0.0004 * irrigation_mm
            - 0.02 * pesticide_kg
            + rng.normal(0, 0.35)
        )
        yield_tph = float(np.clip(yield_tph, 0.5, 8.0))

        rows.append(
            {
                "state": state,
                "crop": crop,
                "season": season,
                "rainfall_mm": round(rainfall, 2),
                "avg_temp_c": round(avg_temp, 2),
                "max_temp_c": round(max_temp, 2),
                "min_temp_c": round(min_temp, 2),
                "humidity_pct": round(humidity, 2),
                "soil_ph": round(soil_ph, 2),
                "nitrogen_kg_ha": round(nitrogen, 2),
                "phosphorus_kg_ha": round(phosphorus, 2),
                "potassium_kg_ha": round(potassium, 2),
                "area_hectares": round(area_hectares, 2),
                "pesticide_kg": round(pesticide_kg, 2),
                "irrigation_mm": round(irrigation_mm, 2),
                "yield_tons_per_hectare": round(yield_tph, 3),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic crop yield data.")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "crop_yield_data.csv",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(n_samples=args.samples, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
