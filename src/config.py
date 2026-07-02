from pathlib import Path

CROPS = ["Wheat", "Rice", "Maize", "Cotton", "Soybean", "Barley"]
SEASONS = ["Kharif", "Rabi", "Zaid"]
STATES = ["Punjab", "Haryana", "Maharashtra", "Karnataka", "Uttar Pradesh", "Bihar"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "crop_yield_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"

TARGET_COLUMN = "yield_tons_per_hectare"
CATEGORICAL_COLUMNS = ["state", "crop", "season"]
NUMERIC_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "humidity_pct",
    "soil_ph",
    "nitrogen_kg_ha",
    "phosphorus_kg_ha",
    "potassium_kg_ha",
    "area_hectares",
    "pesticide_kg",
    "irrigation_mm",
]
TEST_SIZE = 0.2
RANDOM_STATE = 42
