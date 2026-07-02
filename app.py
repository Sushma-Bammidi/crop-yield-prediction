"""Streamlit UI for crop yield prediction."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CROPS,
    DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    SEASONS,
    STATES,
)
from src.predict import models_available, predict_all_models  # noqa: E402

st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #558b2f;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #2e7d32;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .result-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_models() -> bool:
    if models_available():
        return True

    st.warning("Models not found. Training now — this may take a minute...")
    with st.spinner("Training Random Forest and XGBoost models..."):
        from main import run_pipeline

        run_pipeline()
    return models_available()


def default_inputs() -> dict:
    return {
        "state": "Punjab",
        "crop": "Wheat",
        "season": "Rabi",
        "rainfall_mm": 850.0,
        "avg_temp_c": 24.0,
        "max_temp_c": 32.0,
        "min_temp_c": 16.0,
        "humidity_pct": 65.0,
        "soil_ph": 6.8,
        "nitrogen_kg_ha": 80.0,
        "phosphorus_kg_ha": 35.0,
        "potassium_kg_ha": 45.0,
        "area_hectares": 5.0,
        "pesticide_kg": 4.0,
        "irrigation_mm": 600.0,
    }


def render_sidebar() -> None:
    st.sidebar.title("🌾 Crop Yield AI")
    st.sidebar.markdown("Predict agricultural yield using **Random Forest** and **XGBoost**.")
    st.sidebar.divider()

    metrics_path = OUTPUT_DIR / "model_metrics.csv"
    if metrics_path.exists():
        st.sidebar.subheader("Model Performance")
        metrics = pd.read_csv(metrics_path)
        for _, row in metrics.iterrows():
            st.sidebar.metric(
                label=row["model"],
                value=f"R² {row['r2']:.3f}",
                delta=f"MAE {row['mae']:.2f} t/ha",
                delta_color="off",
            )

    st.sidebar.divider()
    st.sidebar.caption("Built with Python · Scikit-learn · XGBoost · Streamlit")


def render_prediction_form() -> dict:
    defaults = default_inputs()

    col1, col2, col3 = st.columns(3)
    with col1:
        state = st.selectbox("State", STATES, index=STATES.index(defaults["state"]))
        crop = st.selectbox("Crop", CROPS, index=CROPS.index(defaults["crop"]))
    with col2:
        season = st.selectbox("Season", SEASONS, index=SEASONS.index(defaults["season"]))
        area = st.number_input("Area (hectares)", min_value=0.1, value=defaults["area_hectares"], step=0.5)
    with col3:
        irrigation = st.number_input("Irrigation (mm)", min_value=0.0, value=defaults["irrigation_mm"], step=10.0)
        pesticide = st.number_input("Pesticide (kg)", min_value=0.0, value=defaults["pesticide_kg"], step=0.5)

    st.subheader("Weather & Soil")
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=defaults["rainfall_mm"], step=10.0)
        humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=defaults["humidity_pct"], step=1.0)
    with w2:
        avg_temp = st.number_input("Avg Temperature (°C)", value=defaults["avg_temp_c"], step=0.5)
        max_temp = st.number_input("Max Temperature (°C)", value=defaults["max_temp_c"], step=0.5)
    with w3:
        min_temp = st.number_input("Min Temperature (°C)", value=defaults["min_temp_c"], step=0.5)
        soil_ph = st.number_input("Soil pH", min_value=4.5, max_value=9.5, value=defaults["soil_ph"], step=0.1)
    with w4:
        nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0.0, value=defaults["nitrogen_kg_ha"], step=5.0)
        phosphorus = st.number_input("Phosphorus (kg/ha)", min_value=0.0, value=defaults["phosphorus_kg_ha"], step=5.0)
        potassium = st.number_input("Potassium (kg/ha)", min_value=0.0, value=defaults["potassium_kg_ha"], step=5.0)

    return {
        "state": state,
        "crop": crop,
        "season": season,
        "rainfall_mm": rainfall,
        "avg_temp_c": avg_temp,
        "max_temp_c": max_temp,
        "min_temp_c": min_temp,
        "humidity_pct": humidity,
        "soil_ph": soil_ph,
        "nitrogen_kg_ha": nitrogen,
        "phosphorus_kg_ha": phosphorus,
        "potassium_kg_ha": potassium,
        "area_hectares": area,
        "pesticide_kg": pesticide,
        "irrigation_mm": irrigation,
    }


def render_results(predictions: dict[str, float], inputs: dict) -> None:
    best_model = max(predictions, key=predictions.get)
    best_yield = predictions[best_model]
    total_production = best_yield * inputs["area_hectares"]

    st.markdown(
        f"""
        <div class="result-card">
            <div>Best prediction — <strong>{best_model}</strong></div>
            <div class="result-value">{best_yield:.2f} tons/hectare</div>
            <div>Estimated total production: <strong>{total_production:.2f} tons</strong>
            across {inputs['area_hectares']:.1f} hectares</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Model Comparison")
    cols = st.columns(len(predictions))
    for col, (name, value) in zip(cols, predictions.items()):
        with col:
            st.metric(name, f"{value:.2f} t/ha")

    comparison_df = pd.DataFrame(
        {"Model": list(predictions.keys()), "Predicted Yield (t/ha)": list(predictions.values())}
    )
    st.bar_chart(comparison_df.set_index("Model"))


def render_analytics_tab() -> None:
    figures = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    if not figures:
        st.info("No analytics charts yet. Run `python main.py` to generate EDA plots.")
        return

    for path in figures:
        st.image(str(path), caption=path.stem.replace("_", " ").title(), use_container_width=True)


def main() -> None:
    render_sidebar()

    st.markdown('<p class="main-header">Crop Yield Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Enter farm and environmental conditions to estimate crop yield.</p>',
        unsafe_allow_html=True,
    )

    if not ensure_models():
        st.error("Could not load or train models. Check that dependencies are installed.")
        st.stop()

    tab_predict, tab_analytics, tab_data = st.tabs(["Predict", "Analytics", "Dataset"])

    with tab_predict:
        inputs = render_prediction_form()
        if st.button("Predict Yield", type="primary", use_container_width=False):
            if inputs["max_temp_c"] < inputs["min_temp_c"]:
                st.error("Max temperature must be greater than or equal to min temperature.")
            else:
                with st.spinner("Running models..."):
                    predictions = predict_all_models(inputs)
                render_results(predictions, inputs)

    with tab_analytics:
        render_analytics_tab()

    with tab_data:
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            st.dataframe(df.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(df):,} records from {DATA_PATH.name}")
        else:
            st.info("Dataset not found. It will be created when you train models.")


if __name__ == "__main__":
    main()
