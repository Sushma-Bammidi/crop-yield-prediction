"""Matplotlib-based exploratory and model evaluation plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

from .config import FIGURES_DIR, TARGET_COLUMN


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_yield_distribution(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], kde=True, ax=ax, color="#2e7d32")
    ax.set_title("Crop Yield Distribution")
    ax.set_xlabel("Yield (tons/hectare)")
    ax.set_ylabel("Frequency")
    path = output_dir / "yield_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], output_dir: Path = FIGURES_DIR) -> Path:
    _ensure_dir(output_dir)
    corr = df[numeric_cols + [TARGET_COLUMN]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    path = output_dir / "correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_yield_by_crop(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="crop", y=TARGET_COLUMN, hue="crop", palette="Set2", legend=False, ax=ax)
    ax.set_title("Yield by Crop Type")
    ax.set_xlabel("Crop")
    ax.set_ylabel("Yield (tons/hectare)")
    plt.xticks(rotation=20)
    path = output_dir / "yield_by_crop.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(
    feature_names: list[str],
    importances: list[float],
    model_name: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    _ensure_dir(output_dir)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=importance_df,
        y="feature",
        x="importance",
        hue="feature",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title(f"Top Feature Importances — {model_name}")
    path = output_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    _ensure_dir(output_dir)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color="#1565c0", edgecolors="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="#c62828", label="Ideal fit")
    ax.set_xlabel("Actual Yield (tons/hectare)")
    ax.set_ylabel("Predicted Yield (tons/hectare)")
    ax.set_title(f"{model_name}: Actual vs Predicted\nR² = {r2:.3f}, MAE = {mae:.3f}")
    ax.legend()
    path = output_dir / f"actual_vs_predicted_{model_name.lower().replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
