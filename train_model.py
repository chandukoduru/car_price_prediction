"""Train and compare used-car price regression pipelines.

Put the licensed CarDekho CSV at dataset/car_data.csv before running this file.
Selling_Price and Present_Price in the standard CarDekho dataset are expressed
in lakh rupees; the web application converts model predictions to rupees.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "car_data.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
IMAGE_DIR = ROOT / "static" / "images"
METRICS_PATH = ROOT / "models" / "model_metrics.csv"

FEATURES = [
    "Car_Name",
    "Year",
    "Present_Price",
    "Driven_kms",
    "Fuel_Type",
    "Selling_type",
    "Transmission",
    "Owner",
]
TARGET = "Selling_Price"
REQUIRED_COLUMNS = [*FEATURES, TARGET]
NUMERIC_FEATURES = ["Year", "Present_Price", "Driven_kms", "Owner"]
CATEGORICAL_FEATURES = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]
ALLOWED_CATEGORIES = {
    "Fuel_Type": {"Petrol", "Diesel", "CNG"},
    "Selling_type": {"Dealer", "Individual"},
    "Transmission": {"Manual", "Automatic"},
}
LOG = logging.getLogger("car_price_training")


def load_and_clean_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load, validate, and clean the expected CarDekho CSV."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Add the CarDekho CSV as "
            "'dataset/car_data.csv' and run training again."
        )

    data = pd.read_csv(path)
    column_aliases = {
        "Kms_Driven": "Driven_kms",
        "Seller_Type": "Selling_type",
    }
    for source, target in column_aliases.items():
        if source in data.columns and target in data.columns:
            raise ValueError(
                f"Dataset contains both '{source}' and '{target}'; "
                "remove the duplicate equivalent column."
            )
    data = data.rename(columns=column_aliases)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )
    if data.empty:
        raise ValueError("Dataset contains no rows.")

    data = data.loc[:, REQUIRED_COLUMNS].copy()
    data = data.drop_duplicates().copy()

    for column in NUMERIC_FEATURES + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        normalized = (
            data[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .str.casefold()
        )
        if column in ALLOWED_CATEGORIES:
            category_names = {
                category.casefold(): category
                for category in ALLOWED_CATEGORIES[column]
            }
            normalized = normalized.map(category_names)
        data[column] = normalized.astype(object)

    # Invalid targets cannot be imputed; invalid feature values are imputed
    # inside each training fold by the fitted preprocessing pipeline.
    data = data.dropna(subset=[TARGET])
    data = data[np.isfinite(data[TARGET]) & (data[TARGET] >= 0)]

    feature_ranges = {
        "Year": (1900, 2100),
        "Present_Price": (0, float("inf")),
        "Driven_kms": (0, float("inf")),
        "Owner": (0, 3),
    }
    for column, (minimum, maximum) in feature_ranges.items():
        valid = data[column].isna() | (
            np.isfinite(data[column])
            & data[column].between(minimum, maximum)
            & ((data[column] > 0) if column == "Present_Price" else True)
        )
        data = data[valid]

    for column, allowed in ALLOWED_CATEGORIES.items():
        data[column] = data[column].where(data[column].isin(allowed), np.nan)
        if data[column].notna().sum() == 0:
            raise ValueError(
                f"Dataset column '{column}' has no supported category values."
            )

    if len(data) < 5:
        raise ValueError(
            "At least five valid rows are required after cleaning; "
            f"found {len(data)}."
        )
    empty_numeric = [column for column in NUMERIC_FEATURES if data[column].notna().sum() == 0]
    if empty_numeric:
        raise ValueError(
            "Dataset has no usable values for numeric features: "
            + ", ".join(empty_numeric)
        )
    return data.reset_index(drop=True)


def build_pipeline(estimator, scale_numeric: bool = False) -> Pipeline:
    """Build preprocessing and regression as a single leakage-safe pipeline."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("regressor", estimator)]
    )


def save_eda_charts(data: pd.DataFrame, output_dir: Path = IMAGE_DIR) -> None:
    """Save dataset exploration charts used by the report and website."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    charts = [
        ("price_distribution.png", "Selling price distribution", "Selling_Price", None),
        ("fuel_price.png", "Selling price by fuel type", "Fuel_Type", "Selling_Price"),
        (
            "transmission_price.png",
            "Selling price by transmission",
            "Transmission",
            "Selling_Price",
        ),
    ]
    for filename, title, x, y in charts:
        fig, ax = plt.subplots(figsize=(8, 5))
        if y is None:
            sns.histplot(data=data, x=x, kde=True, ax=ax)
            ax.set_xlabel("Selling price (lakh INR)")
            ax.set_ylabel("Number of cars")
        else:
            sns.boxplot(data=data, x=x, y=y, ax=ax)
            ax.set_ylabel("Selling price (lakh INR)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    scatter_charts = [
        ("year_price.png", "Manufacturing year vs selling price", "Year"),
        ("kms_price.png", "Kilometers driven vs selling price", "Driven_kms"),
    ]
    for filename, title, x in scatter_charts:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=data, x=x, y="Selling_Price", alpha=0.65, ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Selling price (lakh INR)")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)


def save_model_charts(
    results: pd.DataFrame, best_pipeline: Pipeline, output_dir: Path = IMAGE_DIR
) -> None:
    """Save model comparison and best-model feature importance charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=results, x="Model", y="RMSE", hue="Model", legend=False, ax=ax)
    ax.set_title("Test-set RMSE by model (lower is better)")
    ax.set_ylabel("RMSE (lakh INR)")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    estimator = best_pipeline.named_steps["regressor"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.ravel(estimator.coef_))
    else:
        return

    names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance_data = (
        pd.DataFrame({"Feature": names, "Importance": importances})
        .nlargest(15, "Importance")
        .sort_values("Importance")
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=importance_data, x="Importance", y="Feature", hue="Feature",
        legend=False, ax=ax
    )
    ax.set_title("Most influential model features")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


def train_models(data: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline]:
    """Train all requested regressors and select the lowest test RMSE."""
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.2,
        random_state=42,
    )

    candidates = {
        "Linear Regression": build_pipeline(LinearRegression(), scale_numeric=True),
        "Decision Tree": build_pipeline(
            DecisionTreeRegressor(random_state=42, min_samples_leaf=2)
        ),
        "Random Forest": build_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        ),
    }

    rows = []
    fitted = {}
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, predictions),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R2": r2_score(y_test, predictions),
            }
        )
        fitted[name] = pipeline
        LOG.info("%s: RMSE=%.4f lakh INR", name, np.sqrt(mse))

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return results, fitted[results.iloc[0]["Model"]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data = load_and_clean_dataset()
    LOG.info("Using %s valid unique rows.", len(data))

    save_eda_charts(data)
    results, best_pipeline = train_models(data)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    results.to_csv(METRICS_PATH, index=False)
    save_model_charts(results, best_pipeline)

    LOG.info("Best model: %s", results.iloc[0]["Model"])
    LOG.info("Saved model pipeline to %s", MODEL_PATH)
    LOG.info("Saved test metrics to %s", METRICS_PATH)
    LOG.info("\n%s", results.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        raise SystemExit(f"Training could not start: {error}") from error
