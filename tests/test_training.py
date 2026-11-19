import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from train_model import (
    FEATURES,
    REQUIRED_COLUMNS,
    build_pipeline,
    load_and_clean_dataset,
)


def test_missing_dataset_reports_expected_location(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset/car_data.csv"):
        load_and_clean_dataset(tmp_path / "car_data.csv")


def test_dataset_columns_are_validated(tmp_path):
    path = tmp_path / "car_data.csv"
    pd.DataFrame({"Car_Name": ["Car"], "Year": [2020]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_and_clean_dataset(path)


def test_cleaning_preserves_cng_and_leaves_missing_categories_for_pipeline(tmp_path):
    path = tmp_path / "car_data.csv"
    rows = [
        ["Car A", 2020, 8.0, 10000, "Petrol", "Dealer", "Manual", 0, 5.0],
        ["Car B", 2019, 7.0, 20000, "CNG", "Individual", "Manual", 1, 4.0],
        ["Car C", 2018, 6.0, 30000, None, "Dealer", "Automatic", 0, 3.0],
        ["Car D", 2017, 5.0, 40000, "Diesel", "Dealer", "Manual", 2, 2.0],
        ["Car E", 2016, None, 50000, "Petrol", "Individual", "Manual", 0, 1.0],
    ]
    pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_csv(path, index=False)

    cleaned = load_and_clean_dataset(path)

    assert cleaned.loc[1, "Fuel_Type"] == "CNG"
    assert cleaned.loc[0, "Car_Name"] == "car a"
    assert pd.isna(cleaned.loc[2, "Fuel_Type"])
    assert pd.isna(cleaned.loc[4, "Present_Price"])
    assert "Selling_Price" not in FEATURES

    pipeline = build_pipeline(LinearRegression(), scale_numeric=True)
    pipeline.fit(cleaned[FEATURES], cleaned["Selling_Price"])
    assert len(pipeline.predict(cleaned[FEATURES].iloc[[0]])) == 1
