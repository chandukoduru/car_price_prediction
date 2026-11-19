"""Flask application for the used-car price estimator."""

from __future__ import annotations

import logging
import json
import math
import os
import pickle
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "car_data.csv"
MODEL_PATH = Path(
    os.environ.get("CAR_PRICE_MODEL_PATH", str(ROOT / "models" / "best_model.pkl"))
)
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
CURRENT_YEAR = datetime.now().year
ALLOWED_VALUES = {
    "Fuel_Type": {"Petrol", "Diesel", "CNG"},
    "Selling_type": {"Dealer", "Individual"},
    "Transmission": {"Manual", "Automatic"},
}
OWNER_VALUES = {0, 1, 2, 3}
# The source CSV also contains motorcycles; keep those out of the car picker.
MOTORCYCLE_PREFIXES = (
    "activa",
    "bajaj",
    "hero",
    "honda activa",
    "honda cb ",
    "honda cbr ",
    "honda dream",
    "honda karizma",
    "hyosung",
    "ktm",
    "mahindra mojo",
    "royal enfield",
    "suzuki access",
    "tvs",
    "um renegade",
    "yamaha",
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.context_processor
def inject_current_year():
    return {"current_year": CURRENT_YEAR}


def _available_car_models() -> list[str]:
    """Return unique car names from the project dataset, excluding motorcycles."""
    if not DATASET_PATH.is_file():
        logger.error("Car model list unavailable: dataset/car_data.csv is missing.")
        return []

    names = pd.read_csv(DATASET_PATH, usecols=["Car_Name"])["Car_Name"]
    models = {
        name.strip()
        for name in names.dropna().astype(str)
        if name.strip()
        and not name.strip().casefold().startswith(MOTORCYCLE_PREFIXES)
    }
    return sorted(models, key=str.casefold)


def _read_form() -> tuple[dict, list[str]]:
    """Validate form data and return pipeline-ready values and errors."""
    errors = []
    submitted = {
        "Car_Name": request.form.get("car_name", "").strip(),
        "Year": request.form.get("year", "").strip(),
        "Present_Price": request.form.get("present_price", "").strip(),
        "Driven_kms": request.form.get("kms_driven", "").strip(),
        "Fuel_Type": request.form.get("fuel_type", "").strip(),
        "Selling_type": request.form.get("seller_type", "").strip(),
        "Transmission": request.form.get("transmission", "").strip(),
        "Owner": request.form.get("owner", "").strip(),
    }

    labels = {
        "Car_Name": "Car model",
        "Year": "Manufacturing year",
        "Present_Price": "Present price",
        "Driven_kms": "Kilometers driven",
        "Fuel_Type": "Fuel type",
        "Selling_type": "Seller type",
        "Transmission": "Transmission",
        "Owner": "Previous owners",
    }
    for key, value in submitted.items():
        if not value:
            errors.append(f"{labels[key]} is required.")

    if errors:
        return submitted, errors

    if len(submitted["Car_Name"]) > 80:
        errors.append("Car model must be 80 characters or fewer.")
    elif submitted["Car_Name"].casefold() not in {
        model.casefold() for model in _available_car_models()
    }:
        errors.append("Select a car model from the available list.")

    try:
        year = int(submitted["Year"])
        if not 1900 <= year <= CURRENT_YEAR:
            errors.append(f"Manufacturing year must be between 1900 and {CURRENT_YEAR}.")
    except ValueError:
        year = 0
        errors.append("Manufacturing year must be a whole number.")

    try:
        present_price = float(submitted["Present_Price"])
        if not math.isfinite(present_price) or present_price <= 0:
            errors.append("Present price must be a positive number.")
    except ValueError:
        present_price = 0.0
        errors.append("Present price must be a valid number.")

    try:
        kms = float(submitted["Driven_kms"])
        if not math.isfinite(kms) or kms < 0:
            errors.append("Kilometers driven must be zero or greater.")
        elif not kms.is_integer():
            errors.append("Kilometers driven must be a whole number.")
        else:
            kms = int(kms)
    except ValueError:
        kms = 0.0
        errors.append("Kilometers driven must be a valid number.")

    try:
        owner = int(submitted["Owner"])
        if owner not in OWNER_VALUES:
            errors.append("Select a valid previous-owner count.")
    except ValueError:
        owner = -1
        errors.append("Select a valid previous-owner count.")

    for key, label in (
        ("Fuel_Type", "fuel type"),
        ("Selling_type", "seller type"),
        ("Transmission", "transmission"),
    ):
        if submitted[key] not in ALLOWED_VALUES[key]:
            errors.append(f"Select a supported {label}.")

    values = {
        "Car_Name": submitted["Car_Name"],
        "Year": year,
        "Present_Price": present_price,
        "Driven_kms": kms,
        "Fuel_Type": submitted["Fuel_Type"],
        "Selling_type": submitted["Selling_type"],
        "Transmission": submitted["Transmission"],
        "Owner": owner,
    }
    return values, errors


@app.get("/")
def home():
    gallery_path = ROOT / "static" / "images" / "car-gallery" / "gallery.json"
    if gallery_path.is_file():
        with gallery_path.open(encoding="utf-8") as gallery_file:
            car_gallery = json.load(gallery_file)
    else:
        car_gallery = []
    return render_template(
        "index.html",
        car_gallery=car_gallery,
        car_models=_available_car_models(),
    )


@app.get("/about")
def about():
    return render_template("about.html")


@app.post("/predict")
def predict():
    values, errors = _read_form()
    if errors:
        gallery_path = ROOT / "static" / "images" / "car-gallery" / "gallery.json"
        if gallery_path.is_file():
            with gallery_path.open(encoding="utf-8") as gallery_file:
                car_gallery = json.load(gallery_file)
        else:
            car_gallery = []
        return (
            render_template(
                "index.html",
                errors=errors,
                form=request.form,
                car_gallery=car_gallery,
                car_models=_available_car_models(),
            ),
            400,
        )

    if not MODEL_PATH.is_file():
        return (
            render_template(
                "error.html",
                message=(
                    "The prediction model is not available yet. Add the CarDekho "
                    "dataset and run the training instructions in the project README."
                ),
            ),
            503,
        )

    try:
        model = joblib.load(MODEL_PATH)
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        ImportError,
        AttributeError,
        ValueError,
        RuntimeError,
    ):
        logger.exception("Could not load the prediction model.")
        return (
            render_template(
                "error.html",
                message="The prediction model could not be loaded. Please contact the project administrator.",
            ),
            503,
        )

    try:
        model_values = {key: values[key] for key in FEATURES}
        model_values["Car_Name"] = values["Car_Name"].casefold()
        input_frame = pd.DataFrame([model_values])
        predicted_lakhs = float(model.predict(input_frame)[0])
        if not math.isfinite(predicted_lakhs) or predicted_lakhs < 0:
            raise ValueError("The model returned an invalid price.")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError):
        logger.exception("Prediction failed.")
        return (
            render_template(
                "error.html",
                message="We could not calculate an estimate right now. Please try again later.",
            ),
            500,
        )

    details = {
        "car_name": values["Car_Name"],
        "year": values["Year"],
        "present_price": values["Present_Price"],
        "kms": f'{values["Driven_kms"]:,.0f}',
        "fuel": values["Fuel_Type"],
        "seller": values["Selling_type"],
        "transmission": values["Transmission"],
        "owner": values["Owner"],
    }
    return render_template(
        "result.html",
        price_rupees=predicted_lakhs * 100_000,
        price_lakhs=predicted_lakhs,
        details=details,
    )


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
