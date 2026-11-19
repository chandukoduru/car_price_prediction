# Car Price Prediction Using Machine Learning

AutoValue is a B.Tech-level project that estimates a used car's resale price from its specifications. It demonstrates data validation, exploratory analysis, leakage-safe preprocessing, regression model comparison, and a responsive Flask web application.

> **Dataset:** The project includes the 301-row CarDekho vehicle dataset obtained from [Kaggle — Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), using a public mirror in [RimjimRazdan/cars_price_prediction](https://github.com/RimjimRazdan/cars_price_prediction). The mirror identifies Kaggle as the dataset source. Review the source terms before redistributing this project.

## Abstract

Used-car valuation depends on several interacting factors, making a consistent resale estimate difficult to produce manually. This project trains Linear Regression, Decision Tree Regressor, and Random Forest Regressor models on historical CarDekho listings. A Scikit-learn pipeline imputes and encodes input features without learning preprocessing from the held-out test set. The model with the lowest test-set RMSE is saved and served by Flask. Users submit vehicle details through a responsive form and receive an indicative price in Indian Rupees.

## Problem statement

Buyers and sellers need an accessible way to understand how features such as vehicle age, listed present price, mileage, fuel, seller type, transmission, and owner count relate to used-car prices. The application uses historical data to provide a reproducible estimate while clearly distinguishing a model output from a guaranteed market value.

## Objectives

- Validate and clean a CarDekho dataset before model training.
- Explore price patterns with visualizations.
- Compare three regression algorithms using MAE, MSE, RMSE, and R².
- Prevent preprocessing leakage with `Pipeline` and `ColumnTransformer`.
- Serve the selected model through a validated Flask prediction form.
- Keep the code modular and understandable for academic demonstration.

## Features

- Interactive car details form with browser-side and server-side validation.
- Car-model dropdown populated from the dataset's passenger-car names; motorcycle entries are excluded.
- Pipeline-backed price estimate and submitted-car summary.
- Safe, friendly responses when the trained model is unavailable.
- Training outputs for EDA, model performance, and feature importance.
- Responsive automotive-themed interface; no account or database required.

## Technology stack

| Area | Tools |
|---|---|
| Language / analysis | Python, Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualizations | Matplotlib, Seaborn |
| Backend / model storage | Flask, Joblib |
| Frontend | HTML5, CSS3, JavaScript, Bootstrap 5 |
| Automated tests | Pytest |

## Dataset description

The included dataset is at `dataset/car_data.csv`. The loader accepts the expected columns:

`Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission`, `Owner`

The source file calls `Driven_kms` `Kms_Driven` and `Selling_type` `Seller_Type`; the training loader normalizes those two header names. `Selling_Price` is the target and is excluded from all input features. `Car_Name` is included as a categorical input, so users can enter the car make/model on the prediction form. In this dataset, `Selling_Price` and `Present_Price` are in lakh rupees; the web application converts the predicted target from lakhs to rupees. The cleaner validates required columns, removes duplicate and unusable rows, normalizes categories, and rejects invalid target/feature ranges. Missing feature values are imputed inside the model pipeline.

## System architecture

1. `train_model.py` loads and validates the local CSV.
2. The data is split into 80% training and 20% test partitions.
3. Each estimator receives its own preprocessing pipeline: median imputation for numeric fields, standard scaling for Linear Regression, and one-hot encoding for categorical fields.
4. Models are evaluated on the held-out test set. The lowest-RMSE complete pipeline is serialized as `models/best_model.pkl`.
5. Test metrics and charts are written to `models/` and `static/images/`.
6. `app.py` validates submitted values, builds a one-row DataFrame in training-feature order, calls the saved pipeline, and displays the estimate.

## Machine learning algorithms

- **Linear Regression:** interpretable baseline; numeric values are standardized.
- **Decision Tree Regressor:** models nonlinear splits and interactions.
- **Random Forest Regressor:** averages an ensemble of decision trees.

All models use the same fixed random train-test split. MAE, MSE, RMSE, and R² are computed from test predictions; no metric is hardcoded. Selection is based on the lowest test RMSE. The saved model is the full preprocessing and prediction pipeline, not just its estimator.

## Car image gallery

The app can display a small gallery built from the [Kaggle Car Images Dataset](https://www.kaggle.com/datasets/kshitij192/cars-image-dataset) by Kshitij Kumar. Kaggle lists the dataset license as GPL-2.0 and requests attribution. Downloading and preparing one representative image per class is optional:

```powershell
python download_car_images.py
```

The script uses KaggleHub to download the dataset into its cache and copies only the gallery samples into `static/images/car-gallery/`. The attribution, source-file list, and GPL-2.0 license text are saved alongside the images. Review the dataset license before redistribution.

## Installation

Use Python 3.10 or later. From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The project includes the CarDekho CSV at `dataset/car_data.csv`; review its source and terms before redistribution.

## Execution

```powershell
python train_model.py
python app.py
```

Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/). The training step creates `models/best_model.pkl`, `models/model_metrics.csv`, and the visualizations in `static/images/`. If the model is missing, the form responds with a setup message rather than an invented prediction.

## Testing

```powershell
python -m pytest
```

Tests cover page startup, successful prediction and lakh-to-rupee conversion, missing fields, invalid year/price/mileage/categories, missing model, and safe handling of prediction failures.

## Data visualizations

Training generates:

- Selling-price distribution
- Fuel type versus selling price
- Transmission versus selling price
- Manufacturing year versus selling price
- Kilometers driven versus selling price
- Test RMSE model comparison
- Feature importance or absolute linear coefficients for the selected model

## Screenshots

Add screenshots of the home page, prediction form, and result page here after running the application. Charts are generated in `static/images/` after a successful training run.

## Future enhancements

- Add a sourced, versioned dataset and a documented model retraining workflow.
- Evaluate cross-validation and temporal or regional generalization.
- Add confidence intervals and more market-condition features.
- Add optional SQLite history with privacy controls.
- Add accessible chart/report pages and deployment configuration.

## Conclusion

AutoValue provides an end-to-end educational example of a used-car price regression project. It combines transparent dataset handling and test-set comparison with a real saved Scikit-learn pipeline and a usable Flask interface. Its output is an estimate based on historical data, not a substitute for inspection or a live market appraisal.
