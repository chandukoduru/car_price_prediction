from unittest.mock import Mock

import pytest

import app as car_app


@pytest.fixture
def client():
    car_app.app.config.update(TESTING=True)
    with car_app.app.test_client() as test_client:
        yield test_client


def valid_form(**overrides):
    values = {
        "car_name": "City",
        "year": "2018",
        "present_price": "8.5",
        "kms_driven": "42000",
        "fuel_type": "Petrol",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "0",
    }
    values.update(overrides)
    return values


def test_home_and_about_load(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b'Select a car model' in response.data
    assert b'value="city"' in response.data
    assert b'Royal Enfield' not in response.data
    assert client.get("/about").status_code == 200


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"year": ""}, "Manufacturing year is required."),
        ({"car_name": ""}, "Car model is required."),
        ({"car_name": "C" * 81}, "Car model must be 80 characters or fewer."),
        ({"car_name": "Unlisted Model"}, "Select a car model from the available list."),
        ({"year": "3020"}, "Manufacturing year must be between"),
        ({"kms_driven": "-1"}, "Kilometers driven must be zero or greater."),
        ({"kms_driven": "1.5"}, "Kilometers driven must be a whole number."),
        ({"present_price": "not-a-price"}, "Present price must be a valid number."),
        ({"present_price": "0"}, "Present price must be a positive number."),
        ({"fuel_type": "Electric"}, "Select a supported fuel type."),
        ({"owner": "9"}, "Select a valid previous-owner count."),
    ],
)
def test_invalid_form_is_rejected(client, overrides, message):
    response = client.post("/predict", data=valid_form(**overrides))
    assert response.status_code == 400
    assert message.encode() in response.data


def test_prediction_uses_pipeline_and_converts_lakhs_to_rupees(client, tmp_path, monkeypatch):
    model_path = tmp_path / "best_model.pkl"
    model_path.touch()
    monkeypatch.setattr(car_app, "MODEL_PATH", model_path)
    model = Mock()
    model.predict.return_value = [4.25]
    monkeypatch.setattr(car_app.joblib, "load", lambda _: model)

    response = client.post("/predict", data=valid_form())

    assert response.status_code == 200
    assert "₹425,000".encode() in response.data
    model.predict.assert_called_once()
    frame = model.predict.call_args.args[0]
    assert list(frame.columns) == car_app.FEATURES
    assert frame.iloc[0]["Car_Name"] == "city"
    assert frame.iloc[0]["Year"] == 2018


def test_missing_model_has_safe_message(client, tmp_path, monkeypatch):
    monkeypatch.setattr(car_app, "MODEL_PATH", tmp_path / "missing.pkl")
    response = client.post("/predict", data=valid_form())
    assert response.status_code == 503
    assert b"model is not available yet" in response.data
    assert str(tmp_path).encode() not in response.data


def test_model_load_failure_does_not_expose_internal_error(client, tmp_path, monkeypatch):
    model_path = tmp_path / "best_model.pkl"
    model_path.touch()
    monkeypatch.setattr(car_app, "MODEL_PATH", model_path)

    def fail_load(_):
        raise ValueError(f"private path: {tmp_path}")

    monkeypatch.setattr(car_app.joblib, "load", fail_load)
    response = client.post("/predict", data=valid_form())
    assert response.status_code == 503
    assert b"model could not be loaded" in response.data
    assert str(tmp_path).encode() not in response.data


def test_prediction_failure_does_not_expose_internal_error(client, tmp_path, monkeypatch):
    model_path = tmp_path / "best_model.pkl"
    model_path.touch()
    monkeypatch.setattr(car_app, "MODEL_PATH", model_path)
    model = Mock()
    model.predict.side_effect = RuntimeError(f"private path: {tmp_path}")
    monkeypatch.setattr(car_app.joblib, "load", lambda _: model)

    response = client.post("/predict", data=valid_form())

    assert response.status_code == 500
    assert b"could not calculate an estimate" in response.data
    assert str(tmp_path).encode() not in response.data
