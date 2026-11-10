
# ============================================================
# CUSTOMER CHURN PREDICTION
# REUSABLE ML PREDICTION SERVICE
# ============================================================

import json
import joblib
import pandas as pd

from pathlib import Path


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "customer_churn_final_pipeline.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_model_metadata.json"
)


# ------------------------------------------------------------
# 2. CACHED MODEL ARTIFACTS
# ------------------------------------------------------------

_model = None
_metadata = None


def load_model():

    global _model, _metadata

    # Load only once per Python process.
    if _model is None or _metadata is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found: {METADATA_PATH}"
            )

        loaded_model = joblib.load(MODEL_PATH)

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            loaded_metadata = json.load(file)

        _model = loaded_model
        _metadata = loaded_metadata

    return _model, _metadata


# ------------------------------------------------------------
# 3. PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_customer(customer_data):

    if not isinstance(customer_data, dict):
        raise TypeError(
            "Customer data must be a dictionary."
        )

    model, metadata = load_model()

    expected_features = (
        metadata["numerical_features"]
        + metadata["categorical_features"]
    )

    # Check missing features
    missing_features = (
        set(expected_features) - set(customer_data)
    )

    # Check unexpected features
    extra_features = (
        set(customer_data) - set(expected_features)
    )

    if missing_features:
        raise ValueError(
            f"Missing features: {sorted(missing_features)}"
        )

    if extra_features:
        raise ValueError(
            f"Unexpected features: {sorted(extra_features)}"
        )

    # Create DataFrame with expected feature order
    customer_df = pd.DataFrame(
        [customer_data],
        columns=expected_features
    )

    # Get model score for churn class
    churn_score = float(
        model.predict_proba(customer_df)[0, 1]
    )

    threshold = float(
        metadata["decision_threshold"]
    )

    # Apply selected threshold
    churn_prediction = int(
        churn_score >= threshold
    )

    return {
        "churn_prediction": churn_prediction,

        "churn_label": (
            "Yes" if churn_prediction == 1 else "No"
        ),

        "churn_score": round(
            churn_score,
            4
        ),

        "decision_threshold": threshold,

        "score_note": (
            "Uncalibrated model score, "
            "not a calibrated real-world probability."
        )
    }


# ------------------------------------------------------------
# 4. LOCAL TEST
# ------------------------------------------------------------

if __name__ == "__main__":

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.50,
        "TotalCharges": 1074.00
    }

    result = predict_customer(sample_customer)

    print("\nCUSTOMER CHURN PREDICTION")

    for key, value in result.items():
        print(f"{key}: {value}")
