
# ============================================================
# CUSTOMER CHURN PREDICTION
# EXPERIMENT 6 - FINAL MODEL EVALUATION
#
# Selected model: Balanced Logistic Regression
# C: 0.1
# Decision threshold: 0.547
# ============================================================

import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    MODEL_DIR / "customer_churn_final_pipeline.joblib"
)

METADATA_PATH = (
    MODEL_DIR / "final_model_metadata.json"
)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

data = pd.read_csv(DATA_PATH)

X = data.drop(
    columns=["customerID", "Churn"]
)

y = data["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Invalid churn target values.")


# ------------------------------------------------------------
# 3. IDENTIFY FEATURE TYPES
# ------------------------------------------------------------

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


# ------------------------------------------------------------
# 4. RECREATE THE ORIGINAL TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ------------------------------------------------------------
# 5. NUMERICAL PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# ------------------------------------------------------------
# 6. CATEGORICAL PREPROCESSING
# ------------------------------------------------------------

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# ------------------------------------------------------------
# 7. COLUMN TRANSFORMER
# ------------------------------------------------------------

preprocessor = ColumnTransformer([
    (
        "numerical",
        numeric_pipeline,
        numerical_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ------------------------------------------------------------
# 8. FINAL SELECTED PIPELINE
# ------------------------------------------------------------

final_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    )
])


# ------------------------------------------------------------
# 9. TRAIN ON THE COMPLETE TRAINING SET
# ------------------------------------------------------------

print("\nTraining final model...")

final_model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ------------------------------------------------------------
# 10. PREDICT TEST PROBABILITIES
# ------------------------------------------------------------

test_probabilities = final_model.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------
# 11. APPLY SELECTED THRESHOLD
# ------------------------------------------------------------

DECISION_THRESHOLD = 0.547

test_predictions = (
    test_probabilities >= DECISION_THRESHOLD
).astype(int)


# ------------------------------------------------------------
# 12. CALCULATE FINAL METRICS
# ------------------------------------------------------------

metrics = {
    "accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),

    "balanced_accuracy": float(
        balanced_accuracy_score(y_test, test_predictions)
    ),

    "precision": float(
        precision_score(
            y_test,
            test_predictions,
            zero_division=0
        )
    ),

    "recall": float(
        recall_score(
            y_test,
            test_predictions,
            zero_division=0
        )
    ),

    "f1": float(
        f1_score(
            y_test,
            test_predictions,
            zero_division=0
        )
    ),

    "roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    ),

    "average_precision": float(
        average_precision_score(
            y_test,
            test_probabilities
        )
    )
}


# ------------------------------------------------------------
# 13. DISPLAY FINAL RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

for name, value in metrics.items():

    print(f"{name}: {value:.4f}")


# ------------------------------------------------------------
# 14. CONFUSION MATRIX
# ------------------------------------------------------------

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )
)


# ------------------------------------------------------------
# 15. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Stayed", "Churned"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 16. SAVE THE FINAL PIPELINE
# ------------------------------------------------------------

joblib.dump(
    final_model,
    MODEL_PATH
)

print("\nFinal model saved:")
print(MODEL_PATH)


# ------------------------------------------------------------
# 17. SAVE MODEL METADATA
# ------------------------------------------------------------

metadata = {
    "project": "Customer Churn Prediction",
    "model": "Logistic Regression",
    "class_weight": "balanced",
    "C": 0.1,
    "decision_threshold": DECISION_THRESHOLD,
    "positive_class": 1,
    "positive_class_meaning": "Customer churn",
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "evaluation_metrics": metrics,
    "dataset": "IBM Telco Customer Churn",
    "random_state": 42
}

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )

print("\nMetadata saved:")
print(METADATA_PATH)


# ------------------------------------------------------------
# 18. SAVE TEST PREDICTIONS FOR ANALYSIS
# ------------------------------------------------------------

prediction_results = pd.DataFrame({
    "actual_churn": y_test.values,
    "predicted_churn": test_predictions,
    "churn_probability": test_probabilities
})

prediction_results.to_csv(
    REPORT_DIR / "final_test_predictions.csv",
    index=False
)

print("\nTest predictions saved successfully.")

print("\nFinal evaluation completed.")
