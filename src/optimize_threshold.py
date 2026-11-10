
# ============================================================
# CUSTOMER CHURN PREDICTION
# EXPERIMENT 5 - DECISION THRESHOLD OPTIMIZATION
#
# Models:
# 1. Tuned Logistic Regression
# 2. Tuned Random Forest
#
# Method: Out-of-fold predicted probabilities
# Selection: F1 score
# ============================================================

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    precision_recall_curve
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

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


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
# 4. TRAINING DATA
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Do not use X_test or y_test for threshold selection.


# ------------------------------------------------------------
# 5. PREPROCESSING FUNCTION
# ------------------------------------------------------------

def create_preprocessor(scale_numeric):

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

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

    return ColumnTransformer([
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
# 6. RECREATE TUNED LOGISTIC REGRESSION
# ------------------------------------------------------------

logistic_pipeline = Pipeline([
    (
        "preprocessor",
        create_preprocessor(scale_numeric=True)
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
# 7. RECREATE TUNED RANDOM FOREST
# ------------------------------------------------------------

forest_pipeline = Pipeline([
    (
        "preprocessor",
        create_preprocessor(scale_numeric=False)
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        )
    )
])


# ------------------------------------------------------------
# 8. DEFINE MODELS
# ------------------------------------------------------------

models = {
    "Logistic Regression": logistic_pipeline,
    "Random Forest": forest_pipeline
}


# ------------------------------------------------------------
# 9. CROSS-VALIDATION
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 10. THRESHOLD ANALYSIS FUNCTION
# ------------------------------------------------------------

def optimize_threshold(model_name, model):

    print("\n" + "=" * 65)
    print(model_name)
    print("=" * 65)

    # Generate out-of-fold probabilities.
    # Each observation is predicted by a model
    # that did not train on that observation.

    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    # Calculate precision and recall at all thresholds.

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_train,
            oof_probabilities
        )
    )

    # Precision and recall have one extra final element
    # compared with the threshold array.

    f1_values = (
        2 * precision_values[:-1] * recall_values[:-1]
        /
        (
            precision_values[:-1]
            + recall_values[:-1]
            + 1e-12
        )
    )

    # Identify the threshold with maximum F1.

    best_index = np.argmax(f1_values)

    best_threshold = thresholds[best_index]

    best_precision = precision_values[best_index]

    best_recall = recall_values[best_index]

    best_f1 = f1_values[best_index]

    print("\nBest threshold:", round(float(best_threshold), 4))

    print("Best F1:", round(float(best_f1), 4))

    print(
        "Precision:",
        round(float(best_precision), 4)
    )

    print(
        "Recall:",
        round(float(best_recall), 4)
    )

    # --------------------------------------------------------
    # Compare selected thresholds
    # --------------------------------------------------------

    candidate_thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        float(best_threshold)
    ]

    rows = []

    for threshold in sorted(set(candidate_thresholds)):

        predictions = (
            oof_probabilities >= threshold
        ).astype(int)

        rows.append({
            "Model": model_name,
            "Threshold": threshold,

            "Precision": precision_score(
                y_train,
                predictions,
                zero_division=0
            ),

            "Recall": recall_score(
                y_train,
                predictions,
                zero_division=0
            ),

            "F1": f1_score(
                y_train,
                predictions,
                zero_division=0
            ),

            "Balanced Accuracy": balanced_accuracy_score(
                y_train,
                predictions
            )
        })

    threshold_results = pd.DataFrame(rows)

    print("\nThreshold comparison:")

    print(
        threshold_results
        .round(4)
        .to_string(index=False)
    )

    return threshold_results


# ------------------------------------------------------------
# 11. RUN BOTH MODELS
# ------------------------------------------------------------

all_results = []

for model_name, model in models.items():

    results = optimize_threshold(
        model_name,
        model
    )

    all_results.append(results)


# ------------------------------------------------------------
# 12. COMBINE RESULTS
# ------------------------------------------------------------

final_results = pd.concat(
    all_results,
    ignore_index=True
)


# ------------------------------------------------------------
# 13. SAVE RESULTS
# ------------------------------------------------------------

OUTPUT_PATH = REPORT_DIR / "threshold_comparison.csv"

final_results.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 65)

print("Threshold analysis completed.")

print("Results saved to:", OUTPUT_PATH)
