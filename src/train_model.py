
# ============================================================
# CUSTOMER CHURN PREDICTION
# STEP 4 - BASELINE MODEL TRAINING
#
# Model: Logistic Regression
# Preprocessing: StandardScaler + OneHotEncoder
# Validation: Stratified 5-fold Cross-Validation
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.linear_model import LogisticRegression

from sklearn.dummy import DummyClassifier

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
# 2. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "logistic_regression_baseline.joblib"

METADATA_PATH = MODEL_DIR / "model_metadata.json"


# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------

data = pd.read_csv(DATA_PATH)

print("Dataset loaded:", data.shape)


# ------------------------------------------------------------
# 4. DEFINE FEATURES AND TARGET
# ------------------------------------------------------------

# Remove customerID because it is an identifier,
# not a meaningful predictive feature.

X = data.drop(
    columns=["customerID", "Churn"]
)

# Convert target:
# No  -> 0
# Yes -> 1

y = data["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected target values found.")


# ------------------------------------------------------------
# 5. IDENTIFY FEATURE TYPES
# ------------------------------------------------------------

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ------------------------------------------------------------
# 6. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ------------------------------------------------------------
# 7. NUMERICAL PREPROCESSING
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
# 8. CATEGORICAL PREPROCESSING
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
# 9. COMBINE USING COLUMNTRANSFORMER
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
# 10. BUILD COMPLETE MODEL PIPELINE
# ------------------------------------------------------------

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])


# ------------------------------------------------------------
# 11. CROSS-VALIDATION
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision"
}

print("\n========== CROSS-VALIDATION ==========")

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

for metric in scoring:

    scores = cv_results[f"test_{metric}"]

    print(
        f"{metric}: "
        f"{scores.mean():.4f} "
        f"+/- {scores.std():.4f}"
    )


# ------------------------------------------------------------
# 12. TRAIN BASELINE MODEL
# ------------------------------------------------------------

print("\n========== MODEL TRAINING ==========")

pipeline.fit(
    X_train,
    y_train
)

print("Logistic Regression training completed.")


# ------------------------------------------------------------
# 13. TRAIN A DUMMY BASELINE
# ------------------------------------------------------------

dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(
    X_train,
    y_train
)

print("Dummy baseline training completed.")


# ------------------------------------------------------------
# 14. EVALUATION FUNCTION
# ------------------------------------------------------------

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),

        "Balanced Accuracy": balanced_accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        ),

        "Average Precision": average_precision_score(
            y_test,
            probabilities
        )
    }

    for metric, value in metrics.items():

        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Stayed", "Churned"],
            zero_division=0
        )
    )

    return metrics


# ------------------------------------------------------------
# 15. EVALUATE DUMMY MODEL
# ------------------------------------------------------------

dummy_metrics = evaluate_model(
    "DUMMY CLASSIFIER",
    dummy_model,
    X_test,
    y_test
)


# ------------------------------------------------------------
# 16. EVALUATE LOGISTIC REGRESSION
# ------------------------------------------------------------

model_metrics = evaluate_model(
    "LOGISTIC REGRESSION",
    pipeline,
    X_test,
    y_test
)


# ------------------------------------------------------------
# 17. COMPARE MODELS
# ------------------------------------------------------------

comparison = pd.DataFrame([
    {
        "Model": "Dummy Classifier",
        **dummy_metrics
    },
    {
        "Model": "Logistic Regression",
        **model_metrics
    }
])

print("\n========== MODEL COMPARISON ==========")

print(
    comparison.to_string(index=False)
)


# ------------------------------------------------------------
# 18. SAVE FITTED PIPELINE
# ------------------------------------------------------------

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nSaved fitted pipeline:")
print(MODEL_PATH)


# ------------------------------------------------------------
# 19. SAVE METADATA
# ------------------------------------------------------------

metadata = {
    "model_name": "Logistic Regression Baseline",
    "target": "Churn",
    "target_mapping": {
        "No": 0,
        "Yes": 1
    },
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "all_features": X.columns.tolist(),
    "test_metrics": model_metrics,
    "cross_validation_folds": 5,
    "random_state": 42
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:

    json.dump(
        metadata,
        file,
        indent=4
    )

print("\nSaved metadata:")
print(METADATA_PATH)


# ------------------------------------------------------------
# 20. VERIFY SAVED MODEL
# ------------------------------------------------------------

loaded_pipeline = joblib.load(MODEL_PATH)

sample_customer = X_test.iloc[[0]]

sample_prediction = loaded_pipeline.predict(
    sample_customer
)

sample_probability = loaded_pipeline.predict_proba(
    sample_customer
)[0, 1]

print("\n========== SAVED MODEL TEST ==========")

print("Predicted class:", sample_prediction[0])

print("Churn probability:", round(sample_probability, 4))

print("\nBaseline training workflow completed!")
