
# ============================================================
# CUSTOMER CHURN PREDICTION
# EXPERIMENT 2 - CLASS WEIGHT COMPARISON
# ============================================================

import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)

data = pd.read_csv(DATA_PATH)

X = data.drop(
    columns=["customerID", "Churn"]
)

y = data["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ------------------------------------------------------------
# 2. IDENTIFY FEATURE TYPES
# ------------------------------------------------------------

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


# ------------------------------------------------------------
# 3. TRAINING SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# We will not use X_test or y_test to compare
# these experiments. Model selection uses CV only.


# ------------------------------------------------------------
# 4. CREATE A PIPELINE FUNCTION
# ------------------------------------------------------------

def create_pipeline(class_weight):

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

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

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight=class_weight,
                max_iter=2000,
                random_state=42
            )
        )
    ])

    return pipeline


# ------------------------------------------------------------
# 5. DEFINE BOTH EXPERIMENTS
# ------------------------------------------------------------

models = {
    "Unweighted": create_pipeline(None),
    "Balanced": create_pipeline("balanced")
}


# ------------------------------------------------------------
# 6. CROSS-VALIDATION SETTINGS
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


# ------------------------------------------------------------
# 7. COMPARE BOTH MODELS
# ------------------------------------------------------------

all_results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    row = {"Model": model_name}

    for metric in scoring:

        scores = results[f"test_{metric}"]

        mean_score = scores.mean()
        std_score = scores.std()

        row[metric] = mean_score

        print(
            f"{metric}: "
            f"{mean_score:.4f} "
            f"+/- {std_score:.4f}"
        )

    all_results.append(row)


# ------------------------------------------------------------
# 8. DISPLAY COMPARISON
# ------------------------------------------------------------

comparison = pd.DataFrame(all_results)

print("\n========== FINAL CV COMPARISON ==========")

print(
    comparison.to_string(index=False)
)

print("\nExperiment completed successfully.")
