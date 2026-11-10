
# ============================================================
# CUSTOMER CHURN PREDICTION
# EXPERIMENT 4 - HYPERPARAMETER TUNING
#
# Models:
# 1. Logistic Regression
# 2. Random Forest
#
# Optimization: GridSearchCV
# Validation: 5-fold Stratified Cross-Validation
# Selection metric: F1
# ============================================================

import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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

if y.isna().any():
    raise ValueError("Unexpected values in Churn column.")


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
# 4. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# The test set is deliberately not used in tuning.


# ------------------------------------------------------------
# 5. CREATE PREPROCESSING FUNCTION
# ------------------------------------------------------------

def create_preprocessor(scale_numeric):

    numerical_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]

    if scale_numeric:
        numerical_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numerical_steps)

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
            numeric_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    return preprocessor


# ------------------------------------------------------------
# 6. CREATE LOGISTIC REGRESSION PIPELINE
# ------------------------------------------------------------

logistic_pipeline = Pipeline([
    (
        "preprocessor",
        create_preprocessor(scale_numeric=True)
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
# 7. LOGISTIC REGRESSION SEARCH SPACE
# ------------------------------------------------------------

logistic_param_grid = {
    "classifier__C": [
        0.01,
        0.1,
        1.0,
        10.0,
        100.0
    ],

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}


# ------------------------------------------------------------
# 8. CREATE RANDOM FOREST PIPELINE
# ------------------------------------------------------------

forest_pipeline = Pipeline([
    (
        "preprocessor",
        create_preprocessor(scale_numeric=False)
    ),
    (
        "classifier",
        RandomForestClassifier(
            random_state=42,
            n_jobs=1
        )
    )
])


# ------------------------------------------------------------
# 9. RANDOM FOREST SEARCH SPACE
# ------------------------------------------------------------

forest_param_grid = {
    "classifier__n_estimators": [
        200,
        400
    ],

    "classifier__max_depth": [
        None,
        12
    ],

    "classifier__min_samples_leaf": [
        1,
        3
    ],

    "classifier__class_weight": [
        None,
        "balanced"
    ]
}


# ------------------------------------------------------------
# 10. CROSS-VALIDATION SETTINGS
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
# 11. GRID SEARCH FUNCTION
# ------------------------------------------------------------

def run_grid_search(model_name, pipeline, param_grid):

    print("\n" + "=" * 65)
    print(f"TUNING: {model_name}")
    print("=" * 65)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
        verbose=1
    )

    search.fit(X_train, y_train)

    print("\nBest hyperparameters:")
    print(search.best_params_)

    print("\nBest cross-validation F1:")
    print(search.best_score_)

    results = pd.DataFrame(
        search.cv_results_
    )

    results = results.sort_values(
        by="mean_test_f1",
        ascending=False
    )

    print("\nTop 5 configurations:")

    display_columns = [
        "params",
        "mean_test_f1",
        "std_test_f1",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_roc_auc",
        "mean_test_average_precision"
    ]

    print(
        results[display_columns]
        .head(5)
        .to_string(index=False)
    )

    # Save all search results
    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_tuning_results.csv"
    )

    results.to_csv(
        REPORT_DIR / filename,
        index=False
    )

    print("\nResults saved:", filename)

    return search


# ------------------------------------------------------------
# 12. TUNE LOGISTIC REGRESSION
# ------------------------------------------------------------

logistic_search = run_grid_search(
    "Logistic Regression",
    logistic_pipeline,
    logistic_param_grid
)


# ------------------------------------------------------------
# 13. TUNE RANDOM FOREST
# ------------------------------------------------------------

forest_search = run_grid_search(
    "Random Forest",
    forest_pipeline,
    forest_param_grid
)


# ------------------------------------------------------------
# 14. COMPARE BEST CONFIGURATIONS
# ------------------------------------------------------------

searches = {
    "Logistic Regression": logistic_search,
    "Random Forest": forest_search
}

summary = []

for model_name, search in searches.items():

    index = search.best_index_

    result = {
        "Model": model_name,
        "Best Parameters": str(search.best_params_)
    }

    for metric in scoring:

        result[metric] = search.cv_results_[
            f"mean_test_{metric}"
        ][index]

    result["F1 Standard Deviation"] = search.cv_results_[
        "std_test_f1"
    ][index]

    summary.append(result)


comparison = pd.DataFrame(summary)

comparison = comparison.sort_values(
    by="f1",
    ascending=False
)

print("\n" + "=" * 65)
print("TUNED MODEL COMPARISON")
print("=" * 65)

print(comparison.to_string(index=False))


# ------------------------------------------------------------
# 15. SAVE COMPARISON
# ------------------------------------------------------------

comparison.to_csv(
    REPORT_DIR / "tuned_model_comparison.csv",
    index=False
)

print("\nTuned comparison saved successfully.")

print("\nHyperparameter tuning completed.")
