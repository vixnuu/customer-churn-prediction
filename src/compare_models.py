
# ============================================================
# CUSTOMER CHURN PREDICTION
# EXPERIMENT 3 - ALGORITHM COMPARISON
#
# Models:
# 1. Logistic Regression - Unweighted
# 2. Logistic Regression - Balanced
# 3. Random Forest - Unweighted
# 4. Random Forest - Balanced
#
# Evaluation: 5-Fold Stratified Cross-Validation
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
from sklearn.ensemble import RandomForestClassifier


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
# 3. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# IMPORTANT:
# We use X_train and y_train for model comparison.
# X_test and y_test remain untouched.


# ------------------------------------------------------------
# 4. CREATE PREPROCESSING FOR LOGISTIC REGRESSION
# ------------------------------------------------------------

numeric_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_lr = ColumnTransformer([
    ("numerical", numeric_lr, numerical_features),
    ("categorical", categorical_lr, categorical_features)
])


# ------------------------------------------------------------
# 5. CREATE PREPROCESSING FOR RANDOM FOREST
# ------------------------------------------------------------

# Tree-based models don't normally need scaling.

numeric_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_rf = ColumnTransformer([
    ("numerical", numeric_rf, numerical_features),
    ("categorical", categorical_rf, categorical_features)
])


# ------------------------------------------------------------
# 6. CREATE FOUR COMPLETE PIPELINES
# ------------------------------------------------------------

models = {

    "Logistic Regression - Unweighted": Pipeline([
        ("preprocessor", preprocessor_lr),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Logistic Regression - Balanced": Pipeline([
        ("preprocessor", preprocessor_lr),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Random Forest - Unweighted": Pipeline([
        ("preprocessor", preprocessor_rf),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Random Forest - Balanced": Pipeline([
        ("preprocessor", preprocessor_rf),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

}


# ------------------------------------------------------------
# 7. CROSS-VALIDATION SETTINGS
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
# 8. EVALUATE EACH MODEL
# ------------------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "=" * 65)
    print(model_name)
    print("=" * 65)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    row = {
        "Model": model_name
    }

    for metric in scoring:

        metric_scores = scores[f"test_{metric}"]

        mean_score = metric_scores.mean()
        std_score = metric_scores.std()

        row[metric] = mean_score

        print(
            f"{metric}: "
            f"{mean_score:.4f} "
            f"+/- {std_score:.4f}"
        )

    results.append(row)


# ------------------------------------------------------------
# 9. DISPLAY FINAL COMPARISON
# ------------------------------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="f1",
    ascending=False
)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(comparison.to_string(index=False))


# ------------------------------------------------------------
# 10. SAVE EXPERIMENT RESULTS
# ------------------------------------------------------------

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\nComparison saved successfully.")
