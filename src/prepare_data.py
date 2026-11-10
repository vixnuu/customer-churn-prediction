
# ============================================================
# CUSTOMER CHURN PREDICTION
# STEP 3 - DATA PREPARATION
# ============================================================

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)


# ------------------------------------------------------------
# 2. LOAD CLEANED DATA
# ------------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)


# ------------------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------

if "Churn" not in data.columns:
    raise ValueError("Target column 'Churn' is missing.")

if "customerID" not in data.columns:
    raise ValueError("customerID column is missing.")

# Convert TotalCharges to numerical type
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Handle any unexpected missing values safely
data["TotalCharges"] = data["TotalCharges"].fillna(0)


# ------------------------------------------------------------
# 4. REMOVE IDENTIFIER COLUMN
# ------------------------------------------------------------

data = data.drop(columns=["customerID"])


# ------------------------------------------------------------
# 5. DEFINE FEATURES AND TARGET
# ------------------------------------------------------------

X = data.drop(columns=["Churn"])

# Convert target labels:
# No  -> 0
# Yes -> 1

y = data["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected values found in Churn column.")


# ------------------------------------------------------------
# 6. IDENTIFY NUMERICAL FEATURES
# ------------------------------------------------------------

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)


# ------------------------------------------------------------
# 7. IDENTIFY CATEGORICAL FEATURES
# ------------------------------------------------------------

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)


# ------------------------------------------------------------
# 8. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 9. DISPLAY SPLIT INFORMATION
# ------------------------------------------------------------

print("\n========== DATA SPLIT ==========")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")

print(
    y_train.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nTesting target distribution:")

print(
    y_test.value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ------------------------------------------------------------
# 10. DISPLAY FINAL FEATURE INFORMATION
# ------------------------------------------------------------

print("\n========== FINAL FEATURE SUMMARY ==========")

print("Total features:", X.shape[1])

print("Numerical features:", len(numerical_features))

print("Categorical features:", len(categorical_features))

print("Target classes:")
print(y.value_counts())


print("\nData preparation completed successfully.")
