
# ============================================================
# CUSTOMER CHURN PREDICTION
# EXPERIMENT 7 - MODEL INTERPRETATION
#
# Methods:
# 1. Permutation Importance
# 2. Logistic Regression Coefficients
# ============================================================

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "customer_churn_final_pipeline.joblib"
)

REPORT_DIR = PROJECT_ROOT / "reports"

FIGURES_DIR = REPORT_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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
# 3. RECREATE ORIGINAL TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 4. LOAD FINAL MODEL
# ------------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("Final model loaded successfully.")


# ------------------------------------------------------------
# 5. PERMUTATION IMPORTANCE
# ------------------------------------------------------------

print("\nCalculating permutation importance...")

result = permutation_importance(
    estimator=model,
    X=X_test,
    y=y_test,
    scoring="average_precision",
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)


importance = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
})

importance = importance.sort_values(
    by="importance_mean",
    ascending=False
)

print("\n========== PERMUTATION IMPORTANCE ==========")

print(
    importance.to_string(index=False)
)


# ------------------------------------------------------------
# 6. SAVE IMPORTANCE RESULTS
# ------------------------------------------------------------

importance.to_csv(
    REPORT_DIR / "permutation_importance.csv",
    index=False
)


# ------------------------------------------------------------
# 7. PLOT PERMUTATION IMPORTANCE
# ------------------------------------------------------------

top_features = importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance_mean"][::-1]
)

plt.xlabel("Decrease in Average Precision")
plt.ylabel("Feature")

plt.title("Customer Churn - Permutation Importance")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "permutation_importance.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 8. EXTRACT LOGISTIC REGRESSION COEFFICIENTS
# ------------------------------------------------------------

preprocessor = model.named_steps["preprocessor"]

classifier = model.named_steps["classifier"]

encoded_feature_names = (
    preprocessor.get_feature_names_out()
)

coefficients = classifier.coef_[0]

coefficient_data = pd.DataFrame({
    "feature": encoded_feature_names,
    "coefficient": coefficients
})

coefficient_data["absolute_coefficient"] = (
    coefficient_data["coefficient"].abs()
)

coefficient_data = coefficient_data.sort_values(
    by="absolute_coefficient",
    ascending=False
)

print("\n========== LOGISTIC REGRESSION COEFFICIENTS ==========")

print(
    coefficient_data.head(20).to_string(index=False)
)


# ------------------------------------------------------------
# 9. SAVE COEFFICIENT RESULTS
# ------------------------------------------------------------

coefficient_data.to_csv(
    REPORT_DIR / "logistic_coefficients.csv",
    index=False
)


# ------------------------------------------------------------
# 10. PLOT LARGEST COEFFICIENTS
# ------------------------------------------------------------

top_coefficients = coefficient_data.head(15)

plt.figure(figsize=(11, 7))

colors = [
    "green" if value > 0 else "red"
    for value in top_coefficients["coefficient"]
]

plt.barh(
    top_coefficients["feature"][::-1],
    top_coefficients["coefficient"][::-1],
    color=colors[::-1]
)

plt.axvline(
    0,
    color="black",
    linewidth=0.8
)

plt.xlabel("Logistic Regression Coefficient")

plt.ylabel("Feature")

plt.title("Features Influencing Churn Predictions")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "logistic_coefficients.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 11. FINAL OUTPUT
# ------------------------------------------------------------

print("\nInterpretation completed.")

print(
    "\nPermutation importance saved:",
    REPORT_DIR / "permutation_importance.csv"
)

print(
    "Coefficient results saved:",
    REPORT_DIR / "logistic_coefficients.csv"
)

print("\nFigures saved:", FIGURES_DIR)
