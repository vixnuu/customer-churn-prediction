
# ============================================================
# CUSTOMER CHURN PREDICTION
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

data = pd.read_csv(DATA_PATH)

print("\n========== ORIGINAL DATASET ==========")
print(data.shape)


# ------------------------------------------------------------
# 3. CLEAN TOTALCHARGES
# ------------------------------------------------------------

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"].astype("string").str.strip(),
    errors="coerce"
)

print("\nMissing values after conversion:")
print(data.isnull().sum())

# Inspect rows with missing TotalCharges
blank_rows = data[data["TotalCharges"].isna()]

print("\nRows with missing TotalCharges:")
print(
    blank_rows[
        ["customerID", "tenure", "MonthlyCharges", "Churn"]
    ]
)

# Only replace missing charges with 0 if every affected
# customer has tenure equal to zero.
if not blank_rows.empty:
    if (blank_rows["tenure"] == 0).all():

        data["TotalCharges"] = data["TotalCharges"].fillna(0)

        print(
            "\nFilled blank TotalCharges with 0 "
            "for customers with zero tenure."
        )

    else:
        print(
            "\nWARNING: Some missing TotalCharges values "
            "have non-zero tenure. Investigate before proceeding."
        )


# ------------------------------------------------------------
# 4. VERIFY DATA TYPES AND DUPLICATES
# ------------------------------------------------------------

print("\n========== UPDATED DATA INFO ==========")
data.info()

print("\nDuplicate rows:", data.duplicated().sum())

print(
    "Duplicate customer IDs:",
    data["customerID"].duplicated().sum()
)

print("\nUpdated data types:")
print(data.dtypes)


# ------------------------------------------------------------
# 5. SAVE CLEANED DATA
# ------------------------------------------------------------

CLEANED_PATH = PROCESSED_DIR / "telco_churn_cleaned.csv"

data.to_csv(CLEANED_PATH, index=False)

print("\nCleaned data saved to:", CLEANED_PATH)


# ------------------------------------------------------------
# 6. CHURN DISTRIBUTION
# ------------------------------------------------------------

print("\n========== CHURN DISTRIBUTION ==========")

print(data["Churn"].value_counts())

print("\nChurn percentage:")

print(
    data["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=data,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "churn_distribution.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 7. NUMERICAL FEATURE DISTRIBUTIONS
# ------------------------------------------------------------

numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for feature in numerical_features:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=data,
        x=feature,
        kde=True
    )

    plt.title(f"Distribution of {feature}")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{feature}_distribution.png",
        dpi=150
    )

    plt.close()


# ------------------------------------------------------------
# 8. TENURE VS CHURN
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=data,
    x="tenure",
    hue="Churn",
    bins=24,
    multiple="layer"
)

plt.title("Customer Tenure vs Churn")
plt.xlabel("Tenure (Months)")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "tenure_vs_churn.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 9. MONTHLY CHARGES VS CHURN
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=data,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Churn")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "monthly_charges_vs_churn.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 10. CONTRACT VS CHURN
# ------------------------------------------------------------

contract_churn = (
    data.groupby("Contract")["Churn"]
    .apply(lambda values: (values == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

print("\n========== CONTRACT VS CHURN ==========")

print(contract_churn.round(2))

plt.figure(figsize=(9, 5))

contract_churn.plot(
    kind="bar"
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "contract_vs_churn.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 11. PAYMENT METHOD VS CHURN
# ------------------------------------------------------------

payment_churn = (
    data.groupby("PaymentMethod")["Churn"]
    .apply(lambda values: (values == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

print("\n========== PAYMENT METHOD VS CHURN ==========")

print(payment_churn.round(2))

plt.figure(figsize=(10, 5))

payment_churn.plot(
    kind="bar"
)

plt.title("Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=25, ha="right")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "payment_method_vs_churn.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 12. CORRELATION ANALYSIS
# ------------------------------------------------------------

numerical_data = data[
    [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]
]

correlation = numerical_data.corr()

print("\n========== CORRELATION MATRIX ==========")

print(correlation.round(2))

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Numerical Feature Correlation")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "correlation_heatmap.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 13. FINAL SUMMARY
# ------------------------------------------------------------

print("\n========== EDA COMPLETED ==========")

print("Cleaned dataset:", CLEANED_PATH)

print("Figures saved in:", FIGURES_DIR)

print("\nGenerated figures:")

for file in sorted(FIGURES_DIR.glob("*.png")):
    print(file.name)
