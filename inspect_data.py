import pandas as pd
from pathlib import Path




DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)


data= pd.read_csv(DATA_PATH)

print("\nData Inspection:")
print(data.head())
# print first 5 rows of the dataset

print("\nData Info:")
data.info()
# print the summary of the dataset including the data types and non-null values

print("\nData Description:")
print(data.describe())
# print the statistical summary of the dataset including count, mean, std, min, 25%, 50%, 75%, and max values for numerical columns

print("\n Dimension of the dataset:")
print(data.shape)
# print the number of rows and columns in the dataset

print("\n Missing Values:")
print(data.isnull().sum())
# print the number of missing values in each column of the dataset

print("\n duplicate Rows:")
print(data.duplicated().sum())
# print the number of duplicate rows in the dataset

print("\n Churn Distribution:")
print(data['Churn'].value_counts())
# print the distribution of the target variable 'Churn' in the dataset

print("\n Churn Distribution Percentage:")
print(data['Churn'].value_counts(normalize=True) * 100)
# print the distribution of the target variable 'Churn' in percentage in the dataset

print("\n Blank Total Charges:")
blank_total_charges = (
    data["TotalCharges"]
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)
print(blank_total_charges)
# print the number of blank values in the 'TotalCharges' column of the dataset