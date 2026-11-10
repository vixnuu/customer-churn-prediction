# Customer Churn Prediction — Model Interpretation

## Interpretation methods

Two approaches were used to investigate the trained Logistic Regression model:

1. Permutation Importance
2. Logistic Regression Coefficient Analysis

## Top predictive features

| Rank | Feature | Permutation Importance |
|---:|---|---:|
| 1 | Tenure | 0.19895 |
| 2 | Contract | 0.05121 |
| 3 | InternetService | 0.01937 |
| 4 | TechSupport | 0.01414 |
| 5 | OnlineSecurity | 0.01220 |
| 6 | TotalCharges | 0.00871 |

## Main findings

Tenure was the most influential feature under permutation importance using Average Precision as the evaluation metric.

Contract type was the second most influential feature.

The Logistic Regression coefficients showed a negative association between tenure and predicted churn log-odds, while the month-to-month contract category had a positive coefficient.

These findings are consistent with the exploratory analysis, which showed substantially higher observed churn rates among month-to-month customers.

## Important limitations

Feature importance measures predictive contribution, not causation.

Highly correlated features, including tenure and TotalCharges, can complicate coefficient interpretation.

Small or negative permutation importance values do not necessarily mean a feature is harmful; they can reflect noise, redundancy, and sampling variability.

The interpretation results describe this fitted model and evaluation dataset, not universal customer behavior.