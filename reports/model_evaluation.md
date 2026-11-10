# Customer Churn Prediction — Model Evaluation

## Selected model

Logistic Regression with balanced class weights.

**Hyperparameters**
- C: 0.1
- Class weight: balanced
- Maximum iterations: 2,000
- Decision threshold: 0.547

## Dataset

- Dataset: IBM Telco Customer Churn
- Total records: 7,043
- Input features: 19
- Training records: 5,634
- Test records: 1,409
- Target: Churn (Yes/No)

## Final test results

| Metric | Result |
|---|---:|
| Accuracy | 75.44% |
| Balanced accuracy | 75.52% |
| Precision | 52.60% |
| Recall | 75.67% |
| F1 score | 62.06% |
| ROC-AUC | 84.10% |
| Average precision | 63.33% |

## Confusion matrix

| | Predicted stayed | Predicted churned |
|---|---:|---:|
| Actual stayed | 780 | 255 |
| Actual churned | 91 | 283 |

## Model selection

Logistic Regression and Random Forest were compared using stratified five-fold cross-validation.

Balanced Logistic Regression achieved the highest cross-validation recall, while tuned Random Forest achieved a marginally higher F1 score.

Logistic Regression was selected for the initial final model because it provided a slightly better F1 score at the cross-validation stage, higher recall, balanced accuracy, ROC-AUC, and average precision than the tuned Random Forest comparison.

## Limitations

The model produces probabilistic churn estimates, not guarantees that individual customers will leave.

The dataset represents a fictional telecommunications company. Its results should not be interpreted as demonstrated performance on real company customers.

Further validation, monitoring, and business-cost analysis are required before production use.