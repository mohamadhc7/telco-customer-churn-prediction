# Telco Customer Churn Prediction

## Problem Statement
This project predicts whether a telecom customer will churn, so the business can identify at-risk customers and intervene before they leave.

## Dataset
Telco Customer Churn dataset (Kaggle) — 7,043 customers, 21 original features, ~26% churn rate (imbalanced target).

## Approach

1. Data Cleaning
- Converted `TotalCharges` from text to numeric (it was silently stored as strings, including a small number of blank values)
- Dropped `customerID` — a unique identifier with no predictive value
- Checked for and confirmed no duplicate rows

2. Feature Engineering
- Mapped binary Yes/No columns (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `gender`, `Churn`) to 1/0
- One-hot encoded multi-category columns (`Contract`, `InternetService`, `PaymentMethod`, and all the service add-on columns) using `pd.get_dummies(drop_first=True)`
- Scaled numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) with `StandardScaler`
- Final dataset: 30 fully numeric features

3. Modeling
Trained and compared three models, using an 80/20 train-test split:
- Logistic Regression (baseline)
- Random Forest (default settings)
- Random Forest with `class_weight='balanced'`

## Results

| Model | Accuracy | Churn Precision | Churn Recall | Missed Churners (False Negatives) |
|---|---|---|---|---|
| Logistic Regression | 78.8% | 0.62 | 0.52 | 181 |
| Random Forest (default) | 78.7% | 0.63 | 0.47 | 198 |
| Random Forest (balanced) | 75.3% | 0.53 | 0.61 | 146 |

Chosen model: Random Forest with `class_weight='balanced'`.

Accuracy alone is misleading on this dataset, since ~74% of customers don't churn — a model that just guessed "No" every time would already score ~74%. The metric that actually matters for this business problem is **recall on the churn class**: how many at-risk customers the model successfully flags. The balanced Random Forest catches 61% of actual churners (missing only 146 out of 374), a clear improvement over the baseline's 52% recall — at the cost of more false alarms (201 vs. 117). For a churn-prevention use case, that trade-off is worth it: the cost of missing a customer who leaves for good is typically much higher than the cost of a false-positive retention offer.

## Key Insight

The Random Forest's feature importances show that churn is driven primarily by billing and tenure patterns rather than demographics:

1. `TotalCharges`, `tenure`, and `MonthlyCharges` are by far the strongest predictors — customers who are newer and paying more are the highest churn risk.
2. **Long-term contracts** (`Contract_Two year`) sharply reduce churn risk — customers locked into two-year contracts are far less likely to leave than month-to-month customers.
3. Fiber optic internet and paying by electronic check are both associated with higher churn, hinting at either service satisfaction or billing friction issues worth investigating further.
4. As a sanity check, `gender` — which shouldn't meaningfully affect churn — ranked lowest in importance, as expected.

**Business takeaway:** the highest-value retention targets are newer, high-paying, month-to-month customers on fiber internet. Incentivizing longer contract commitments for this group could meaningfully reduce churn.

## What I'd Do With More Time
- Try gradient boosting models (XGBoost/LightGBM), which often outperform Random Forest on tabular data
- Tune the classification threshold directly instead of relying only on `class_weight`, to fine-tune the precision/recall trade-off
- Engineer additional features, such as a charges-per-tenure-month ratio
- Use cross-validation instead of a single train/test split for more robust performance estimates

## Tech Stack
Python, pandas, scikit-learn, matplotlib
