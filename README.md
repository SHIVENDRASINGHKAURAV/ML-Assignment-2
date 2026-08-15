# HR Employee Attrition — Classification Model Comparison

## Problem Statement

Employee attrition is costly for organizations, both in terms of lost expertise and the expense of hiring and onboarding replacements. This project frames attrition prediction as a **binary classification problem**: given an employee's demographic, job, and compensation attributes, predict whether they will leave the company (`Attrition = Yes`) or stay (`Attrition = No`).

Five classical machine learning classifiers are trained on the same dataset and train/test split, evaluated using a consistent set of metrics, and compared to identify the best-performing model for this task. The trained models are exposed through an interactive Streamlit application that lets a user upload a test CSV, choose a model, and view its evaluation results.

## Dataset Description

**Source:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle)

- 1,470 employee records, 35 original columns (features + target).
- No missing values, no duplicate rows.
- Target column: `Attrition` (`Yes` / `No`), mapped to `1` / `0` for modeling.
  - Class distribution: **No = 1233 (83.9%)**, **Yes = 237 (16.1%)** — the dataset is imbalanced, which shapes both the preprocessing choices (stratified split) and the model configuration (`class_weight="balanced"` where supported) and how results should be interpreted (accuracy alone is misleading; MCC is emphasized).
- Four columns were dropped before modeling as they carry no predictive signal: `EmployeeCount` and `StandardHours` (constant for all rows), `Over18` (constant value `Y`), and `EmployeeNumber` (a unique row identifier).
- Seven categorical features (`BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`) were one-hot encoded; the remaining numeric features were standardized. Both transformations are handled by a single `ColumnTransformer` (`sklearn.compose.ColumnTransformer`), fit once on the training split and reused across all five models and the deployed app.
- Data was split into training (80%) and test (20%) sets using a **stratified** split (`stratify=y`, `random_state=42`) so both sets preserve the original ~84/16 class ratio. The test split is saved as `test_data.csv` and is what the Streamlit app expects to be uploaded.

## Repository

GitHub: [https://github.com/SHIVENDRASINGHKAURAV/shivendra-ml-assignment2](https://github.com/SHIVENDRASINGHKAURAV/shivendra-ml-assignment2)

## Live App

Streamlit app: [https://shivendra-ml-assignment2-g2qev3kkblrwytyuyvwbh3.streamlit.app/](https://shivendra-ml-assignment2-g2qev3kkblrwytyuyvwbh3.streamlit.app/)

## Models Used

All five models were trained on an identical preprocessed train/test split, and evaluated on the held-out test set using six metrics: Accuracy, AUC (ROC AUC on predicted probabilities), Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7517 | 0.8019 | 0.3488 | 0.6383 | 0.4511 | 0.3316 |
| Decision Tree | 0.7721 | 0.5801 | 0.2917 | 0.2979 | 0.2947 | 0.1589 |
| K-Nearest Neighbors | 0.8333 | 0.6072 | 0.4286 | 0.1277 | 0.1967 | 0.1640 |
| Naive Bayes (Gaussian) | 0.6429 | 0.6874 | 0.2583 | 0.6596 | 0.3713 | 0.2231 |
| Random Forest | 0.8401 | 0.7903 | 0.5000 | 0.0638 | 0.1132 | 0.1340 |

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall balance: highest AUC (0.8019) and by far the highest recall (0.6383) among all models, with reasonable precision (0.3488) — yields the best F1 (0.4511) and MCC (0.3316). `class_weight="balanced"` is clearly effective here at pushing the model to catch minority-class (attrition) cases. |
| Decision Tree | Weakest model overall — lowest AUC (0.5801) and mediocre recall (0.2979) despite `class_weight="balanced"`, indicating it overfits the training data and generalizes poorly to the minority class on the test set. |
| kNN | High accuracy (0.8333) is misleading — recall is very low (0.1277), meaning it misses most actual attrition cases and mostly predicts the majority "No" class. kNN has no `class_weight` support, so it has no built-in mechanism to correct for the class imbalance. |
| Naive Bayes | Lowest accuracy (0.6429) but relatively high recall (0.6596) — it is biased toward predicting attrition (trading precision for recall). Like kNN, Gaussian Naive Bayes has no `class_weight` parameter, so this behavior comes purely from the algorithm's independence assumption on this feature set. |
| Random Forest (Ensemble) | Highest accuracy (0.8401) and second-highest AUC (0.7903), but the worst recall of all five models (0.0638) despite using `class_weight="balanced"`. Bagging across 200 trees dilutes the per-tree balancing effect, so the ensemble defaults heavily toward the majority class — a good reminder that high accuracy alone does not mean a model is useful on imbalanced data. |
| Overall Winner for your dataset? | **Logistic Regression**, chosen by MCC (0.3316) — the most reliable single metric for imbalanced binary classification since it weighs all four confusion-matrix outcomes symmetrically. It also has the best AUC and recall, making it the most practically useful model for actually identifying employees likely to leave, which matters more here than raw accuracy. |
