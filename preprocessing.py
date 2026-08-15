"""Shared data preparation pipeline for the HR Attrition classification task.

Run once locally:
    python preprocessing.py

Produces:
    test_data.csv           -- raw-format test split (used by app.py and for grading)
    model/preprocessor.pkl  -- fitted ColumnTransformer (used by app.py and model scripts)

Model training scripts import `load_train_test()` to get the already-encoded,
already-scaled arrays so every model is trained/evaluated on an identical split.
"""

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
TEST_DATA_PATH = "test_data.csv"
PREPROCESSOR_PATH = "model/preprocessor.pkl"

TARGET_COL = "Attrition"
DROP_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
CATEGORICAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def _load_raw():
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.drop(columns=DROP_COLS)
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return df


def _build_preprocessor(feature_df):
    numeric_cols = [c for c in feature_df.columns if c not in CATEGORICAL_COLS]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
                CATEGORICAL_COLS,
            ),
        ]
    )


def prepare():
    df = _load_raw()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.map({1: "Yes", 0: "No"})
    test_df.to_csv(TEST_DATA_PATH, index=False)

    preprocessor = _build_preprocessor(X_train)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    return X_train_enc, X_test_enc, y_train.values, y_test.values


def load_train_test():
    """Used by model training scripts to get an identical, already-encoded split."""
    df = _load_raw()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_train_enc = preprocessor.transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    return X_train_enc, X_test_enc, y_train.values, y_test.values


if __name__ == "__main__":
    prepare()
    print(f"Saved {TEST_DATA_PATH} and {PREPROCESSOR_PATH}")
