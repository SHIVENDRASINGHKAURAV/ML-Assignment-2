"""Streamlit app for the HR Attrition classification assignment.

Loads the pre-trained preprocessor and models (produced by preprocessing.py
and the scripts in model/) and evaluates them on a user-uploaded CSV.
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
TARGET_COL = "Attrition"

MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}


@st.cache_resource
def load_preprocessor():
    return joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))


@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.set_page_config(page_title="Who's Leaving? — IBM HR Attrition", layout="centered")
st.title("🧑‍💼 Who's Leaving? — IBM HR Attrition Model Explorer")

st.markdown(
    "This app evaluates 5 classifiers trained on IBM's HR Analytics dataset "
    "to predict employee attrition (1,470 employees, 16.1% attrition rate). "
    "Upload the held-out test split (`test_data.csv`) and pick a model below "
    "to inspect how it performs — including its confusion matrix and "
    "per-class precision/recall, since accuracy alone is misleading on this "
    "imbalanced target."
)

uploaded_file = st.file_uploader("Upload test CSV", type="csv")
model_name = st.selectbox("Select model", list(MODELS.keys()))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if TARGET_COL not in df.columns:
        st.error(f"Uploaded CSV must contain a '{TARGET_COL}' column.")
    else:
        X = df.drop(columns=[TARGET_COL])
        y_true = df[TARGET_COL].map({"Yes": 1, "No": 0})

        preprocessor = load_preprocessor()
        X_enc = preprocessor.transform(X)

        model = load_model(MODELS[model_name])
        y_pred = model.predict(X_enc)
        y_proba = model.predict_proba(X_enc)[:, 1]

        metrics = compute_metrics(y_true, y_pred, y_proba)

        st.subheader(f"Metrics — {model_name}")
        st.caption(
            "Overall scorecard for the selected model on the uploaded test set: "
            "Accuracy/AUC/Precision/Recall/F1/MCC computed against the true "
            "`Attrition` labels in the CSV."
        )
        st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

        st.subheader("Confusion Matrix")
        st.caption(
            "Rows = actual attrition status, columns = predicted status. "
            "Diagonal cells are correct predictions; off-diagonal cells are "
            "errors (e.g. bottom-left = employees who left but the model "
            "predicted 'No')."
        )
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.caption(
            "Per-class breakdown of precision/recall/F1, plus macro and "
            "weighted averages. Useful here since accuracy alone hides how "
            "well the model catches the minority 'Yes' (attrition) class."
        )
        report = classification_report(
            y_true, y_pred, target_names=["No", "Yes"], output_dict=True
        )
        st.table(pd.DataFrame(report).transpose())

        with st.expander("📊 Compare all 5 models on this upload"):
            st.caption(
                "Runs every trained model on the same uploaded test set so "
                "you can see how they stack up side by side, instead of "
                "switching the dropdown one model at a time."
            )
            comparison_rows = []
            for name, filename in MODELS.items():
                other_model = load_model(filename)
                other_pred = other_model.predict(X_enc)
                other_proba = other_model.predict_proba(X_enc)[:, 1]
                row = {"Model": name}
                row.update(compute_metrics(y_true, other_pred, other_proba))
                comparison_rows.append(row)

            comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
            st.dataframe(comparison_df.style.format("{:.4f}"))
            st.bar_chart(comparison_df["MCC"])
            st.caption(
                "MCC (Matthews Correlation Coefficient) is shown above since "
                "it's the most reliable single metric for this imbalanced "
                "dataset — it accounts for all four confusion-matrix cells "
                "symmetrically, unlike accuracy."
            )
