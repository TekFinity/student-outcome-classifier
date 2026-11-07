# ML Assignment 2 - Student Outcome Prediction
#
# Streamlit app: upload test data, pick a model, see the results.
# Models are already trained (train_models.py) - here we only load and predict.

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from features import CLASS_ORDER, split_features_target

st.set_page_config(page_title="Student Outcome Prediction")

# bigger font for the metric numbers and the tables
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2.4rem; font-weight: 600; }
[data-testid="stMetricLabel"] p { font-size: 1.2rem; }
[data-testid="stTable"] td, [data-testid="stTable"] th { font-size: 1.15rem; }
</style>
""", unsafe_allow_html=True)

models = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}


def get_metrics(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)
    try:
        auc = roc_auc_score(y, prob, multi_class="ovr", average="weighted",
                            labels=model.classes_)
    except ValueError:  # happens if the file does not contain all 3 classes
        auc = float("nan")
    scores = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return scores, pred


STUDENT_NAME = "Asmita Chhanchure"
BITS_ID = "2026K05172"

# student details in the sidebar (always visible, even before upload)
st.sidebar.markdown(f"""
<div style="font-size:1.6rem; font-weight:600;">Submitted by</div>
<div style="font-size:1.4rem; margin-top:0.5rem;"><b>{STUDENT_NAME}</b></div>
<div style="font-size:1.3rem;">BITS ID: <b>{BITS_ID}</b></div>
<div style="font-size:1.05rem; color:gray; margin-top:0.5rem;">
M.Tech (AIML) - Machine Learning, Assignment 2</div>
""", unsafe_allow_html=True)

st.title("Student Outcome Prediction")
st.write("Predicts if a student will **Dropout**, stay **Enrolled** or **Graduate**. "
         "Dataset: UCI - Predict Students' Dropout and Academic Success.")

# 1. Upload test data
file = st.file_uploader("Upload test data (CSV)", type="csv")
if file is None:
    st.info("Please upload test_data.csv to see the results.")
    st.stop()

data = pd.read_csv(file)
st.write(f"Rows in file: {data.shape[0]}")
st.dataframe(data.head())

X, y = split_features_target(data)
if y is None:
    st.error("The file must have a 'Target' column.")
    st.stop()

# 2. Choose model
model_name = st.selectbox("Select model", list(models.keys()))
model = joblib.load(models[model_name])

# 3. Evaluation metrics
scores, pred = get_metrics(model, X, y)
st.subheader(f"Results - {model_name}")
row1 = st.columns(3)
row2 = st.columns(3)
for col, (name, value) in zip(row1 + row2, scores.items()):
    col.metric(name, f"{value:.4f}")

# 4. Confusion matrix and classification report
st.subheader("Confusion Matrix")
cm = confusion_matrix(y, pred, labels=CLASS_ORDER)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report")
report = classification_report(y, pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
st.table(pd.DataFrame(report).T.round(3))

# 5. All models on the same test data
st.subheader("Comparison of All Models")
rows = []
for name, path in models.items():
    s, _ = get_metrics(joblib.load(path), X, y)
    rows.append({"Model": name, **s})
st.table(pd.DataFrame(rows).set_index("Model").round(4))

st.divider()
st.markdown(f"<div style='text-align:center; font-size:1.2rem;'>"
            f"ML Assignment 2 | <b>{STUDENT_NAME}</b> | BITS ID: <b>{BITS_ID}</b></div>",
            unsafe_allow_html=True)
