"""
Credit Risk Classifier Bench — Streamlit front end.

Loads the five pre-trained pipelines from model/ and evaluates them on a
user-uploaded test CSV. No model is re-trained here.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
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
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_TEST = ROOT / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

INK = "#1D3A4A"
GOOD = "#3E7C59"
BAD = "#B5452F"

st.set_page_config(page_title="Credit Risk Classifier Bench", page_icon="🏦", layout="wide")

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; max-width: 1200px; }}
      h1 {{ color: {INK}; letter-spacing: -0.5px; }}
      div[data-testid="stMetric"] {{
          border-left: 4px solid {INK};
          background: rgba(29, 58, 74, 0.05);
          padding: 0.6rem 0.9rem;
          border-radius: 2px;
      }}
      .verdict {{ font-size: 1.05rem; padding: 0.7rem 1rem; border-radius: 3px;
                  background: rgba(62, 124, 89, 0.10); border-left: 4px solid {GOOD}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loaders
@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_schema():
    with open(MODEL_DIR / "feature_schema.json") as f:
        return json.load(f)


@st.cache_data
def load_training_metrics():
    with open(MODEL_DIR / "metrics.json") as f:
        return json.load(f)


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    both_classes = len(np.unique(y)) == 2
    scores = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob) if both_classes else np.nan,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return scores, y_pred, y_prob


def validate(df, schema):
    """Return a list of human-readable problems with the uploaded file."""
    problems = []
    needed = schema["feature_order"] + [schema["target"]]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        problems.append(f"Missing columns: {', '.join(missing)}")
        return problems
    target_vals = set(pd.unique(df[schema["target"]]))
    if not target_vals.issubset({0, 1}):
        problems.append(
            f"Target column '{schema['target']}' must contain only 0 (good) and 1 (bad). "
            f"Found: {sorted(map(str, target_vals))}"
        )
    for col in schema["numeric"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"Column '{col}' should be numeric.")
    return problems


def plot_confusion(cm, title):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        cm, annot=True, fmt="d", cbar=False, cmap="Blues", linewidths=1, linecolor="white",
        xticklabels=["Good (0)", "Bad (1)"], yticklabels=["Good (0)", "Bad (1)"],
        annot_kws={"size": 14}, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def plot_roc(curves):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall on bad risk)")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- layout
models = load_models()
schema = load_schema()
train_metrics = load_training_metrics()

st.title("🏦 Credit Risk Classifier Bench")
st.write(
    "Five classifiers trained on the UCI Statlog German Credit data predict whether a loan "
    "applicant is a **bad credit risk (1)** or a **good one (0)**. Upload a labelled test file "
    "and the saved models are scored on it; nothing is re-trained."
)

with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    use_sample = st.checkbox("Use bundled test_data.csv instead", value=uploaded is None)
    st.download_button(
        "Download sample test_data.csv",
        data=SAMPLE_TEST.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )
    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()), index=4)
    st.caption(f"Tuned hyper-parameters: `{train_metrics[model_name]['best_params']}`")
    st.divider()
    st.caption(
        f"The CSV needs all 20 feature columns plus `{schema['target']}` (0/1). "
        "Download the sample to see the expected format."
    )

if uploaded is not None and not use_sample:
    df = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    df = pd.read_csv(SAMPLE_TEST)
    source = "test_data.csv (bundled)"
else:
    st.info("Upload a CSV in the sidebar, or tick the box to use the bundled test set.")
    st.stop()

problems = validate(df, schema)
if problems:
    st.error("This file can't be scored yet:")
    for p in problems:
        st.write(f"- {p}")
    st.stop()

X = df[schema["feature_order"]]
y = df[schema["target"]].astype(int).to_numpy()

st.caption(
    f"Scoring **{len(df)} rows** from *{source}*  |  "
    f"bad-risk share: {y.mean():.1%}"
)

tab_single, tab_compare, tab_data = st.tabs(["Selected model", "Compare all five", "Data preview"])

# ---- selected model
with tab_single:
    scores, y_pred, y_prob = compute_metrics(models[model_name], X, y)
    st.subheader(model_name)
    cols = st.columns(6)
    for col, m in zip(cols, METRIC_NAMES):
        col.metric(m, "n/a" if np.isnan(scores[m]) else f"{scores[m]:.4f}")

    left, right = st.columns([1, 1.25])
    with left:
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        st.pyplot(plot_confusion(cm, f"Confusion matrix: {model_name}"))
        tn, fp, fn, tp = cm.ravel()
        st.caption(
            f"{fn} bad-risk applicants were approved as good (missed), "
            f"and {fp} good applicants were flagged as bad."
        )
    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y, y_pred, labels=[0, 1], target_names=["good (0)", "bad (1)"],
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

    with st.expander("Row-level predictions"):
        out = X.copy()
        out["actual"] = y
        out["predicted"] = y_pred
        out["p_bad"] = y_prob.round(3)
        st.dataframe(out, width="stretch", height=300)
        st.download_button(
            "Download predictions", out.to_csv(index=False).encode(),
            file_name=f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv",
        )

# ---- comparison
with tab_compare:
    rows, curves = {}, {}
    for name, mdl in models.items():
        s, _, prob = compute_metrics(mdl, X, y)
        rows[name] = s
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, prob)
            curves[name] = (fpr, tpr, s["AUC"])
    table = pd.DataFrame(rows).T[METRIC_NAMES]
    # UCI cost matrix for this dataset: approving a bad applicant costs 5, rejecting a good one costs 1
    costs = {}
    for name, mdl in models.items():
        tn, fp, fn, tp = confusion_matrix(y, mdl.predict(X), labels=[0, 1]).ravel()
        costs[name] = 5 * fn + fp
    table["Cost (5·FN + FP)"] = pd.Series(costs)

    best_mcc = table["MCC"].idxmax()
    best_cost = table["Cost (5·FN + FP)"].idxmin()
    st.markdown(
        f'<div class="verdict">Highest MCC on this file: <b>{best_mcc}</b> '
        f"({table.loc[best_mcc, 'MCC']:.4f}). Lowest misclassification cost under the UCI "
        f"cost matrix: <b>{best_cost}</b> ({int(table.loc[best_cost, 'Cost (5·FN + FP)'])}). "
        f"Accuracy is not the headline because it flatters models that lean towards "
        f"predicting 'good' on a 70/30 split.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.dataframe(
        table.style.format("{:.4f}", subset=METRIC_NAMES)
        .format("{:.0f}", subset=["Cost (5·FN + FP)"])
        .highlight_max(axis=0, subset=METRIC_NAMES, color="#cfe3d6")
        .highlight_min(axis=0, subset=["Cost (5·FN + FP)"], color="#cfe3d6"),
        width="stretch",
    )

    c1, c2 = st.columns(2)
    with c1:
        if curves:
            st.pyplot(plot_roc(curves))
    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        table[["F1", "MCC", "Recall"]].plot.barh(ax=ax, color=[INK, GOOD, BAD], width=0.75)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

# ---- data
with tab_data:
    st.dataframe(df.head(50), width="stretch")
    st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
