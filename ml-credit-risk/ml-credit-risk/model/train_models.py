"""
Train and evaluate five classifiers on the UCI Statlog (German Credit) dataset.

Run from the project root:
    python model/train_models.py

What it does
1. Reads data/german_credit_raw.csv (original UCI coding, no header).
2. Renames columns and decodes the A-codes into readable categories.
3. Stratified 80/20 split -> test_data.csv is written to the project root.
4. For each model: a Pipeline (preprocessing + classifier) is tuned with
   5-fold stratified CV on the TRAIN split only, then evaluated on the TEST split.
5. Saves each fitted pipeline to model/<name>.joblib, plus metrics.json
   and feature_schema.json used by the Streamlit app.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "german_credit_raw.csv"
CLEAN_PATH = ROOT / "data" / "german_credit.csv"
TEST_PATH = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"

TARGET = "credit_risk"  # 1 = bad credit risk, 0 = good credit risk

COLUMNS = [
    "checking_account", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since",
    "installment_rate_pct", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age_years", "other_installment_plans",
    "housing", "existing_credits", "job", "num_dependents", "telephone",
    "foreign_worker", TARGET,
]

# Decoding of the UCI attribute codes (from german.doc in the UCI archive)
CODE_MAP = {
    "checking_account": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM",
                         "A14": "no checking account"},
    "credit_history": {"A30": "no credits / all paid", "A31": "all paid at this bank",
                       "A32": "existing paid duly", "A33": "past delays",
                       "A34": "critical account"},
    "purpose": {"A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
                "A43": "radio/television", "A44": "domestic appliances",
                "A45": "repairs", "A46": "education", "A47": "vacation",
                "A48": "retraining", "A49": "business", "A410": "others"},
    "savings_account": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
                        "A64": ">= 1000 DM", "A65": "unknown / none"},
    "employment_since": {"A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years",
                         "A74": "4-7 years", "A75": ">= 7 years"},
    "personal_status_sex": {"A91": "male: divorced/separated",
                            "A92": "female: divorced/separated/married",
                            "A93": "male: single", "A94": "male: married/widowed",
                            "A95": "female: single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings agreement / life insurance",
                 "A123": "car or other", "A124": "unknown / none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unemployed / unskilled non-resident", "A172": "unskilled resident",
            "A173": "skilled employee", "A174": "management / highly qualified"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

CATEGORICAL = list(CODE_MAP.keys())
NUMERIC = [c for c in COLUMNS if c not in CATEGORICAL and c != TARGET]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, header=None, names=COLUMNS)
    for col, mapping in CODE_MAP.items():
        df[col] = df[col].map(mapping)
    # Original coding: 1 = good, 2 = bad  ->  0 = good, 1 = bad (positive class)
    df[TARGET] = (df[TARGET] == 2).astype(int)
    assert df.isna().sum().sum() == 0, "Unmapped category codes found"
    return df


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def model_specs():
    """Each entry: display name, file stem, estimator, hyper-parameter grid."""
    return [
        ("Logistic Regression", "logistic_regression",
         LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
         {"clf__C": [0.01, 0.1, 1, 10]}),
        ("Decision Tree", "decision_tree",
         DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
         {"clf__max_depth": [3, 4, 5, 6, 8, None], "clf__min_samples_leaf": [1, 5, 10, 20]}),
        ("KNN", "knn",
         KNeighborsClassifier(),
         {"clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21], "clf__weights": ["uniform", "distance"]}),
        ("Naive Bayes", "naive_bayes",
         GaussianNB(),
         {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
        ("Support Vector Machine", "svm",
         # SVC has no native probabilities; Platt-style calibration provides them for AUC
         CalibratedClassifierCV(
             SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
             ensemble=False),
         {"clf__estimator__C": [0.1, 1, 10], "clf__estimator__gamma": ["scale", 0.01, 0.1]}),
    ]


def evaluate(pipe, X, y) -> dict:
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]).tolist(),
    }


def main():
    df = load_and_clean()
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1] - 1} features")
    print(f"Class balance (1 = bad): {df[TARGET].mean():.2%} positive")

    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_PATH, index=False)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  -> saved {TEST_PATH.name}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, stem, estimator, grid in model_specs():
        t0 = time.time()
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", estimator)])
        # Tune on F1 of the "bad risk" class: accuracy alone is misleading at 70/30
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        joblib.dump(best, MODEL_DIR / f"{stem}.joblib")
        metrics = evaluate(best, X_test, y_test)
        metrics["best_params"] = {k.replace("clf__", "").replace("estimator__", ""): v for k, v in search.best_params_.items()}
        metrics["cv_f1"] = search.best_score_
        metrics["file"] = f"{stem}.joblib"
        results[name] = metrics
        print(f"{name:<24} acc={metrics['Accuracy']:.4f} auc={metrics['AUC']:.4f} "
              f"f1={metrics['F1']:.4f} mcc={metrics['MCC']:.4f} "
              f"params={metrics['best_params']} ({time.time() - t0:.1f}s)")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    schema = {
        "target": TARGET,
        "positive_label": "bad credit risk",
        "numeric": NUMERIC,
        "categorical": {c: sorted(set(m.values())) for c, m in CODE_MAP.items()},
        "feature_order": [c for c in COLUMNS if c != TARGET],
    }
    with open(MODEL_DIR / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    table = pd.DataFrame({k: {m: v[m] for m in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}
                          for k, v in results.items()}).T.round(4)
    print("\n" + table.to_string())


if __name__ == "__main__":
    main()
