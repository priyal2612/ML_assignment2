# Credit Risk Classification: German Credit Data

ML Assignment 2, M.Tech (AIML), BITS Pilani WILP

## a. Problem statement

A bank has to decide whether a loan applicant is likely to repay. Approving an applicant who later defaults is far more expensive than turning away one who would have repaid. The goal is to predict, from an applicant's financial and personal attributes, whether they are a **good (0)** or **bad (1)** credit risk, to compare five classification algorithms on the same data, and to serve the trained models through an interactive Streamlit app that scores a labelled test file without re-training.

## b. Dataset description

**Source:** Statlog (German Credit Data), UCI Machine Learning Repository, contributed by Prof. Hans Hofmann. https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

| Property | Value |
|---|---|
| Instances | 1,000 loan applicants |
| Features | 20 (7 numeric, 13 categorical) |
| Target | `credit_risk`: 0 = good (700), 1 = bad (300) |
| Task | Binary classification, imbalanced 70 / 30 |
| Missing values | None |

**Features**

| Type | Columns |
|---|---|
| Numeric | `duration_months`, `credit_amount`, `installment_rate_pct`, `residence_since`, `age_years`, `existing_credits`, `num_dependents` |
| Categorical | `checking_account`, `credit_history`, `purpose`, `savings_account`, `employment_since`, `personal_status_sex`, `other_debtors`, `property`, `other_installment_plans`, `housing`, `job`, `telephone`, `foreign_worker` |

The raw UCI file stores categories as codes (`A11`, `A34`, ...). `model/train_models.py` decodes them into readable labels using the UCI documentation, and recodes the target from 1/2 to 0/1 so that "bad risk" is the positive class.

**Cost matrix.** UCI supplies a cost matrix for this dataset: classifying a bad applicant as good costs 5, classifying a good applicant as bad costs 1. This is reported alongside the standard metrics because it reflects the real business trade-off.

**Preprocessing (inside each saved pipeline)**
- Numeric columns: `StandardScaler`
- Categorical columns: `OneHotEncoder(handle_unknown="ignore")`, giving 61 model inputs
- Stratified 80 / 20 train-test split (`random_state=42`): 800 train rows, 200 test rows (`test_data.csv`)
- Hyper-parameters tuned with 5-fold stratified `GridSearchCV` on the training split only, optimising F1 of the bad-risk class
- `class_weight="balanced"` used for Logistic Regression, Decision Tree and SVM to counter the 70 / 30 imbalance

## c. GitHub repository link

https://github.com/YOUR-USERNAME/YOUR-REPO-NAME

**Live Streamlit app:** https://YOUR-APP-NAME.streamlit.app

### Repository structure

```
.
├── app.py                       # Streamlit app (loads saved models, never re-trains)
├── requirements.txt
├── README.md
├── test_data.csv                # 200-row held-out test set (features + credit_risk)
├── .streamlit/config.toml       # app theme
├── data/
│   ├── german_credit_raw.csv    # original UCI file (coded)
│   └── german_credit.csv        # decoded, readable version
└── model/
    ├── train_models.py          # preprocessing, tuning, evaluation, saving
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics.json             # test metrics, confusion matrices, best params
    └── feature_schema.json      # expected columns, used to validate uploads
```

Each `.joblib` file is a complete scikit-learn `Pipeline` (preprocessing + classifier), so the app can score raw CSV rows directly.

### How to run

```bash
pip install -r requirements.txt
python model/train_models.py     # re-creates test_data.csv, models and metrics.json
streamlit run app.py
```

## d. Models used

Five classifiers, all trained on the same 800 rows and evaluated on the same 200 held-out rows:

| Model | Tuned hyper-parameters |
|---|---|
| Logistic Regression | C = 0.1, class_weight = balanced |
| Decision Tree | max_depth = 5, min_samples_leaf = 1, class_weight = balanced |
| K-Nearest Neighbors | k = 5, weights = distance |
| Naive Bayes (Gaussian) | var_smoothing = 1e-3 |
| Support Vector Machine | RBF kernel, C = 1, gamma = 0.01, class_weight = balanced, probabilities via `CalibratedClassifierCV` |

### Comparison table (test set, n = 200, 60 bad / 140 good)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7250 | 0.8094 | 0.5281 | 0.7833 | 0.6309 | 0.4457 |
| Decision Tree | 0.5700 | 0.5990 | 0.3774 | 0.6667 | 0.4819 | 0.1793 |
| KNN | 0.7250 | 0.7330 | 0.5676 | 0.3500 | 0.4330 | 0.2782 |
| Naive Bayes | 0.6950 | 0.7429 | 0.4937 | 0.6500 | 0.5612 | 0.3415 |
| Support Vector Machine | 0.7850 | 0.8201 | 0.6889 | 0.5167 | 0.5905 | 0.4573 |

Precision, Recall and F1 refer to the positive class (bad credit risk).

**Confusion matrices and UCI cost (5 × FN + 1 × FP, lower is better)**

| Model | TN | FP | FN | TP | Cost |
|---|---|---|---|---|---|
| Logistic Regression | 98 | 42 | 13 | 47 | **107** |
| Decision Tree | 74 | 66 | 20 | 40 | 166 |
| KNN | 124 | 16 | 39 | 21 | 211 |
| Naive Bayes | 100 | 40 | 21 | 39 | 145 |
| Support Vector Machine | 126 | 14 | 29 | 31 | 159 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest recall (0.783) and F1 (0.631): it caught 47 of the 60 bad applicants, at the price of 42 false alarms. Strong regularisation (C = 0.1) was chosen, and its AUC (0.809) is close to the SVM's, which suggests the one-hot encoded data is largely linearly separable. The largest coefficients are intuitive: a negative checking balance and low savings push towards "bad", while having no checking account or a "critical account" history push towards "good" (a known quirk of this dataset). It has the lowest UCI cost of all models (107). |
| Decision Tree | Weakest model on every metric (accuracy 0.570, AUC 0.599, MCC 0.179). Its cross-validation F1 (0.564) was well above its test F1 (0.482), which points to high variance: with only 800 rows and 61 mostly binary inputs, a single tree's splits are unstable. Balanced class weights made it flag 66 good applicants as bad. An ensemble such as Random Forest would be the natural fix, but was outside the required model list. |
| KNN | Shows why accuracy is misleading here. Its accuracy (0.725) equals Logistic Regression's, yet recall is only 0.350: it missed 39 of 60 bad applicants, giving the highest cost (211). KNN has no class weighting, so the 70 % majority dominates neighbourhoods, and Euclidean distance over 61 dimensions (54 of them one-hot) carries little signal. |
| Naive Bayes | Middle of the pack (AUC 0.743, MCC 0.341, recall 0.650). Its assumptions are violated: duration and credit amount are correlated (r = 0.63), and Gaussian likelihoods are a poor fit for 0/1 one-hot columns. A larger var_smoothing (1e-3) helped by damping over-confident per-feature likelihoods. Fast and simple, but outperformed by the linear models. |
| Support Vector Machine | Best accuracy (0.785), AUC (0.820), precision (0.689) and MCC (0.457), so it ranks applicants best overall. Calibrated probabilities with a 0.5 threshold make it conservative: recall drops to 0.517 and it misses 29 bad applicants, costing 159. The small gamma (0.01) gives a smooth, nearly linear boundary, consistent with Logistic Regression doing so well. Lowering its threshold would trade precision for recall. |
| **Overall winner for this dataset** | **Logistic Regression.** SVM edges it on MCC (0.457 vs 0.446) and AUC, but the gap is within noise for a 200-row test set with 60 positives. For credit scoring, missing a defaulter is the expensive error, and Logistic Regression catches far more of them (recall 0.783 vs 0.517), has the lowest UCI cost (107 vs 159), and is interpretable through its coefficients, which matters for lending decisions. SVM is a close second. |

## Streamlit app features

- **CSV upload** of test data (or one-click use of the bundled `test_data.csv`, which can also be downloaded from the sidebar)
- **Upload validation**: missing columns, non-numeric values or a target not coded 0/1 are reported clearly
- **Model selection dropdown** for the five saved models
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion matrix and classification report** for the selected model
- **Comparison tab**: all five models side by side, including UCI cost, with ROC curves and a bar chart
- **Row-level predictions** with predicted probability of bad risk, downloadable as CSV

The app only loads the saved pipelines and calls `predict` / `predict_proba`; no model is fitted on uploaded data.
