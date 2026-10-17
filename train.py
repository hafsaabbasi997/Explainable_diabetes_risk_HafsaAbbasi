"""
train.py - Explainable Diabetes Prediction | 2026 Edition
==========================================================
Dataset : PIMA Indians Diabetes (768 patients, all 8 clinical features)
Model   : XGBoost (XGBClassifier) only
Pipeline: train_test_split -> median imputation -> SMOTE (train only) -> XGBoost (GridSearchCV)
XAI     : SHAP TreeExplainer (global summary plot + waterfall for highest-risk patient)

Outputs (all written to ./assets/):
    model.pkl             best XGBoost model (pickle)
    model.json            best XGBoost model (native JSON, version-portable)
    imputer_medians.json  training-set medians (default values for the app)
    confusion_matrix.png
    roc_curve.png
    shap_summary.png
    shap_waterfall.png
    metrics.json          CV ROC-AUC, Test ROC-AUC and other metrics
    results_table.csv     metrics as a table

Run in Google Colab:  upload this file, then  `!python train.py`
"""
import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

# Colab convenience: install anything that is missing.
for _import_name, _pip_name in [
    ("imblearn", "imbalanced-learn"),
    ("shap", "shap"),
    ("xgboost", "xgboost"),
]:
    if importlib.util.find_spec(_import_name) is None:
        print(f"Installing {_pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pip_name])

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE                 # 2026 Edition: SMOTE for imbalance
from imblearn.pipeline import Pipeline as ImbPipeline    # resampling inside CV = no leakage
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
# All 8 features are used: Pregnancies, Glucose, BloodPressure, SkinThickness,
# Insulin, BMI, DiabetesPedigreeFunction, Age
FEATURES = COLUMNS[:-1]
TARGET = "Outcome"
# A 0 in these columns is a missing measurement (a living patient cannot have 0 glucose
# or 0 BMI). They are set to NaN and replaced with the MEDIAN (computed on train only).
# Pregnancies and Age zeros are valid values and are left untouched.
ZERO_AS_NAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

ASSETS = Path("assets")
ASSETS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ----------------------------------------------------------------------- #
    # 1) Load data; mark impossible zeros as missing (NaN)
    # ----------------------------------------------------------------------- #
    df = pd.read_csv(DATA_URL, header=None, names=COLUMNS)
    df[ZERO_AS_NAN] = df[ZERO_AS_NAN].replace(0, np.nan)
    print(f"Loaded {len(df)} patients | diabetes prevalence = {df[TARGET].mean():.1%}")
    print("Missing values after zero->NaN:\n", df[ZERO_AS_NAN].isna().sum().to_string(), "\n")

    X, y = df[FEATURES], df[TARGET]

    # ----------------------------------------------------------------------- #
    # 2) DATA-LEAKAGE FIX - correct order of operations:
    #       (a) train_test_split FIRST (80/20, stratified, random_state=42)
    #       (b) median imputation + SMOTE are fitted on X_train ONLY
    #    The test set is never used to compute medians or to create synthetic samples.
    # ----------------------------------------------------------------------- #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Illustration only: effect of SMOTE on the TRAIN set's class balance.
    _imp_demo = SimpleImputer(strategy="median").fit_transform(X_train)
    _, y_demo = SMOTE(random_state=RANDOM_STATE).fit_resample(_imp_demo, y_train)
    print("Train class counts BEFORE SMOTE:", y_train.value_counts().to_dict())
    print("Train class counts AFTER  SMOTE:", pd.Series(y_demo).value_counts().to_dict(), "\n")

    # ----------------------------------------------------------------------- #
    # 3) Pipeline: median imputation -> SMOTE -> XGBoost
    #    2026 Edition: SMOTE for imbalance. SMOTE is applied ONLY to X_train (and, inside
    #    GridSearchCV, only to the training part of each CV fold, so validation folds stay
    #    real data). It is automatically skipped when predicting on X_test.
    # ----------------------------------------------------------------------- #
    pipeline = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("xgb", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=1,  # parallelism is handled by GridSearchCV below
            )),
        ]
    )

    param_grid = {
        "xgb__n_estimators": [100, 200, 300],
        "xgb__max_depth": [3, 4, 5],
        "xgb__learning_rate": [0.01, 0.05, 0.1],
    }
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid.fit(X_train, y_train)  # only the training data is ever seen here

    best_pipe = grid.best_estimator_
    best_params = {k.replace("xgb__", ""): v for k, v in grid.best_params_.items()}
    cv_auc = float(grid.best_score_)
    print(f"\nBest params : {best_params}")
    print(f"CV ROC-AUC  : {cv_auc:.4f}")

    # ----------------------------------------------------------------------- #
    # 4) Evaluate once on the untouched test set
    # ----------------------------------------------------------------------- #
    proba = best_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_auc = float(roc_auc_score(y_test, proba))
    print(f"Test ROC-AUC: {test_auc:.4f}\n")
    print(classification_report(y_test, pred, target_names=["No diabetes", "Diabetes"]))

    metrics = {
        "cv_roc_auc": round(cv_auc, 4),
        "test_roc_auc": round(test_auc, 4),
        "test_pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "test_precision": round(float(precision_score(y_test, pred)), 4),
        "test_recall": round(float(recall_score(y_test, pred)), 4),
        "test_f1": round(float(f1_score(y_test, pred)), 4),
        "best_params": best_params,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    with open(ASSETS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(
        {
            "Metric": ["CV ROC-AUC", "Test ROC-AUC", "Test PR-AUC", "Accuracy",
                       "Precision", "Recall", "F1"],
            "Value": [metrics["cv_roc_auc"], metrics["test_roc_auc"], metrics["test_pr_auc"],
                      metrics["test_accuracy"], metrics["test_precision"],
                      metrics["test_recall"], metrics["test_f1"]],
        }
    ).to_csv(ASSETS / "results_table.csv", index=False)

    # ----------------------------------------------------------------------- #
    # 5) Plots: confusion matrix + ROC curve
    # ----------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No diabetes", "Diabetes"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title("Confusion matrix (held-out test set)")
    fig.tight_layout()
    fig.savefig(ASSETS / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(
        y_test, proba, name=f"XGBoost + SMOTE (AUC = {test_auc:.3f})", ax=ax
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_title("ROC curve (held-out test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSETS / "roc_curve.png", dpi=200)
    plt.close(fig)

    # ----------------------------------------------------------------------- #
    # 6) Explainable AI: SHAP with TreeExplainer
    #    SHAP must see the same (imputed) numbers the XGBoost model was trained on.
    # ----------------------------------------------------------------------- #
    imputer = best_pipe.named_steps["imputer"]
    xgb_model = best_pipe.named_steps["xgb"]
    X_test_imp = imputer.transform(X_test)

    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer(X_test_imp)
    explanation = shap.Explanation(
        values=sv.values,
        base_values=sv.base_values,
        data=X_test_imp,
        feature_names=FEATURES,
    )

    # Global explanation
    plt.figure()
    shap.summary_plot(explanation, X_test_imp, feature_names=FEATURES, show=False)
    plt.title("SHAP summary - global feature impact")
    plt.tight_layout()
    plt.savefig(ASSETS / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Local explanation: WHY is the highest-risk test patient high risk?
    # Cropping fix: figsize=(10, 6) is applied AFTER shap draws (shap resizes the figure
    # itself), and the PNG is saved with bbox_inches="tight" + padding so nothing is cut off.
    top_idx = int(np.argmax(proba))
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation[top_idx], max_display=len(FEATURES), show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig.suptitle(
        f"SHAP waterfall - highest-risk test patient (predicted risk {proba[top_idx]:.0%})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ASSETS / "shap_waterfall.png", dpi=200, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)

    # ----------------------------------------------------------------------- #
    # 7) Save the best XGBoost model in 2 formats
    #    - model.pkl  : pickle (convenient in Python, but sensitive to XGBoost versions)
    #    - model.json : XGBoost native format (portable across versions) -> app loads this first
    # ----------------------------------------------------------------------- #
    joblib.dump(xgb_model, ASSETS / "model.pkl")
    xgb_model.save_model(str(ASSETS / "model.json"))

    # Training-set medians (used by the app as default slider values)
    with open(ASSETS / "imputer_medians.json", "w") as f:
        json.dump({k: float(v) for k, v in zip(FEATURES, imputer.statistics_)}, f, indent=2)

    print("\nDone. Files in assets/:", sorted(p.name for p in ASSETS.iterdir()))


if __name__ == "__main__":
    main()
