"""
Train and evaluate 5 classifiers on the UCI "Predict Students' Dropout and
Academic Success" dataset, then save the models + a held-out test CSV.

Run:  python train_models.py
Outputs:
    model/<name>.joblib          one fitted sklearn Pipeline per classifier
    model/feature_columns.json   column order expected by the models
    model/metrics_summary.csv    comparison table (used in README + app)
    test_data.csv                20 % stratified hold-out (raw features + Target)
"""

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from features import CLASS_ORDER, TARGET_COL, split_features_target, tidy_column_names

warnings.filterwarnings("ignore", category=FutureWarning)  # SVC probability deprecation notice

SEED = 3621
ROOT = Path(__file__).parent
RAW_CSV = ROOT / "data" / "students_raw.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)


def build_search_spaces():
    """Each model is a Pipeline (scaler + estimator) with a small grid to tune."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    spaces = {
        "Logistic Regression": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, random_state=SEED)),
            ]),
            {"clf__C": [0.05, 0.3, 1.0, 3.0], "clf__class_weight": [None, "balanced"]},
        ),
        "Decision Tree": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
            {
                "clf__max_depth": [4, 6, 8, 12],
                "clf__min_samples_leaf": [5, 15, 30],
                "clf__criterion": ["gini", "entropy"],
            },
        ),
        "KNN": (
            Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {
                "clf__n_neighbors": [7, 15, 25, 35],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
        "Naive Bayes": (
            Pipeline([("scale", StandardScaler()), ("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
        ),
        "Support Vector Machine": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=SEED)),
            ]),
            {"clf__C": [0.5, 1.0, 3.0], "clf__gamma": ["scale", 0.01]},
        ),
    }
    return spaces, cv


def evaluate(model, X, y):
    """All six required metrics. Multi-class -> weighted averages, AUC = one-vs-rest."""
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="weighted",
                             labels=list(model.classes_)),
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


def main():
    raw = tidy_column_names(pd.read_csv(RAW_CSV, sep=";"))
    print(f"Raw dataset: {raw.shape[0]} rows x {raw.shape[1] - 1} features")
    print(raw[TARGET_COL].value_counts(), "\n")

    # Hold out 20 % as the "test data" that will be uploaded in the app
    train_raw, test_raw = train_test_split(
        raw, test_size=0.20, stratify=raw[TARGET_COL], random_state=SEED
    )
    test_raw.to_csv(ROOT / "test_data.csv", index=False)

    X_train, y_train = split_features_target(train_raw)
    X_test, y_test = split_features_target(test_raw)
    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(list(X_train.columns), indent=2)
    )

    spaces, cv = build_search_spaces()
    rows = []
    for name, (pipe, grid) in spaces.items():
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        scores = evaluate(best, X_test, y_test)
        rows.append({"ML Model Name": name, **scores,
                     "Best Params": json.dumps(search.best_params_)})
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(best, MODEL_DIR / fname)
        print(f"{name:<24} CV-F1={search.best_score_:.4f}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))
        print(f"{'':<24} best params: {search.best_params_}")

    summary = pd.DataFrame(rows)
    summary.to_csv(MODEL_DIR / "metrics_summary.csv", index=False)
    print("\n" + summary.drop(columns="Best Params").round(4).to_markdown(index=False))


if __name__ == "__main__":
    main()
