"""Train and save the dengue classification model.

Run from the repository root with: python train.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "Data" / "dataset.csv"
CLEAN_DATA = ROOT / "Data" / "dataset_cleaned.csv"
MODEL_PATH = ROOT / "Models" / "best_dengue_model.joblib"


def clean_data() -> pd.DataFrame:
    """Clean the raw data and persist the reusable cleaned dataset."""
    df = pd.read_csv(RAW_DATA).drop_duplicates().copy()
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype(str).str.strip()
    for column in ["Gender", "Area", "AreaType", "HouseType", "District", "Joint_Pain"]:
        if column in df:
            df[column] = df[column].str.title()
    df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce")
    df = df.dropna().copy()
    df["Outcome_Label"] = df["Outcome"].map({0: "Negative", 1: "Positive"})
    CLEAN_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA, index=False)
    return df


def build_candidates(preprocessor: ColumnTransformer) -> dict[str, dict]:
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"model__C": [0.1, 1, 10], "model__solver": ["liblinear", "lbfgs"]},
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "params": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_leaf": [1, 2],
            },
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "params": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0],
            },
        },
        "svc": {
            "estimator": SVC(probability=True, random_state=42),
            "params": {
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", "auto"],
                "model__kernel": ["rbf", "linear"],
            },
        },
        "mlp_classifier": {
            "estimator": MLPClassifier(random_state=42, max_iter=500),
            "params": {
                "model__hidden_layer_sizes": [(32,), (64,), (32, 16)],
                "model__learning_rate_init": [0.001, 0.01],
                "model__alpha": [0.0001, 0.001],
                "model__activation": ["relu", "tanh"],
            },
        },
    }


def train_model() -> tuple[Pipeline, dict]:
    df = clean_data()
    feature_columns = [c for c in df.columns if c not in {"Outcome", "Outcome_Label"}]
    X, y = df[feature_columns], df["Outcome"].astype(int)
    categorical = X.select_dtypes(include="object").columns.tolist()
    numeric = X.select_dtypes(exclude="object").columns.tolist()
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results, fitted = [], {}
    for name, config in build_candidates(preprocessor).items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", config["estimator"])])
        search = GridSearchCV(pipeline, config["params"], cv=5, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        predictions = search.best_estimator_.predict(X_test)
        probabilities = search.best_estimator_.predict_proba(X_test)[:, 1]
        result = {
            "Model": name,
            "Best_CV_F1": round(search.best_score_, 4),
            "Test_Accuracy": round(accuracy_score(y_test, predictions), 4),
            "Test_Precision": round(precision_score(y_test, predictions, zero_division=0), 4),
            "Test_Recall": round(recall_score(y_test, predictions, zero_division=0), 4),
            "Test_F1": round(f1_score(y_test, predictions, zero_division=0), 4),
            "ROC_AUC": round(roc_auc_score(y_test, probabilities), 4),
            "Best_Params": search.best_params_,
        }
        results.append(result)
        fitted[name] = search.best_estimator_
        print(f"{name}: CV F1={result['Best_CV_F1']}, test F1={result['Test_F1']}, ROC AUC={result['ROC_AUC']}")

    results.sort(key=lambda row: (row["Test_F1"], row["Best_CV_F1"]), reverse=True)
    best_name = results[0]["Model"]
    bundle = {
        "model": fitted[best_name],
        "feature_columns": feature_columns,
        "model_name": best_name,
        "metrics": results[0],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    (MODEL_PATH.with_suffix(".metrics.json")).write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"Best model: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    return fitted[best_name], bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    train_model()
