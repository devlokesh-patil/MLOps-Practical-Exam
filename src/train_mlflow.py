import argparse
import inspect
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_dataset(input_csv: str | None = None, target_col: str = "target"):
    """Load Digits dataset from a processed/feature CSV if provided, else sklearn load_digits."""
    if input_csv:
        df = pd.read_csv(input_csv)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in {input_csv}")
        drop_cols = [target_col]
        for col in ("event_timestamp", "digit_id"):
            if col in df.columns:
                drop_cols.append(col)
        X = df.drop(columns=drop_cols)
        y = df[target_col]
        return X, y, f"csv:{input_csv}"

    digits = load_digits()
    feature_names = [f"pixel_{i}" for i in range(digits.data.shape[1])]
    X = pd.DataFrame(digits.data, columns=feature_names)
    y = pd.Series(digits.target, name=target_col)
    return X, y, "sklearn:load_digits"


def build_candidate_models(random_state: int = 42) -> dict[str, tuple[object, dict]]:
    """Return candidate classification pipelines and parameter dicts to log in MLflow."""
    return {
        "logistic_regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)),
            ]),
            {"model_type": "LogisticRegression", "max_iter": 1000, "C": 1.0, "scaler": "StandardScaler"},
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=random_state,
                n_jobs=-1,
            ),
            {"model_type": "RandomForestClassifier", "n_estimators": 200, "max_depth": "None"},
        ),
        "svm_rbf": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(C=5.0, gamma="scale", kernel="rbf", random_state=random_state)),
            ]),
            {"model_type": "SVC", "kernel": "rbf", "C": 5.0, "gamma": "scale", "scaler": "StandardScaler"},
        ),
    }


def log_sklearn_model_safely(model, model_name: str = "model"):
    """Log a scikit-learn model across MLflow 3.x versions, trusting tree nodes when needed."""
    sig = inspect.signature(mlflow.sklearn.log_model)
    kwargs = {}
    if "name" in sig.parameters:
        kwargs["name"] = model_name
    elif "artifact_path" in sig.parameters:
        kwargs["artifact_path"] = model_name

    if "skops_trusted_types" in sig.parameters:
        kwargs["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]

    mlflow.sklearn.log_model(model, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Practical 6: MLflow Experiment Tracking & Model Selection")
    parser.add_argument("--input", type=str, default=None, help="Optional path to preprocessed/feature CSV")
    parser.add_argument("--target-col", type=str, default="target", help="Target column name")
    parser.add_argument("--experiment-name", type=str, default="digits-classification-mlflow")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--best-model-path", type=str, default="models/best_digits_model.joblib")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = mlflow.set_experiment(args.experiment_name)

    X, y, source_label = load_dataset(args.input, args.target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    candidates = build_candidate_models(random_state=args.random_state)
    results = []
    artifacts_dir = Path("reports/mlflow_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {args.experiment_name} (ID: {experiment.experiment_id})")
    print(f"Tracking URI: {args.tracking_uri} | Data source: {source_label}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")

    for run_name, (model, params) in candidates.items():
        with mlflow.start_run(run_name=run_name) as run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
                "precision_weighted": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
                "recall_weighted": float(recall_score(y_test, preds, average="weighted", zero_division=0)),
            }

            mlflow.log_params({
                **params,
                "data_source": source_label,
                "test_size": args.test_size,
                "random_state": args.random_state,
                "n_features": int(X.shape[1]),
            })
            mlflow.log_metrics(metrics)

            report_txt = classification_report(y_test, preds, digits=4)
            cm_list = confusion_matrix(y_test, preds).tolist()

            report_file = artifacts_dir / f"{run_name}_classification_report.txt"
            cm_file = artifacts_dir / f"{run_name}_confusion_matrix.json"
            report_file.write_text(report_txt, encoding="utf-8")
            cm_file.write_text(json.dumps({"confusion_matrix": cm_list}, indent=2), encoding="utf-8")

            mlflow.log_artifact(str(report_file), artifact_path="evaluations")
            mlflow.log_artifact(str(cm_file), artifact_path="evaluations")
            log_sklearn_model_safely(model, model_name="model")

            results.append({
                "run_name": run_name,
                "run_id": run.info.run_id,
                "model": model,
                **metrics,
            })
            print(
                f"[{run_name}] run_id={run.info.run_id[:8]} | "
                f"accuracy={metrics['accuracy']:.4f} | f1_weighted={metrics['f1_weighted']:.4f}"
            )

    results_sorted = sorted(
        results,
        key=lambda r: (r["f1_weighted"], r["accuracy"]),
        reverse=True,
    )
    best = results_sorted[0]

    best_model_file = Path(args.best_model_path)
    best_model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], best_model_file)

    summary_df = pd.DataFrame([
        {k: v for k, v in row.items() if k != "model"}
        for row in results_sorted
    ])
    summary_path = Path("reports/mlflow_model_comparison.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n--- Model Comparison (Ranked by f1_weighted, accuracy) ---")
    print(summary_df.to_string(index=False))
    print(f"\nSelected Best Model: {best['run_name']} (f1_weighted={best['f1_weighted']:.4f}, accuracy={best['accuracy']:.4f})")
    print(f"Saved best model artifact to: {best_model_file}")


if __name__ == "__main__":
    main()
