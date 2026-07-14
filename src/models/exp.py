from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple
import dagshub
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import pandas as pd
import seaborn as sns
import xgboost as xgb
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import LabelEncoder
from mlflow.models import infer_signature

try:
    import dagshub  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    dagshub = None


LOGGER_NAME = "experiment"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure console and file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(log_dir / "exp.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def load_params(path: Path) -> Dict[str, Any]:
    """Load YAML configuration and validate the expected experiment sections."""
    with path.open("r", encoding="utf-8") as file_obj:
        params = yaml.safe_load(file_obj) or {}

    required_sections = {"experiment", "xgboost"}
    missing_sections = [section for section in required_sections if section not in params or not isinstance(params[section], dict)]
    if missing_sections:
        raise KeyError(f"Missing or invalid section(s) in params.yaml: {', '.join(missing_sections)}")

    return params


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target column."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' was not found in the dataset.")

    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y



def train_and_track(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    label_encoder: LabelEncoder,
    params: Dict[str, Any],
    logger: logging.Logger,
    project_root: Path,
) -> xgb.XGBClassifier:
    """Train an XGBoost model, log metrics, and save local artifacts."""
    experiment_name = params["experiment"]["name"]
    run_name = params["experiment"]["run_name"]

    xgb_params = params["xgboost"]
    target_model_dir = project_root / "trained_model"
    reports_dir = project_root / "reports"
    model_path = target_model_dir / "xgboost_model.pkl"
    classification_report_path = reports_dir / "classification_report.csv"
    confusion_matrix_path = reports_dir / "confusion_matrix.png"
    feature_importance_path = reports_dir / "feature_importance.png"

    target_model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(experiment_name)

    logger.info("Starting MLflow run '%s' under experiment '%s'.", run_name, experiment_name)
    with mlflow.start_run(run_name=run_name):
        model = xgb.XGBClassifier(**xgb_params)

        logger.info("Training model on %d rows and %d features.", x_train.shape[0], x_train.shape[1])
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_test_labels = label_encoder.inverse_transform(pd.Series(y_test).astype(int))
        y_pred_labels = label_encoder.inverse_transform(pd.Series(y_pred).astype(int))

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        logger.info("Logging parameters and metrics to MLflow.")
        mlflow.log_params(
            xgb_params
        )
        mlflow.log_metrics(metrics)

        report = classification_report(y_test_labels, y_pred_labels, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(classification_report_path, index=True)
        mlflow.log_artifact(str(classification_report_path), artifact_path="reports")

        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(confusion_matrix_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(confusion_matrix_path), artifact_path="reports")

        plt.figure(figsize=(10, 6))
        xgb.plot_importance(model)
        plt.tight_layout()
        plt.savefig(feature_importance_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(feature_importance_path), artifact_path="reports")

        signature_input = x_train.astype(float)
        signature = infer_signature(signature_input, model.predict(x_train))
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="xgboost_model",
            signature=signature,
            input_example=signature_input.head(),
        )

        with model_path.open("wb") as model_file:
            pickle.dump(model, model_file)

        logger.info("Saved trained model to %s", model_path)
        return model


mlflow.set_tracking_uri("https://dagshub.com/Rahulkumar1029/test_deployment.mlflow")
dagshub.init(repo_owner='Rahulkumar1029', repo_name='test_deployment', mlflow=True)

def main() -> None:
    """Main entry point for training and experiment tracking."""
    project_root = Path(__file__).resolve().parents[2]
    logger = setup_logging(project_root / "reports" / "logs")

    try:
        params = load_params(project_root / "params.yaml")

        dataset_path = project_root / "data" / "processed" / "train_processed.csv"
        logger.info("Loading processed dataset from %s", dataset_path)
        df = load_csv(dataset_path)

        x, y = split_features_target(
            df,
            target_column="Segmentation",
        )
        label_encoder = LabelEncoder()
        y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index, name=y.name)
        x_train, x_test, y_train, y_test = sklearn_train_test_split(
            x,
            y_encoded,
            test_size=float(params.get("test_size", 0.2)),
            random_state=42,
            stratify=y_encoded,
        )

        train_and_track(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            label_encoder=label_encoder,
            params=params,
            logger=logger,
            project_root=project_root,
        )

        logger.info("Experiment tracking completed successfully.")
    except Exception:
        logger.exception("Error in main execution.")
        raise


if __name__ == "__main__":
    main()
