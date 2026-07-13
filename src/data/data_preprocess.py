from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


LOGGER_NAME = "data_preprocess"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure console and file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(log_dir / "data_preprocess.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_csv(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load a CSV file with clear error reporting."""
    try:
        logger.info("Loading file: %s", path)
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        logger.exception("File not found: %s", path)
        raise
    except pd.errors.EmptyDataError as exc:
        logger.exception("CSV file is empty: %s", path)
        raise
    except Exception:
        logger.exception("Unexpected error while reading: %s", path)
        raise


def identify_categorical_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Return categorical columns excluding the specified columns."""
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return [column for column in categorical_columns if column not in exclude]


def split_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: List[str],
) -> Tuple[List[str], List[str]]:
    """Split categorical columns into binary and multi-class columns."""
    binary_columns: List[str] = []
    multi_class_columns: List[str] = []

    for column in categorical_columns:
        unique_values = df[column].dropna().nunique()
        if unique_values <= 2:
            binary_columns.append(column)
        else:
            multi_class_columns.append(column)

    return binary_columns, multi_class_columns


def encode_binary_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    binary_columns: List[str],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map binary categorical columns to 0/1 using train-set categories."""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for column in binary_columns:
        categories = [value for value in train_encoded[column].dropna().unique()]
        mapping: Dict[str, int] = {category: index for index, category in enumerate(categories[:2])}

        logger.info("Binary encoding column '%s' with mapping: %s", column, mapping)
        train_encoded[column] = train_encoded[column].map(mapping)
        test_encoded[column] = test_encoded[column].map(mapping)

        train_encoded[column] = train_encoded[column].astype("Int64")
        test_encoded[column] = test_encoded[column].astype("Int64")

    return train_encoded, test_encoded


def one_hot_encode_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: List[str],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode the supplied columns and align train/test features."""
    if not columns:
        logger.info("No multi-class categorical columns found for one-hot encoding.")
        return train_df, test_df

    logger.info("Applying one-hot encoding to columns: %s", columns)

    original_train_columns = train_df.columns.tolist()
    train_encoded = pd.get_dummies(train_df, columns=columns, drop_first=False)
    test_encoded = pd.get_dummies(test_df, columns=columns, drop_first=False)

    encoded_dummy_columns = [column for column in train_encoded.columns if column not in original_train_columns]

    if encoded_dummy_columns:
        train_encoded[encoded_dummy_columns] = train_encoded[encoded_dummy_columns].astype(int)
        test_encoded[encoded_dummy_columns] = test_encoded[encoded_dummy_columns].astype(int)

    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    if encoded_dummy_columns:
        test_encoded[encoded_dummy_columns] = test_encoded[encoded_dummy_columns].astype(int)

    return train_encoded, test_encoded


def preprocess_data(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full preprocessing pipeline."""
    train_df = load_csv(train_path, logger)
    test_df = load_csv(test_path, logger)

    if "ID" not in train_df.columns or "ID" not in test_df.columns:
        raise KeyError("Expected 'ID' column to be present in both train and test data.")

    logger.info("Dropping rows with missing values from train data")
    train_df = train_df.dropna().reset_index(drop=True)

    target_column = "Segmentation" if "Segmentation" in train_df.columns else None
    train_target = train_df[target_column].copy() if target_column else None
    train_features = train_df.drop(columns=[target_column]) if target_column else train_df.copy()
    test_features = test_df.copy()

    exclude_columns = ["ID"]
    categorical_columns = identify_categorical_columns(train_features, exclude=exclude_columns)
    logger.info("Detected categorical columns: %s", categorical_columns)

    binary_columns, multi_class_columns = split_categorical_columns(train_features, categorical_columns)
    logger.info("Binary categorical columns: %s", binary_columns)
    logger.info("Multi-class categorical columns: %s", multi_class_columns)

    train_features, test_features = encode_binary_columns(train_features, test_features, binary_columns, logger)
    train_features, test_features = one_hot_encode_columns(train_features, test_features, multi_class_columns, logger)

    logger.info("Dropping ID column from train and test data")
    train_features = train_features.drop(columns=["ID"])
    test_features = test_features.drop(columns=["ID"])

    if target_column:
        train_df = train_features.copy()
        train_df[target_column] = train_target
    else:
        train_df = train_features

    if target_column and target_column in train_df.columns:
        # Keep the target as the last column for easier downstream use.
        train_df = train_df[[column for column in train_df.columns if column != target_column] + [target_column]]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_output_path = output_dir / "train_processed.csv"
    test_output_path = output_dir / "test_processed.csv"

    logger.info("Saving processed train data to %s", train_output_path)
    train_df.to_csv(train_output_path, index=False)
    logger.info("Saving processed test data to %s", test_output_path)
    test_features.to_csv(test_output_path, index=False)

    logger.info("Preprocessing completed successfully")
    return train_df, test_features


def main() -> None:
    """Entry point for script execution."""
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    logger = setup_logging(processed_dir / "logs")

    try:
        preprocess_data(
            train_path=raw_dir / "Train.csv",
            test_path=raw_dir / "Test.csv",
            output_dir=processed_dir,
            logger=logger,
        )
    except Exception as exc:
        logger.exception("Preprocessing failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
