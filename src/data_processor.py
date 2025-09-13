"""
feature_processing.py

Utilities to process weather feature CSVs by:
- Aggregating readings to hourly resolution
- Handling missing data via interpolation
- Merging multiple feature files on timestamp
- Performing time-based train/test splits
"""

import os
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig

from .data_loader import load_hourly_weather_data

# Initialize module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def process_feature_folder(
    folder_path: str,
    output_path: str,
    feature: str,
    interpolate_missing_data: bool = False
) -> None:
    """
    Process all CSVs in a feature folder: aggregate hourly, interpolate, and save.

    Args:
        folder_path: Path to the folder containing raw CSVs for a specific feature.
        output_path: Path to store processed hourly-aggregated CSV.
        feature: Name of the weather feature (e.g., "rainfall").
        interpolate_missing_data: Whether to interpolate missing hourly data.
    """
    folder = Path(folder_path)
    all_dfs = []

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in: %s", folder_path)
        return

    for file in csv_files:
        try:
            logger.info("Reading file: %s", file.name)
            df = pd.read_csv(file, parse_dates=["timestamp"])

            if "reading_value" not in df.columns:
                logger.warning("Skipping file (no 'reading_value'): %s", file.name)
                continue

            df = df[["timestamp", "reading_value"]].copy()
            df["reading_value"] = pd.to_numeric(df["reading_value"], errors="coerce")
            df.dropna(subset=["reading_value"], inplace=True)

            df["timestamp"] = df["timestamp"].dt.floor("h")
            df_hourly = df.groupby("timestamp")["reading_value"].mean().reset_index()

            all_dfs.append(df_hourly)
        except Exception as e:
            logger.error("Failed to process %s: %s", file.name, e, exc_info=True)

    if not all_dfs:
        logger.warning("No valid data found in %s", folder_path)
        return

    combined = pd.concat(all_dfs).sort_values("timestamp")
    combined = combined.groupby("timestamp")["reading_value"].mean().reset_index()
    combined.set_index("timestamp", inplace=True)

    combined = combined.resample("h").mean()

    if interpolate_missing_data:
        logger.info("Interpolating missing values for %s", feature)
        combined.interpolate(method="time", inplace=True)
        combined.bfill(inplace=True)
        combined.ffill(inplace=True)

    output_file = Path(output_path) / f"{feature}_hourly.csv"
    combined.reset_index().rename(columns={"reading_value": feature}).to_csv(output_file, index=False)
    logger.info("Saved hourly aggregated data: %s", output_file)


def merge_hourly_features(input_dir: str, output_file: str) -> None:
    """
    Merge all *_hourly.csv files in a directory on 'timestamp' column.

    Args:
        input_dir: Directory containing feature-wise hourly CSVs.
        output_file: Path to the final merged CSV.
    """
    input_path = Path(input_dir)
    all_files = sorted(input_path.glob("*_hourly.csv"))

    dfs = []

    for file in all_files:
        logger.info("Reading %s", file.name)
        df = pd.read_csv(file, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)

        feature_name = file.stem.replace("_hourly", "")
        df.rename(columns={df.columns[0]: feature_name}, inplace=True)
        dfs.append(df)

    if not dfs:
        logger.warning("No CSV files found for merging.")
        return

    merged = pd.concat(dfs, axis=1)
    merged.reset_index().to_csv(output_file, index=False)
    logger.info("Merged data saved to: %s", output_file)


def train_test_split(
    df: pd.DataFrame,
    split_date: str = "2024-01-01"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed DataFrame into training and testing sets.

    Args:
        df: Time-indexed DataFrame to split.
        split_date: Date (YYYY-MM-DD) to split at.

    Returns:
        Tuple of (train_df, test_df).
    """
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    train_df = df.loc[df.index < split_date]
    test_df = df.loc[df.index >= split_date]

    logger.info("Split data at %s: %d train, %d test rows",
                split_date, len(train_df), len(test_df))
    return train_df, test_df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using time-based interpolation.

    Args:
        df: DataFrame with potential missing values.

    Returns:
        DataFrame with interpolated values.
    """
    logger.debug("Interpolating missing values")
    df_interpolated = df.copy()
    df_interpolated = df_interpolated.interpolate(method="time")
    return df_interpolated


if __name__ == "__main__":
    base_path = "./data"  # Replace with your actual folder path
    output_path = "aggregated_output"
    os.makedirs(output_path, exist_ok=True)

    features = ["humidity", "rainfall", "temperature", "wind_direction", "wind_speed"]
    for feature in features:
        process_feature_folder(
            folder_path=os.path.join(base_path, feature),
            output_path=output_path,
            feature=feature,
            interpolate_missing_data=True  # Enable interpolation by default
        )

    merge_hourly_features(output_path, os.path.join(output_path, "all_features_hourly.csv"))

def load_and_prepare_df(cfg: DictConfig) -> pd.DataFrame:
    """
    Loads raw data, applies missing value imputation, and encodes wind direction.

    Args:
        cfg (DictConfig): Hydra configuration object containing `data.file_path`.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    logger.info("Loading hourly weather data...")
    df = load_hourly_weather_data(cfg.data.file_path)

    logger.info("Imputing missing values...")
    df = impute_missing_values(df)

    logger.info("Encoding wind direction...")
    df = encode_wind_direction(df)

    return df


def encode_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts wind direction in degrees into sine and cosine components.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'wind_direction' column.

    Returns:
        pd.DataFrame: DataFrame with 'wind_dir_sin' and 'wind_dir_cos' added,
                      and original 'wind_direction' dropped.
    """
    if "wind_direction" in df.columns:
        logger.debug("Encoding wind direction using sine and cosine.")
        df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction"]))
        df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction"]))
        df.drop("wind_direction", axis=1, inplace=True)
    else:
        logger.warning("Column 'wind_direction' not found; skipping encoding.")
    return df


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    use_scaling: bool
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler | None]:
    """
    Applies StandardScaler to features (excluding target) if `use_scaling` is True.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.
        target (str): Name of target column to exclude from scaling.
        use_scaling (bool): Whether to apply standard scaling to features.

    Returns:
        tuple: Scaled train_df, val_df, and the fitted StandardScaler instance or None.
    """
    feature_cols = train_df.columns
    if not use_scaling:
        logger.info("Skipping feature scaling.")
        return train_df, val_df, None

    logger.info("Fitting StandardScaler on training features...")
    feature_scaler = StandardScaler().fit(train_df.drop(columns=[target]))

    for df in [train_df, val_df]:
        features = df.drop(columns=[target])
        logger.debug("Applying feature scaler to dataset.")
        df.loc[:, features.columns] = feature_scaler.transform(features)

    return train_df, val_df, feature_scaler


def log_transform_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies log1p transformation to the target variable in both train and validation sets.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.
        target (str): Name of target column to transform.

    Returns:
        tuple: Log-transformed train_df and val_df.
    """
    logger.info(f"Applying log1p transform to target column '{target}'...")
    train_df[target] = np.log1p(train_df[target])
    val_df[target] = np.log1p(val_df[target])
    return train_df, val_df
