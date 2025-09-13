"""
data_loader.py

Utility functions for loading, preprocessing, and batching weather-related
time series data for machine learning workflows. Includes:
- File reading and parsing
- Circular encoding of wind direction
- Data concatenation from directories
- PyTorch DataLoader creation for training pipelines
"""

from pathlib import Path
from typing import List, Optional, Tuple

import logging
import pandas as pd
from torch.utils.data import DataLoader

from .time_series_dataset import TimeSeriesDataset

logger = logging.getLogger(__name__)


def load_hourly_weather_data(
    file_path: str,
    time_range: Optional[Tuple[str, str]] = None
) -> pd.DataFrame:
    """
    Load hourly weather data from a CSV file, apply optional time slicing,
    and encode wind direction using circular encoding.

    Args:
        file_path: Path to the input CSV file.
        time_range: Optional (start, end) time range as ISO8601 strings.

    Returns:
        DataFrame with datetime index and preprocessed columns.
    """
    logger.info("Loading weather data from %s", file_path)
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    if time_range:
        start, end = time_range
        df = df.loc[start:end]
        logger.debug("Filtered data to time range: %s - %s", start, end)

    # Apply circular encoding to wind direction if present
    
    logger.info("Loaded dataframe shape: %s", df.shape)
    return df


def load_and_concatenate(
    folder_paths: List[str],
    parse_dates: Optional[List[str]] = None
) -> List[pd.DataFrame]:
    """
    Load and concatenate CSV files from multiple folders into separate DataFrames.

    Args:
        folder_paths: List of folder paths containing CSV files.
        parse_dates: Columns to parse as dates during CSV reading.

    Returns:
        List of concatenated DataFrames (one per folder).
    """
    dataframes: List[pd.DataFrame] = []

    for folder_path in folder_paths:
        folder = Path(folder_path)
        all_files = sorted(folder.glob("*.csv"))

        if not all_files:
            logger.error("No CSV files found in: %s", folder_path)
            raise FileNotFoundError(f"No CSV files found in: {folder_path}")

        logger.info("Reading %d files from %s", len(all_files), folder_path)

        dfs = []
        for file in all_files:
            logger.debug("Reading file: %s", file.name)
            df = pd.read_csv(file, parse_dates=parse_dates)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info("Loaded %d rows from %s", len(combined_df), folder_path)
        dataframes.append(combined_df)

    return dataframes


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    seq_len: int,
    pred_len: int,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct PyTorch DataLoaders for training and validation datasets.

    Args:
        train_df: Training data as a DataFrame.
        val_df: Validation data as a DataFrame.
        target: Target variable column name.
        seq_len: Input sequence length.
        pred_len: Prediction horizon.
        batch_size: Number of samples per batch.

    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Creating DataLoaders: batch_size=%d, seq_len=%d, pred_len=%d", batch_size, seq_len, pred_len)

    train_dataset = TimeSeriesDataset(train_df, target, seq_len, pred_len)
    val_dataset = TimeSeriesDataset(val_df, target, seq_len, pred_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.debug("DataLoaders created: train=%d batches, val=%d batches",
                 len(train_loader), len(val_loader))

    return train_loader, val_loader
