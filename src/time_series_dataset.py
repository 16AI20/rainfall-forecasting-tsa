"""
time_series_dataset.py

Defines a PyTorch Dataset for supervised learning on time series data.

Each sample consists of:
- A sequence of `seq_len` past time steps as input (X)
- A sequence of `pred_len` future target values as output (y)
"""

from typing import Tuple

import torch
from torch.utils.data import Dataset
import pandas as pd


class TimeSeriesDataset(Dataset):
    """
    A PyTorch-compatible dataset for time series forecasting.

    Generates input-output pairs for supervised learning:
    - Input: past sequence of features (length = `seq_len`)
    - Output: future sequence of the target (length = `pred_len`)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        seq_len: int = 168,
        pred_len: int = 24
    ) -> None:
        """
        Args:
            df: Preprocessed DataFrame with both features and target columns.
            target: Name of the column to forecast.
            seq_len: Number of time steps in input sequence.
            pred_len: Number of future steps to predict.
        """
        self.df = df.copy()
        self.target = target
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.features = df.columns
        self.X = self.df[self.features].values
        self.y = self.df[target].values

    def __len__(self) -> int:
        """
        Returns:
            Number of samples in the dataset.
        """
        return len(self.df) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (input_sequence, target_sequence) as float32 tensors.
        """
        x = self.X[idx : idx + self.seq_len]
        y = self.y[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
